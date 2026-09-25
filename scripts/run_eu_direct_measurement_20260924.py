#!/usr/bin/env python3
"""Frozen EU geometry versus three directly fitted measurement distributions."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from opal2.gram_oof_ridge import transform_input
from opal2.observable_quantile_distribution import (
    GRID, make_nonnegative_quantile_law, fit_quantile_offsets)

NAME = 'eu_measurement_direct_20260924_v1'
RUN = ROOT / 'runs' / NAME
REPORT = ROOT / 'reports' / NAME
R4 = ROOT / 'runs/r4_confirmation_20260921_v1'
PROTOCOL = ROOT / 'protocols/EU_direct_measurement_20260924.md'
DEV = ROOT / 'reports/eu_core_development_20260917_v1/prepared_data_cc904/data.npz'
TARGETS = ('average_Z1_Z2', 'average_Z1_V', 'average_Z2_V')
PARAMETERS = ((7, 20), (15, 10), (31, 10))
LEVELS = np.array([.5, .8, .9, .95, .99])
SEED = 20260921
BOOTSTRAPS = 10000


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def status(folder, state, **extra):
    record = dict(state=state, updated_at=now(), pid=os.getpid(), **extra)
    write_json(Path(folder) / 'status.json', record)
    return record


def configuration():
    return dict(experiment=NAME, sklearn_version='1.9.1', seed=SEED,
        targets=list(TARGETS), quantiles=GRID.tolist(),
        parameters=[dict(max_leaf_nodes=a, min_samples_leaf=b) for a, b in PARAMETERS],
        max_iter=200, learning_rate=.05, l2_regularization=1., max_bins=255,
        max_features=1., max_depth=None, early_stopping=False,
        primary='CAL minus CORE equal-target-weight CRPS', secondary='RAW minus CORE',
        selection='one setting per target using RAW VALIDATION CRPS; no CAL/query selection',
        tail='adjacent linear extrapolation to p=0/1; floor zero; no upper clipping',
        n_fits=171, n_train=615, n_valid=108, n_cal=181,
        n_predict=1527, n_complete=1520, levels=LEVELS.tolist(),
        bootstrap_repetitions=BOOTSTRAPS, bootstrap_seed=20260924,
        replay_seed=SEED, replay_samples=100000, replay_chunk_size=8,
        workers=2, threads_per_worker=1, post_hoc=True, original_lists_changed=False)


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if sklearn.__version__ != '1.9.1':
        raise RuntimeError('Use the existing scikit-learn 1.9.1 environment')
    if shutil.disk_usage(RUN).free < 2 * 1024**3:
        raise RuntimeError('Less than 2 GiB free; no existing artifacts removed')
    cfg = configuration()
    config_path = RUN / 'configuration.json'
    if config_path.exists() and read_json(config_path) != cfg:
        raise ValueError('Existing configuration differs; do not mix runs')
    write_json(config_path, cfg)
    protocol_text = PROTOCOL.read_text()
    protocol_copy = RUN / 'PROTOCOL.md'
    if protocol_copy.exists() and protocol_copy.read_text() != protocol_text:
        raise ValueError('Protocol changed after run preparation')
    if not protocol_copy.exists():
        shutil.copy2(PROTOCOL, protocol_copy)
        write_json(RUN / 'prepared.json', dict(prepared_at=now(), configuration=cfg))
    snapshot = RUN / 'source_snapshot'
    for name in ('scripts/run_eu_direct_measurement_20260924.py',
                 'opal2/observable_quantile_distribution.py',
                 'opal2/quantile_distribution.py',
                 'opal2/measurement_forecast_replay.py'):
        src, dst = ROOT / name, snapshot / name
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def training_inputs():
    """Development labels and confirmation X only; never query future outcomes."""
    data = load_npz(DEV)
    stats = read_json(R4 / 'model/mean/preprocessing.json')
    parts = read_json(R4 / 'model/partitions.json')
    lookup = {str(oid): i for i, oid in enumerate(data['ids'])}
    rows = {k: np.array([lookup[str(oid)] for oid in values], int)
            for k, values in parts.items()}
    if {k: len(v) for k, v in rows.items()} != dict(
            TRAIN=434, VALIDATION=108, REF_FIT=181, DIST_CAL=181):
        raise ValueError('Frozen role counts changed')
    if len(set(np.concatenate(list(rows.values())))) != 904:
        raise ValueError('Development roles overlap or are incomplete')
    xdev = np.column_stack((transform_input(data['Y'][:, 0], stats), data['chem']))
    y = data['Y']
    denom = np.square(y[:, 0]).sum(-1)
    targets = np.column_stack([
        np.log1p(np.square((y[:, a] + y[:, b]) / 2).sum(-1) / denom)
        for a, b in ((1, 2), (1, 3), (2, 3))])
    if not np.isfinite(targets).all() or np.any(targets < 0):
        raise ValueError('Invalid development observable targets')
    query = load_npz(R4 / 'ingest/x/query.npz')
    with np.load(R4 / 'predictions/CORE_ORIGINAL.npz', allow_pickle=False) as z:
        ids = z['ids'].astype(str)
    qlookup = {str(oid): i for i, oid in enumerate(query['ids'])}
    ix = np.array([qlookup[oid] for oid in ids])
    if len(ix) != 1527 or not query['eligible'][ix].all():
        raise ValueError('Common frozen first-well population differs')
    if set(ids) & set(data['ids'].astype(str)):
        raise ValueError('Development/query identities overlap')
    if set(query['groups'][ix]) & set(data['groups']):
        raise ValueError('Development/query chemical groups overlap')
    xquery = np.column_stack((transform_input(query['X'][ix], stats), query['chem'][ix]))
    return dict(data=data, parts=rows, xdev=xdev, targets=targets,
                xquery=xquery, ids=ids, groups=query['groups'][ix], layout=query['layout'][ix])


def fit_target(index):
    folder = RUN / f'target_{index}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        if not (folder / 'query_predictions.npz').exists():
            raise ValueError('Missing completed prediction artifact')
        return read_json(folder / 'complete.json')
    started = time.perf_counter()
    d = training_inputs()
    t = np.r_[d['parts']['TRAIN'], d['parts']['REF_FIT']]
    v, c = d['parts']['VALIDATION'], d['parts']['DIST_CAL']
    y = d['targets'][:, index]
    candidates, validation, metadata = [], [], []
    with threadpool_limits(limits=1):
        for ci, (leaves, minimum) in enumerate(PARAMETERS):
            cdir = folder / f'candidate_{ci}'
            cdir.mkdir(exist_ok=True)
            columns = []
            for qi, alpha in enumerate(GRID):
                checkpoint = cdir / f'quantile_{qi:02d}.joblib'
                done = ci * len(GRID) + qi
                status(folder, 'FITTING', target=TARGETS[index], completed_fits=done,
                       total_fits=57, elapsed_seconds=time.perf_counter()-started)
                if checkpoint.exists():
                    bundle = joblib.load(checkpoint)
                else:
                    model = HistGradientBoostingRegressor(loss='quantile', quantile=float(alpha),
                        max_leaf_nodes=leaves, min_samples_leaf=minimum, max_iter=200,
                        learning_rate=.05, l2_regularization=1., max_bins=255,
                        max_features=1., max_depth=None, early_stopping=False,
                        random_state=SEED, categorical_features=None)
                    start_fit, start_cpu = time.perf_counter(), time.process_time()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model.fit(d['xdev'][t], y[t])
                    meta = dict(candidate=ci, quantile_index=qi, quantile=float(alpha),
                        target=index, seed=SEED, n_iter=int(model.n_iter_),
                        fit_seconds=time.perf_counter()-start_fit,
                        fit_cpu_seconds=time.process_time()-start_cpu,
                        warnings=[str(w.message) for w in caught])
                    if model.n_iter_ != 200:
                        raise RuntimeError('Full 200-iteration fitting not completed')
                    bundle = dict(model=model, validation=model.predict(d['xdev'][v]), metadata=meta)
                    temp = checkpoint.with_name(checkpoint.name + f'.tmp-{os.getpid()}')
                    joblib.dump(bundle, temp, compress=3)
                    temp.replace(checkpoint)
                    write_json(checkpoint.with_suffix('.json'), meta)
                meta = bundle['metadata']
                if (meta['target'], meta['candidate'], meta['quantile_index'], meta['seed']) != (
                        index, ci, qi, SEED):
                    raise ValueError('Checkpoint identity differs')
                metadata.append(meta)
                columns.append(bundle['validation'])
                if (done + 1) % 5 == 0 or done + 1 == 57:
                    print(f'{TARGETS[index]} fits={done+1}/57 elapsed={time.perf_counter()-started:.1f}s', flush=True)
                del bundle
            raw = np.column_stack(columns)
            loss = float(make_nonnegative_quantile_law(raw).crps(y[v]).mean())
            candidates.append(dict(index=ci, max_leaf_nodes=leaves, min_samples_leaf=minimum,
                                   validation_crps=loss))
            validation.append(raw)
            write_json(folder / 'validation_candidates.json', candidates)
        selected = 0
        for ci in range(1, len(candidates)):
            old, new = candidates[selected]['validation_crps'], candidates[ci]['validation_crps']
            if new < old and not np.isclose(new, old, atol=1e-12, rtol=1e-12):
                selected = ci
        cal, query = [], []
        predict_start = time.perf_counter()
        for qi in range(len(GRID)):
            bundle = joblib.load(folder / f'candidate_{selected}/quantile_{qi:02d}.joblib')
            cal.append(bundle['model'].predict(d['xdev'][c]))
            query.append(bundle['model'].predict(d['xquery']))
        cal, query = np.column_stack(cal), np.column_stack(query)
        predict_seconds = time.perf_counter()-predict_start
        offsets = fit_quantile_offsets(cal, y[c])
        write_npz(folder / 'query_predictions.npz', ids=d['ids'], groups=d['groups'], layout=d['layout'],
                  quantiles_raw=query, calibration_offsets=offsets, levels=GRID,
                  selected_candidate=np.asarray(selected))
        write_npz(folder / 'fit_predictions.npz', validation_quantiles=np.asarray(validation),
                  validation_y=y[v], calibration_quantiles=cal, calibration_y=y[c],
                  train_ids=d['data']['ids'][t], validation_ids=d['data']['ids'][v],
                  calibration_ids=d['data']['ids'][c])
        complete = dict(target=TARGETS[index], target_index=index, complete=True,
            completed_at=now(), candidates=candidates, selected_candidate=selected,
            fit_count=57, fit_seconds=sum(m['fit_seconds'] for m in metadata),
            fit_cpu_seconds=sum(m['fit_cpu_seconds'] for m in metadata),
            predict_seconds=predict_seconds, elapsed_seconds=time.perf_counter()-started,
            law_metadata=make_nonnegative_quantile_law(query).metadata(),
            query_outcomes_used_for_training=False)
        write_json(folder / 'complete.json', complete)
        status(folder, 'COMPLETE', completed_fits=57, **complete)
        return complete


def replay():
    from opal2.measurement_forecast_replay import replay_frozen_observables
    path = RUN / 'core_observables.npz'
    if path.exists():
        return dict(reused=True, path=str(path))
    return replay_frozen_observables(R4 / 'predictions', R4 / 'ingest/outcomes/outcomes.npz', path)


def bootstrap_means(matrix, groups, seed):
    """Equal object weights, whole-group paired resampling, variable sample size."""
    unique, ix = np.unique(np.asarray(groups, str), return_inverse=True)
    totals = np.zeros((len(unique), matrix.shape[1]))
    np.add.at(totals, ix, matrix)
    counts = np.bincount(ix, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAPS, matrix.shape[1]))
    for begin in range(0, BOOTSTRAPS, 100):
        end = min(begin+100, BOOTSTRAPS)
        samples = rng.integers(0, len(unique), (end-begin, len(unique)))
        draws[begin:end] = totals[samples].sum(1)/counts[samples].sum(1)[:, None]
    return draws


def evaluate():
    """All fits are complete before query outcomes are used here."""
    core = load_npz(RUN / 'core_observables.npz')
    if len(core['ids']) != 1527 or int(core['observed'].sum()) != 1520:
        raise ValueError('Unexpected common complete-observation population')
    observed = core['observed'].astype(bool)
    actual = core['actual'][observed, 6:9]
    arms = {'CORE': dict(crps=core['crps'][observed, 6:9],
                         mean=core['mean'][observed, 6:9],
                         coverage=core['coverage'][observed, 6:9],
                         width=core['width'][observed, 6:9])}
    raw_scores, cal_scores = [], []
    raw_means, cal_means = [], []
    raw_coverage, cal_coverage, raw_width, cal_width = [], [], [], []
    saved = dict(ids=core['ids'][observed], groups=core['groups'][observed],
                 layout=core['layout'][observed], actual=actual, targets=np.asarray(TARGETS),
                 nominal_coverage=LEVELS)
    for index in range(3):
        direct = load_npz(RUN / f'target_{index}/query_predictions.npz')
        np.testing.assert_array_equal(direct['ids'], core['ids'])
        raw = direct['quantiles_raw'][observed]
        for arm, q, scores, means, covers, widths in (
                ('RAW', raw, raw_scores, raw_means, raw_coverage, raw_width),
                ('CAL', raw + direct['calibration_offsets'], cal_scores, cal_means, cal_coverage, cal_width)):
            law = make_nonnegative_quantile_law(q)
            intervals = law.interval_metrics(actual[:, index], LEVELS)
            scores.append(law.crps(actual[:, index]))
            means.append(law.mean())
            covers.append(intervals['covered'])
            widths.append(intervals['width'])
            saved[f'{arm}_target{index}_quantile_knots'] = law.quantiles
            saved[f'{arm}_target{index}_interval_lower'] = intervals['lower']
            saved[f'{arm}_target{index}_interval_upper'] = intervals['upper']
    for arm, scores, means, covers, widths in (
            ('RAW', raw_scores, raw_means, raw_coverage, raw_width),
            ('CAL', cal_scores, cal_means, cal_coverage, cal_width)):
        arms[arm] = dict(crps=np.column_stack(scores), mean=np.column_stack(means),
                         coverage=np.stack(covers, 1), width=np.stack(widths, 1))
    summary, layout_rows, pair_rows = [], [], []
    labels = (*TARGETS, 'three_target_mean')
    for name, arm in arms.items():
        for key, values in arm.items():
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite {name}/{key} score')
            saved[name+'_'+key] = values
        for ti, target in enumerate(labels):
            scores = arm['crps'][:, ti] if ti < 3 else arm['crps'].mean(1)
            cover = arm['coverage'][:, ti] if ti < 3 else arm['coverage'].mean(1)
            width = arm['width'][:, ti] if ti < 3 else arm['width'].mean(1)
            mse = np.square(arm['mean']-actual)[:, ti] if ti < 3 else np.square(arm['mean']-actual).mean(1)
            record = dict(arm=name, target=target, n=len(scores), crps=float(scores.mean()), mse=float(mse.mean()))
            for li, level in enumerate(LEVELS):
                key = str(int(level*100))
                record['coverage_'+key] = float(cover[:, li].mean())
                record['width_'+key] = float(width[:, li].mean())
            summary.append(record)
            for layout in np.unique(saved['layout']):
                m = saved['layout'] == layout
                layout_rows.append(dict(arm=name, target=target, layout=str(layout), n=int(m.sum()),
                    crps=float(scores[m].mean()), coverage95=float(cover[m, 3].mean()),
                    width95=float(width[m, 3].mean())))
    for direct_name in ('CAL', 'RAW'):
        for ti, target in enumerate(labels):
            def column(arm, key):
                a = arms[arm][key]
                return a[:, ti] if ti < 3 else a.mean(1)
            c, d = column('CORE', 'crps'), column(direct_name, 'crps')
            coverage_diff = column(direct_name, 'coverage')-column('CORE', 'coverage')
            width_diff = column(direct_name, 'width')-column('CORE', 'width')
            matrix = np.column_stack((c, d, coverage_diff, width_diff))
            for grouping, group_values in (('chemical_identity', saved['groups']), ('library_layout', saved['layout'])):
                boot = bootstrap_means(matrix, group_values, 20260924)
                delta, relative = boot[:, 1]-boot[:, 0], 100*(boot[:, 1]-boot[:, 0])/boot[:, 1]
                row = dict(comparator=direct_name, reference='CORE', target=target,
                    primary=(direct_name == 'CAL' and ti == 3), resampling=grouping,
                    n_blocks=len(np.unique(group_values)), repetitions=BOOTSTRAPS,
                    crps_difference=float((d-c).mean()),
                    difference_lower=float(np.quantile(delta, .025)), difference_upper=float(np.quantile(delta, .975)),
                    core_relative_improvement_percent=float(100*(d.mean()-c.mean())/d.mean()),
                    relative_lower=float(np.quantile(relative, .025)), relative_upper=float(np.quantile(relative, .975)))
                for li, level in enumerate(LEVELS):
                    key = str(int(level*100))
                    for prefix, raw, idx in (('coverage', coverage_diff[:, li], 2+li), ('width', width_diff[:, li], 7+li)):
                        row[f'{prefix}{key}_difference'] = float(raw.mean())
                        row[f'{prefix}{key}_lower'] = float(np.quantile(boot[:, idx], .025))
                        row[f'{prefix}{key}_upper'] = float(np.quantile(boot[:, idx], .975))
                pair_rows.append(row)
    write_npz(REPORT / 'object_predictions.npz', **saved)
    pd.DataFrame(summary).to_csv(REPORT / 'summary.csv', index=False)
    pd.DataFrame(layout_rows).to_csv(REPORT / 'layout_summary.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(REPORT / 'paired_intervals.csv', index=False)
    records = []
    for i, oid in enumerate(saved['ids']):
        r = dict(object_id=str(oid), chemical_group=str(saved['groups'][i]), layout=str(saved['layout'][i]))
        for arm in arms:
            r[arm+'_family_crps'] = float(arms[arm]['crps'][i].mean())
            r[arm+'_family_coverage95'] = float(arms[arm]['coverage'][i, :, 3].mean())
        records.append(r)
    pd.DataFrame(records).to_csv(REPORT / 'object_family_scores.csv', index=False)
    result = dict(completed_at=now(), n_complete=1520, n_predicted=1527,
                  primary=[r for r in pair_rows if r['primary']], summary=summary,
                  timings=[read_json(RUN / f'target_{i}/complete.json') for i in range(3)],
                  original_r4_lists_changed=False, new_independent_confirmation=False)
    write_json(REPORT / 'results.json', result)
    return result


def run():
    prepare()
    started = time.perf_counter()
    with (RUN / 'execution.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status(RUN, 'RUNNING', stage='fitting and frozen replay', device='CPU', workers=2)
        try:
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs = {pool.submit(replay): 'CORE replay'}
                jobs.update({pool.submit(fit_target, i): TARGETS[i] for i in range(3)})
                for future in as_completed(jobs):
                    future.result()
                    print(f'COMPLETE: {jobs[future]}', flush=True)
            status(RUN, 'RUNNING', stage='paired evaluation', device='CPU')
            result = evaluate()
            status(RUN, 'COMPLETE', stage='all fits, replay and paired intervals',
                   elapsed_seconds=time.perf_counter()-started, primary=result['primary'])
            print(json.dumps(dict(state='COMPLETE', primary=result['primary'])), flush=True)
        except Exception as exc:
            status(RUN, 'FAILED', stage='execution', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run', 'evaluate', 'fit-target'))
    parser.add_argument('--target', type=int, choices=range(3))
    args = parser.parse_args()
    if args.action == 'run':
        run()
    elif args.action == 'evaluate':
        prepare()
        evaluate()
    elif args.action == 'fit-target':
        prepare()
        if args.target is None:
            parser.error('--target is required')
        fit_target(args.target)
    else:
        prepare()
