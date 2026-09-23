"""EU R3 complete well-profile modules; reuse the saved CORE and R2 fits."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.biology_kernel_evaluation import write_json
from opal2.eu_core_experiment import partitions, select, extra_seed_moments, SAMPLES, SEED as CORE_SEED
from opal2.eu_core_training import predict_eu_core
from opal2.eu_fit_dataset import read_rows
from opal2.gram_geometry import gram_to_coordinates, profiles_to_gram
from opal2.gram_oof_ridge import transform_target
from opal2.conditional_joint_error_experiment import observable_forward
from opal2.conditional_residual_information import ScalePredictor, bounded_scale_ratio
from opal2.residual_descriptors import ResidualDescriptorTransformer
from opal2.dual_branch_features import apply_increment, select_strength, calibration_frame_covariance
from opal2.empirical_radial_experiment import score, LEVELS
from opal2.reference_information_diagnostic import bootstrap_difference
from scripts.run_r2_core_comparison_20260917 import load_means, read_npz, summarize

PHASE = PROJECT/'reports/eu_core_development_20260917_v1'
DATA = PHASE/'prepared_data_cc904'
SOURCE = PROJECT/'runs/eu_core_cc904_20260917_v1'
R2 = PROJECT/'runs/r2_core_comparison_20260917_v1'
ROOT = PROJECT/'runs/r3_eu_modules_20260918_v1'
REPORT = PROJECT/'reports/r3_eu_modules_20260918_v1'
SEED = 20260918
BASE_ARMS = ('GELU', 'BIO_GENERIC', 'BIO_STRUCTURED', 'BIO_RANDOM',
             'PCA_REP', 'DIRECT_REP', 'COND_REP', 'BOTH_GENERIC',
             'BOTH_STRUCTURED', 'BOTH_RANDOM', 'DESCRIPTORS_HGB')
ARMS = ('CORE', *BASE_ARMS, *(a+'_CAL' for a in BASE_ARMS), 'CONDITIONAL')
SWITCH_CANDIDATES = ('CORE', 'BIO_STRUCTURED', 'COND_REP', 'BOTH_STRUCTURED')


def load_scope():
    from opal2.eu_r3_biology import load_eu_biology_metadata
    data = read_npz(DATA/'data.npz')
    meta = json.loads((DATA/'metadata.json').read_text())
    phase = json.loads((PHASE/'phase_manifest.json').read_text())
    if len(data['ids']) != 904 or meta['confirmation_data_loaded']:
        raise ValueError('Expected the previously opened 904-object development dataset')
    if set(data['ids']) | set(meta['excluded_incomplete_ids']) != set(phase['allowed_compound_ids']):
        raise ValueError('Development population changed')
    biology = load_eu_biology_metadata(data['ids'], data['groups'], human_only=True)
    data.update(biology['arrays'])
    metadata = biology['metadata']
    space = json.loads((DATA/'control_space.json').read_text())
    # Name aliases only support the existing descriptor grouping convention.
    # No values, CORE columns or endpoint coordinates are changed.
    aliases = {'Nuc': 'Nuclei', 'Cyto': 'Cytoplasm'}
    names = []
    for original in space['feature_names']:
        prefix, rest = original.split('_', 1)
        names.append(aliases.get(prefix, prefix)+'_'+rest)
    data['feature_names'] = np.asarray(names)
    for i, unit in enumerate(metadata['units']):
        unit['id'] = str(data['ids'][i])
        _, batch, plate, well = str(data['well_ids'][i, 0]).split('|')
        unit.setdefault('roles', {})['X'] = dict(plate=plate, batch=batch,
            well=well, cell_count=float(data['cell_count'][i, 0]))
        unit['layout_block'] = str(data['layout'][i])
    plans = read_rows(PHASE/'identity_split_plan.csv')
    parts = [partitions(data['ids'], data['groups'], plans, f,
        excluded_ids=meta['excluded_incomplete_ids']) for f in range(5)]
    core = read_npz(SOURCE/'AMP_EMP_LOCAL.npz')
    r1 = read_npz(PROJECT/'runs/r1_completion_20260917_v1/observables.npz')
    for saved in (core, r1):
        np.testing.assert_array_equal(saved['ids'], data['ids'])
    # Use the exact original CORE arithmetic for target-side score replay.
    # R1's unnormalised Gram is a scientific cross-check, not a new target.
    gram = profiles_to_gram(torch.as_tensor(data['Y'], dtype=torch.float64))
    raw = gram_to_coordinates(gram).numpy()
    actual, observed, difference, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, core['actual'], atol=1e-12, rtol=1e-12)
    return data, metadata, biology, parts, core, raw, observed, difference


def select_module(calibrations, groups):
    """Choose among the declared candidates using CAL only, with zero fallback."""
    groups = np.asarray(groups)
    scores = []
    names = []
    first = np.asarray(calibrations['BIO_STRUCTURED']['scores'])
    scores.append(first[:, 0]); names.append('CORE')
    for name in SWITCH_CANDIDATES[1:]:
        choice = calibrations[name]
        index = list(choice['strengths']).index(choice['strength'])
        scores.append(np.asarray(choice['scores'])[:, index])
        names.append(name+'_CAL')
    scores = np.stack(scores, axis=1)
    grouped = np.stack([scores[groups == g].mean(0) for g in np.unique(groups)])
    mean = grouped.mean(0)
    best = int(np.argmin(mean))
    eligible = [0]
    for j in range(1, len(names)):
        delta = grouped[:, j]-grouped[:, 0]
        se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
        if delta.mean() < -se:
            eligible.append(j)
    chosen = min(eligible, key=lambda j: (mean[j], j))
    return dict(arm=names[chosen], mean_calibration_nll=mean.tolist(),
        candidates=names, best_unrestricted=names[best], eligible_indices=eligible,
        criterion='group-LOO CAL NLL; paired one-SE admission vs zero; best admitted, deterministic tie',
        formal_guarantee=False)


def reuse_saved_query_mean(replayed, rows, saved):
    """Verify replay, then use the original predictions bit-for-bit.

    Batched linear algebra can differ at machine precision when replaying the
    same fitted model on all objects rather than on its original query batch.
    The saved query predictions, not a fresh floating-point replay, define CORE.
    """
    np.testing.assert_allclose(replayed[rows], saved, atol=1e-12, rtol=1e-12)
    result = np.array(replayed, copy=True)
    result[rows] = saved
    return result


def fit_modules(data, metadata, part, means, raw, stats, oldarrays, folder, seed, *,
                distribution_state=None, calibration_group_representatives=False):
    from opal2.eu_r3_biology import build_eu_biology_features
    from opal2.eu_r3_representation import fit_eu_states, load_eu_states
    from opal2.eu_r3_reference_training import build_adapter_training
    from opal2.eu_r3_adapters import fit_left, fit_right, predict

    t, r, c, q = (part[k] for k in ('TRAIN', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL'))
    checkpoint = folder/'module_predictions.npz'
    if (folder/'modules_complete.json').exists():
        cached = read_npz(checkpoint)
        np.testing.assert_array_equal(cached['cal_ids'], data['ids'][c])
        np.testing.assert_array_equal(cached['query_ids'], data['ids'][q])
        return cached
    transformer_path = folder/'descriptors.joblib'
    if transformer_path.exists():
        transformer = joblib.load(transformer_path)
    else:
        transformer = ResidualDescriptorTransformer.fit(data, metadata, t, seed=seed)
        joblib.dump(transformer, transformer_path)
        write_json(folder/'descriptors.json', transformer.report)
    desc = transformer.transform(data, metadata)
    if (folder/'representations/complete.json').exists():
        states = load_eu_states(folder/'representations', data['Y'][:, 0])
    else:
        states = fit_eu_states(data, t, folder/'representations', seed)
    features = {'GELU': desc.values}
    for name, key in (('PCA_REP', 'PCA_STATE'), ('DIRECT_REP', 'DIRECT_STATE'),
                      ('COND_REP', 'CONDITIONAL_STATE')):
        z, sd = states['states'][key]
        features[name] = np.column_stack((desc.values, z/sd))
    logamp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    edges = np.quantile(logamp[t], [.2, .4, .6, .8])
    reference_folder = folder/'adapter_reference'
    if (reference_folder/'summary.json').exists():
        reference_report = json.loads((reference_folder/'summary.json').read_text())
        if not reference_report['complete']:
            raise ValueError('Incomplete saved reference training records')
        records = read_npz(reference_folder/'records.npz')
    else:
        bundle = build_adapter_training(data, metadata, r, means, raw, stats,
            oldarrays['base_scatter'], float(logamp[t].std()), reference_folder,
            seed=seed, amplitude_edges=edges)
        records, reference_report = bundle['records'], bundle['report']
    np.testing.assert_array_equal(records['ids'], data['ids'][r])
    nested = dict(energies=records['energies'])
    for prefix in ('biology', 'random_biology'):
        nested[prefix] = dict(names=reference_report['biological_names'],
            **{key: records[prefix+'_'+key] for key in ('values', 'support', 'support_by_relation')})
    # DIST_CAL summaries use their own group-LOO radial covariance, not a
    # covariance learned when they served as queries in a different outer fold.
    cal_residual = transform_target(raw[c], stats)-means[c]
    state = distribution_state if distribution_state is not None else json.loads(
        (SOURCE/f'fold_{int(folder.name.split("_")[-1])}'/'distribution_state.json').read_text())
    bandwidth = state['radial_reference_bandwidth']
    calcov = calibration_frame_covariance(oldarrays['cal_scatter_u'], cal_residual,
        logamp[c], data['groups'][c], bandwidth,
        representative_ids=data['ids'][c] if calibration_group_representatives else None)
    qcov = oldarrays['query_scatter_u']*oldarrays['radial_variance_multiplier'][:, None, None]
    scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
    rawmean = means*scale+center
    rr = raw[r]-rawmean[r]
    biological = {}
    random = {}
    for key, rows, cov in (('cal', c, calcov), ('query', q, qcov)):
        kwargs = dict(amplitude_edges=edges)
        args = (data, metadata, rows, r, rawmean[rows],
                cov*scale[None, :, None]*scale[None, None, :], rr)
        biological[key] = build_eu_biology_features(*args, **kwargs)
        random[key] = build_eu_biology_features(*args, random_seed=seed+7000, **kwargs)
    models = {}
    increments = {}
    for name in ('GELU', 'PCA_REP', 'DIRECT_REP', 'COND_REP'):
        path = folder/(name+'.joblib')
        if path.exists():
            model = joblib.load(path)
        else:
            model = fit_left(features[name][r], nested['energies'], data['ids'][r], seed=seed)
            joblib.dump(model, path)
        models[name] = model
        write_json(folder/(name+'_training.json'), model.report)
        increments[name] = {k: predict(model, features[name][rows])['total']
                            for k, rows in (('cal', c), ('query', q))}
        for k, rows in (('cal', c), ('query', q)):
            np.testing.assert_array_equal(predict(model, features[name][rows], enabled=False)['total'],
                                          np.zeros((len(rows), 2)))
    for name, left_name, mode, is_random in (
        ('BIO_GENERIC', 'GELU', 'generic', False),
        ('BIO_STRUCTURED', 'GELU', 'structured', False),
        ('BIO_RANDOM', 'GELU', 'structured', True),
        ('BOTH_GENERIC', 'COND_REP', 'generic', False),
        ('BOTH_STRUCTURED', 'COND_REP', 'structured', False),
        ('BOTH_RANDOM', 'COND_REP', 'structured', True)):
        train_bio = nested['random_biology' if is_random else 'biology']
        held_bio = random if is_random else biological
        path = folder/(name+'.joblib')
        if path.exists():
            model = joblib.load(path)
        else:
            model = fit_right(models[left_name], features[left_name][r], train_bio['values'],
                train_bio['names'], train_bio['support'], nested['energies'], data['ids'][r],
                mode=mode, seed=seed+100)
            joblib.dump(model, path)
        write_json(folder/(name+'_training.json'), model.report)
        increments[name] = {}
        for k, rows in (('cal', c), ('query', q)):
            b = held_bio[k]
            comp = predict(model, features[left_name][rows], b['values'], b['support'])
            np.testing.assert_array_equal(comp['left'], increments[left_name][k])
            np.testing.assert_array_equal(predict(model, features[left_name][rows], b['values'],
                b['support'], enabled=False)['total'], np.zeros((len(rows), 2)))
            np.testing.assert_array_equal(predict(model, features[left_name][rows], b['values'],
                b['support'], right_enabled=False)['total'], increments[left_name][k])
            increments[name][k] = comp['total']
    hgb_path = folder/'DESCRIPTORS_HGB.joblib'
    if hgb_path.exists():
        hgb = joblib.load(hgb_path)
    else:
        hgb = ScalePredictor.fit(desc.values[r], nested['energies'],
            np.arange(desc.values.shape[1]), data['ids'][r], seed=seed)
        joblib.dump(hgb, hgb_path)
    increments['DESCRIPTORS_HGB'] = {k: np.log(bounded_scale_ratio(hgb.predict(desc.values[rows]),
        np.ones((len(rows), 2)))) for k, rows in (('cal', c), ('query', q))}
    output = dict(cal_ids=data['ids'][c], query_ids=data['ids'][q],
        cal_support=biological['cal']['support'], query_support=biological['query']['support'],
        cal_biology=biological['cal']['values'], query_biology=biological['query']['values'],
        cal_covariance=calcov, cal_residual=cal_residual,
        **{name+'_'+key: value for name, record in increments.items() for key, value in record.items()})
    np.savez_compressed(checkpoint, **output)
    write_json(folder/'modules_complete.json', dict(complete=True, query_n=len(q), reference_training_n=len(r),
        reference_supported=int(nested['biology']['support'].sum()),
        query_supported=int(biological['query']['support'].sum()),
        calibration_supported=int(biological['cal']['support'].sum()),
        descriptor_count=desc.values.shape[1], state_dim=8,
        left_support_independent=True, mean_retrained=False, original_image_representation=False))
    return output


def collect_and_analyze(data, core, cells):
    ids, groups, layout = data['ids'], data['groups'], data['layout']
    actual, folds = core['actual'], core['fold']
    stores = {}
    for arm in ARMS:
        entries = []
        fields = None
        assigned = np.zeros(len(ids), int)
        for f in range(5):
            rows = np.flatnonzero(folds == f)
            saved = read_npz(ROOT/f'fold_{f}'/(arm+'.npz'))
            np.testing.assert_array_equal(saved['ids'], ids[rows])
            np.testing.assert_allclose(saved['actual'], actual[rows], atol=1e-12, rtol=0)
            keys = {k for k, v in saved.items() if k not in ('ids', 'actual') and v.shape[:1] == (len(rows),)}
            fields = keys if fields is None else fields & keys
            entries.append((rows, saved)); assigned[rows] += 1
        np.testing.assert_array_equal(assigned, np.ones(len(ids), int))
        out = {}
        for key in fields:
            v = entries[0][1][key]
            full = np.empty((len(ids), *v.shape[1:]), v.dtype)
            for rows, saved in entries:
                full[rows] = saved[key]
            if full.dtype.kind in 'fci' and not np.isfinite(full).all():
                raise ValueError('Nonfinite aggregated '+arm+'/'+key)
            out[key] = full
        stores[arm] = out
    metrics = {arm: summarize(out, actual, ids, folds) for arm, out in stores.items()}
    support = stores['CORE']['resource_support'].astype(bool)
    for arm, out in stores.items():
        out['brier'] = np.square(out['p_null']-(actual <= 0))
        out['policy_value'] = actual*out['selected_lambda_0.2']
        out['policy_null'] = ((actual <= 0)&out['selected_lambda_0.2']).astype(float)
        metrics[arm]['supported_n'] = int(support.sum())
        metrics[arm]['supported_scores'] = {k: float(out[k][support].mean()) if support.any() else None
                                           for k in ('crps', 'brier', 'nll', 'energy')}
        metrics[arm]['mean_unchanged'] = bool(np.array_equal(out['mean_u'], core['mean_u']))
        metrics[arm]['active_n'] = int(np.any(out['increment'] != 0, axis=1).sum())
        metrics[arm]['increment_rms'] = float(np.sqrt(np.square(out['increment']).mean()))
        np.savez_compressed(ROOT/(arm+'.npz'), ids=ids, groups=groups, layout=layout,
            fold=folds, actual=actual, **out)
    pairs = [(a, 'CORE') for a in ARMS if a != 'CORE']
    pairs += [('BIO_STRUCTURED', 'BIO_GENERIC'), ('BIO_STRUCTURED', 'BIO_RANDOM'),
        ('BIO_STRUCTURED', 'GELU'), ('COND_REP', 'PCA_REP'), ('COND_REP', 'DIRECT_REP'),
        ('COND_REP', 'GELU'), ('COND_REP', 'DESCRIPTORS_HGB'),
        ('BOTH_STRUCTURED', 'BIO_STRUCTURED'), ('BOTH_STRUCTURED', 'COND_REP'),
        ('BOTH_STRUCTURED', 'BOTH_GENERIC'), ('BOTH_STRUCTURED', 'BOTH_RANDOM')]
    paired = {}
    for a, b in pairs:
        record = {}
        for scope, mask in (('all', np.ones(len(ids), bool)), ('supported', support)):
            if not mask.any():
                record[scope] = None; continue
            record[scope] = {k: {unit: bootstrap_difference(stores[a][k][mask], stores[b][k][mask], lab[mask])
                for unit, lab in (('chemical_group', groups), ('layout', layout))}
                for k in ('crps', 'brier', 'nll', 'energy')}
        record['policy'] = {k: {unit: bootstrap_difference(stores[a][k], stores[b][k], lab)
            for unit, lab in (('chemical_group', groups), ('layout', layout))}
            for k in ('policy_value', 'policy_null')}
        paired[a+' minus '+b] = record
    # Read-only comparison with R2 outputs. No simple baseline is refitted.
    r2_comparisons = {}
    for baseline in ('DIRECT_ACCESS_MATCHED_HISTGB_COHERENT',
                     'DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL', 'HR_REF'):
        old = read_npz(R2/(baseline+'.npz'))
        np.testing.assert_array_equal(old['ids'], ids)
        summarize(old, actual, ids, folds)
        for arm in ('CORE', 'BIO_STRUCTURED', 'COND_REP', 'BOTH_STRUCTURED', 'CONDITIONAL'):
            out = stores[arm]
            quantities = {'crps': (out['crps'], old['crps']),
                'brier': (out['brier'], np.square(old['p_null']-(actual <= 0))),
                'policy_value': (out['policy_value'], actual*old['selected_lambda_0.2'])}
            r2_comparisons[arm+' minus '+baseline] = {k: {unit: bootstrap_difference(x, y, lab)
                for unit, lab in (('chemical_group', groups), ('layout', layout))}
                for k, (x, y) in quantities.items()}
    monte_carlo = {}
    for arm in ARMS:
        records = []
        for offset in (100000, 200000):
            moment = {k: np.empty(len(ids)) for k in ('predicted', 'p_null')}
            for f in range(5):
                rows = np.flatnonzero(folds == f)
                saved = read_npz(ROOT/f'fold_{f}'/f'{arm}_mc{offset}.npz')
                np.testing.assert_array_equal(saved['ids'], ids[rows])
                for key in moment:
                    moment[key][rows] = saved[key]
            m = summarize(moment, actual, ids, folds)
            for lam in (.2, 0.):
                key = f'lambda_{lam:g}'
                m['policies'][key]['list_symmetric_difference'] = int(np.sum(
                    moment['selected_'+key] != stores[arm]['selected_'+key]))
            records.append(dict(seed_offset=offset, policies=m['policies']))
        monte_carlo[arm] = records
    deployment_costs = []
    for cell in cells:
        take = folds == cell['fold']
        nquery = int(take.sum())
        costs = cell['costs']
        rows = {}
        for arm, out in stores.items():
            net = float(np.sum(actual[take]*out['selected_lambda_0.2'][take]))
            new = int(costs['reference_if_all_new_wells'])
            after_x = int(costs['reference_if_X_already_available'])
            rows[arm] = dict(existing_reference_action_net_value=net,
                new_reference_net_value=net-.01*new,
                reference_X_available_net_value=net-.01*after_x,
                extra_reference_wells_beyond_CORE=0,
                training_compute_is_additional=True,
                amortization_queries_if_value_persists=None if net <= 0 else .01*new*nquery/net)
        deployment_costs.append(dict(fold=cell['fold'], query_n=nquery, counts=costs, arms=rows))
    payload = dict(state='COMPLETE', n=len(ids), metrics=metrics, paired=paired,
        r2_comparisons=r2_comparisons, cells=cells, supported_n=int(support.sum()),
        monte_carlo_sensitivity=monte_carlo, deployment_costs=deployment_costs,
        core_retrained=False, samples=SAMPLES, query_endpoints_changed=False,
        original_image_representation_tested=False, confirmation_opened=False,
        evidence='EU well-profile module development; fixed-fitted-model and fixed-selection paired diagnostics',
        multiple_comparisons_adjusted=False, independent_confirmation=False)
    write_json(ROOT/'summary.json', payload)
    lines = ['# R3 EU well-profile module comparison', '',
        '904 opened development objects; original CORE, endpoint and budgets preserved.', '',
        '| Arm | Γ CRPS | NULL Brier | NULL AUC | selected NULL | selected mean Γ |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, m in metrics.items():
        p = m['policies']['lambda_0.2']
        lines.append(f'| {arm} | {m["crps"]:.8f} | {m["brier"]:.8f} | {m["null_auc"]:.4f} | '
                     f'{p["null_selected"]} | {p["selected_mean_value"]:.8f} |')
    lines += ['', 'Raw-image representation remains a separate, uncompleted comparison.',
        'Fixed-target encoders are not residual-target encoders or EMA JEPA.',
        'CAL selects adapter strength; CONDITIONAL selects among the declared module candidates on CAL only.',
        'CORE/radius laws and strong R2 baselines are reused. No confirmation measurements were opened.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return payload


def run(*, prepare_only=False, aggregate_only=False):
    start = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    def status(stage, **extras):
        write_json(ROOT/'status.json', dict(state='RUNNING', stage=stage,
            elapsed_seconds=time.monotonic()-start, **extras))
    if (ROOT/'status.json').exists():
        prior = json.loads((ROOT/'status.json').read_text())
        if prior['state'] == 'COMPLETE' and not aggregate_only:
            print(json.dumps(prior)); return
    try:
        status('loading opened development data')
        data, metadata, biology, parts, core, raw, observed, difference = load_scope()
        write_json(ROOT/'biology_metadata_audit.json', biology['report'])
        spec = dict(n=904, folds=5, arms=ARMS, samples=SAMPLES, seed=SEED,
            core_sampling_seed=CORE_SEED, data=str(DATA), source=str(SOURCE),
            core_retrained=False, input_modality='well profiles only', confirmation_opened=False,
            raw_images_opened=False, predefined_primary=('CORE', 'BIO_STRUCTURED', 'COND_REP', 'BOTH_STRUCTURED'))
        if (ROOT/'run_manifest.json').exists():
            old = json.loads((ROOT/'run_manifest.json').read_text())
            if json.loads(json.dumps(spec)) != old:
                raise ValueError('Run configuration changed; preserve this run')
        else:
            write_json(ROOT/'run_manifest.json', spec)
            shutil.copy2(REPORT/'PROTOCOL.md', ROOT/'PROTOCOL.md')
        if aggregate_only:
            cells = [json.loads((ROOT/f'fold_{f}/complete.json').read_text()) for f in range(5)]
            collect_and_analyze(data, core, cells)
            write_json(ROOT/'status.json', dict(state='COMPLETE', completed_folds=5,
                aggregation_only=True, elapsed_seconds=time.monotonic()-start)); return
        cells = []
        for f, part in enumerate(parts):
            folder = ROOT/f'fold_{f}'; folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():
                cells.append(json.loads((folder/'complete.json').read_text())); continue
            status('preparing full modules', fold=f, completed_folds=len(cells))
            t, r, c, q = (part[k] for k in ('TRAIN', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL'))
            source = SOURCE/f'fold_{f}'
            stats = json.loads((source/'mean/preprocessing.json').read_text())
            oldarrays = read_npz(source/'distribution_arrays.npz')
            for name, rows in (('ref', r), ('cal', c), ('query', q)):
                np.testing.assert_array_equal(oldarrays[name+'_ids'], data['ids'][rows])
            if (folder/'mean_replay.npz').exists():
                means = read_npz(folder/'mean_replay.npz')['mean_u']
            else:
                _, _, model = load_means(source/'mean')
                means = predict_eu_core(model, stats, data['Y'][:, 0], data['chem'], data['chem_mask'])['mean_u']
                np.savez_compressed(folder/'mean_replay.npz', ids=data['ids'], mean_u=means)
            np.testing.assert_array_equal(oldarrays['query_mean_u'], core['mean_u'][q])
            replay_difference = float(np.max(np.abs(means[q]-oldarrays['query_mean_u'])))
            means = reuse_saved_query_mean(means, q, oldarrays['query_mean_u'])
            write_json(folder/'mean_replay_audit.json', dict(
                maximum_query_replay_difference=replay_difference,
                query_predictions_reused_exactly=True,
                source=str(source/'distribution_arrays.npz')))
            predictions = fit_modules(data, metadata, part, means, raw, stats, oldarrays, folder, SEED+100*f)
            logamp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
            state = json.loads((source/'distribution_state.json').read_text())
            scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
            rawmean = means*scale+center
            calibrations = {}
            for arm in BASE_ARMS:
                path = folder/(arm+'_calibration.json')
                if path.exists():
                    choice = json.loads(path.read_text())
                else:
                    choice = select_strength(rawmean[c], scale, oldarrays['cal_scatter_u'],
                        predictions['cal_residual'], logamp[c], data['groups'][c],
                        state['radial_reference_bandwidth'], predictions[arm+'_cal'])
                    write_json(path, choice)
                calibrations[arm] = choice
            conditional = select_module(calibrations, data['groups'][c])
            write_json(folder/'conditional_selection.json', conditional)
            if prepare_only:
                print(f'fold {f+1}/5 prepared (no joint sampling)', flush=True); continue
            original = read_npz(source/'AMP_EMP_LOCAL.npz')
            target = transform_target(raw[q], stats)
            np.testing.assert_allclose(target, original['actual_u'], atol=1e-11, rtol=1e-11)
            norm2 = np.square(np.asarray(data['Y'][q, 0], dtype=np.float64)).mean(1)
            absolute = np.log1p(difference[q]*norm2[:, None])
            evaluated = {}
            increments = {}
            arm_cells = {}
            for arm in ARMS:
                path = folder/(arm+'.npz')
                if arm == 'CORE':
                    increment = np.zeros((len(q), 2))
                elif arm == 'CONDITIONAL':
                    increment = increments[conditional['arm']].copy()
                else:
                    base = arm.removesuffix('_CAL')
                    strength = calibrations[base]['strength'] if arm.endswith('_CAL') else 1.
                    increment = strength*predictions[base+'_query']
                changed = np.any(increment != 0, axis=1)
                # Full-row equality is used for reuse; no approximate MC shortcut.
                equal = next((a for a, value in increments.items() if np.array_equal(value, increment)), None)
                increments[arm] = increment
                status('100k joint evaluation', fold=f, arm=arm, completed_folds=len(cells))
                if path.exists():
                    out = {k: v for k, v in read_npz(path).items() if k not in ('ids', 'actual')}
                elif arm == 'CORE':
                    out = {k: v.copy() for k, v in original.items() if k not in ('ids', 'groups', 'actual')}
                elif equal is not None:
                    out = {k: v.copy() for k, v in evaluated[equal].items()}
                else:
                    scatter = apply_increment(rawmean[q], scale, oldarrays['query_scatter_u'], increment)
                    np.testing.assert_array_equal(scatter[~changed], oldarrays['query_scatter_u'][~changed])
                    out = score(means[q], scatter, target, stats, core['actual'][q], observed[q],
                        absolute, norm2, CORE_SEED+100*f, law=state['law'],
                        weights=oldarrays['radial_weights'], samples=SAMPLES)
                    for key, value in out.items():
                        if key in original:
                            np.testing.assert_array_equal(value[~changed], original[key][~changed])
                    out.update(mean_u=means[q], actual_u=target, scatter_u=scatter)
                out.update(increment=increment, resource_support=predictions['query_support'])
                for lam in (.2, 0.):
                    out[f'selected_lambda_{lam:g}'] = select(data['ids'][q], out['predicted'], out['p_null'], lam)
                np.savez_compressed(path, ids=data['ids'][q], actual=core['actual'][q], **out)
                evaluated[arm] = out
                for offset in (100000, 200000):
                    dest = folder/f'{arm}_mc{offset}.npz'
                    if dest.exists(): continue
                    if arm == 'CORE':
                        shutil.copy2(source/f'AMP_EMP_LOCAL_mc{offset}.npz', dest)
                    elif equal is not None:
                        shutil.copy2(folder/f'{equal}_mc{offset}.npz', dest)
                    else:
                        scatter = out['scatter_u']
                        mc = extra_seed_moments(means[q], scatter, stats, law=state['law'],
                            weights=oldarrays['radial_weights'], seed=CORE_SEED+100*f+offset)
                        base_mc = read_npz(source/f'AMP_EMP_LOCAL_mc{offset}.npz')
                        for key in mc:
                            np.testing.assert_array_equal(mc[key][~changed], base_mc[key][~changed])
                        np.savez_compressed(dest, ids=data['ids'][q], **mc)
                arm_cells[arm] = dict(changed=int(changed.sum()), reused_equal_arm=equal,
                    selected_null=int(((core['actual'][q] <= 0)&out['selected_lambda_0.2']).sum()),
                    crps=float(out['crps'].mean()))
                print(f'fold={f+1}/5 arm={arm} changed={changed.sum()} elapsed={time.monotonic()-start:.1f}', flush=True)
            cell = dict(fold=f, n=len(q), query_ids=data['ids'][q], arms=arm_cells,
                conditional=conditional, calibrations={a: v['strength'] for a, v in calibrations.items()},
                costs=json.loads((source/'summary.json').read_text())['costs'])
            write_json(folder/'complete.json', cell); cells.append(cell)
        if prepare_only:
            write_json(ROOT/'status.json', dict(state='PREPARED', completed_module_folds=5,
                joint_evaluation_complete=False, elapsed_seconds=time.monotonic()-start)); return
        status('aggregate', completed_folds=5)
        collect_and_analyze(data, core, cells)
        write_json(ROOT/'status.json', dict(state='COMPLETE', completed_folds=5,
            elapsed_seconds=time.monotonic()-start, report=str(ROOT/'REPORT.md')))
    except Exception:
        write_json(ROOT/'status.json', dict(state='FAILED', elapsed_seconds=time.monotonic()-start,
            traceback=traceback.format_exc(), retained_completed_artifacts=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run(prepare_only=args.prepare_only, aggregate_only=args.aggregate_only)
