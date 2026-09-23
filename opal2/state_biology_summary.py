"""Fixed-epoch saved-output comparison of state-conditioned relation kernels.

The primary equal-capacity contrast substitutes target/MoA relations for
X-derived relations. It is not a nested additive-biology ablation. No training,
checkpoint selection, covariance fitting, or threshold search occurs here.
"""
from __future__ import annotations

import argparse
import filecmp
import json
from pathlib import Path

import numpy as np

from .baseline_policy import _metrics
from .biology_kernel_evaluation import write_json
from .conditional_response_summary import ARRAY_SHAPES, _model_scores
from .geometry_kernel_summary import _policy
from .gram_oof_experiment import selection_mask
from .hierarchical_geometry_summary import _folds
from .independent_biology_summary import _finite, _read_diagnostics, _format_stat
from .lincs_biology_summary import bootstrap_weights, compare, difference, interval


ARMS = ('A_FROZEN', 'BIO_TEMPLATE50', 'STATE50', 'STATE_BIO50')
PRIMARY = ('STATE_BIO50', 'STATE50')
PAIRS = (PRIMARY, ('STATE_BIO50', 'BIO_TEMPLATE50')) + tuple((a, 'A_FROZEN') for a in ARMS[1:])
EPOCHS = 50
SAMPLES = 10000
BOOTSTRAP = 2000
SCOPE = ('paired chemistry-group bootstrap of saved OOF predictions; layout-block resampling is a separate '
         'sensitivity analysis; excludes refitting, repeated development selection and Monte Carlo uncertainty; '
         'not independent certification or cross-batch validation')


def _json(path):
    return json.loads(Path(path).read_text())


def validate_manifest(manifest):
    if manifest.get('arms') != list(ARMS) or manifest.get('fixed_epochs') != EPOCHS:
        raise ValueError('Expected four declared arms and fixed epoch50')
    if manifest.get('actual_checkpoint_epoch', EPOCHS) != EPOCHS:
        raise ValueError('New fitted arms require actual epoch50')
    ids, allocation = _folds(manifest)
    if len(manifest['folds']) != 5:
        raise ValueError('All five original folds are required')
    groups = np.asarray(manifest['groups'], dtype=str)
    if groups.shape != ids.shape:
        raise ValueError('Chemistry groups must align with identities')
    for group in np.unique(groups):
        if len(np.unique(allocation[groups == group])) != 1:
            raise ValueError('A chemistry group crosses outer test folds')
    units = manifest['dataset']['units']
    if len(units) != len(ids) or any('layout_block' not in row for row in units):
        raise ValueError('Each eligible object requires its original layout block')
    cfg = manifest['config']
    if cfg.get('samples') != SAMPLES or cfg.get('stage_epochs') != EPOCHS or cfg.get('max_epochs') != 100:
        raise ValueError('Expected 10000 draws and fixed50/cosine100 training schedule')
    for key in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed',
                'original_contract_changed', 'endpoint_changed', 'covariance_updated_by_branch',
                'reference_selection_changed', 'formal_certificate', 'independent_new_holdout'):
        if manifest.get(key, False) is not False:
            raise ValueError('Protected state-comparison scope changed: '+key)
    if not isinstance(manifest.get('reference_run'), str):
        raise ValueError('Original scale-comparison reference_run is required')
    return ids, allocation


def _source_checks(root, manifest):
    source = Path(manifest['reference_run']).expanduser().resolve()
    if source == root.resolve():
        raise ValueError('New state experiment must not overwrite the reference run')
    old = _json(source/'run_manifest.json')
    for key in ('ids', 'groups', 'folds', 'scopes', 'dataset'):
        if manifest.get(key) != old.get(key):
            raise ValueError('Original scope or cohort differs: '+key)
    checked = []
    for record in manifest['folds']:
        relative = Path('folds')/f"fold_{record['fold']}"/'arms/A_FROZEN'
        for name in ('evaluation/metrics.json', 'evaluation/predictions.npz',
                     'evaluation/u_predictions.npz', 'model_diagnostics.npz'):
            a, b = root/relative/name, source/relative/name
            if not a.is_file() or not b.is_file() or not filecmp.cmp(a, b, shallow=False):
                raise ValueError('Frozen A copied artifact differs: '+str(relative/name))
        checked.append(dict(fold=record['fold'], frozen_A_identical=True))
    return dict(reference_run=str(source), frozen_A_source_epoch=30,
                frozen_A_unchanged=True, folds=checked,
                source_training_epochs_not_applied_to_new_arms=True)


def read_arm(root, manifest, ids, allocation, arm, *, fitted_epochs=EPOCHS):
    """Read saved predictions; expected fit epoch is explicit, never a global override."""
    n = len(ids)
    arrays = {key: np.empty((n, *tail)) for key, tail in ARRAY_SHAPES.items()}
    arrays.update(actual_u=np.empty((n, 9)), mean_u=np.empty((n, 9)), covariance_u=np.empty((n, 9, 9)))
    diagnostics, rows = {}, []
    for record in manifest['folds']:
        ix = np.asarray(record['test'], int)
        fold = root/'folds'/f"fold_{record['fold']}"
        if not (fold/'complete.json').is_file():
            raise ValueError(f"Fold {record['fold']} is not complete")
        folder = fold/'arms'/arm
        evaluation = folder/'evaluation'
        if _json(evaluation/'metrics.json').get('samples') != SAMPLES:
            raise ValueError('Each evaluation requires 10000 samples')
        with np.load(evaluation/'predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Prediction identity/order mismatch')
            for key, tail in ARRAY_SHAPES.items():
                arrays[key][ix] = _finite(saved[key], (len(ix), *tail), key)
            draws = _finite(saved['utility_samples'], (SAMPLES, len(ix), 3), 'utility draws')
            if not np.allclose(draws.mean(0), saved['predicted'], rtol=1e-12, atol=1e-14):
                raise ValueError('Predictive means do not use all 10000 draws')
            if not np.array_equal((draws <= 0).mean(0), saved['p_null']):
                raise ValueError('NULL probabilities do not use all 10000 draws')
        with np.load(evaluation/'u_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Coordinate identity/order mismatch')
            for key in ('actual_u', 'mean_u'):
                arrays[key][ix] = _finite(saved[key], (len(ix), 9), key)
            cov = np.asarray(saved['covariance_u'])
            if cov.shape == (9, 9):
                cov = np.broadcast_to(cov, (len(ix), 9, 9))
            arrays['covariance_u'][ix] = _finite(cov, (len(ix), 9, 9), 'covariance_u')
        row = dict(fold=record['fold'], n=len(ix))
        if arm != 'A_FROZEN':
            complete = _json(folder/'training_complete.json')
            count = complete.get('trainable_parameters')
            if complete.get('actual_checkpoint_epoch') != fitted_epochs:
                raise ValueError(f'Fitted arms must use actual epoch{fitted_epochs}')
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError('Actual positive trainable-parameter count is required')
            history = [json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines() if s.strip()]
            if not history or not any(entry.get('epoch') == fitted_epochs for entry in history):
                raise ValueError(f'Missing actual epoch{fitted_epochs} history')
            row.update(training=complete, history=history,
                training_configuration=_json(folder/'training_config.json'),
                validation_history_is_descriptive_not_checkpoint_selection=True)
        numeric = _read_diagnostics(folder/'model_diagnostics.npz', ids[ix], arrays['mean_u'][ix], required=arm!='A_FROZEN')
        for key, value in numeric.items():
            if key not in diagnostics:
                diagnostics[key] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
            if diagnostics[key].shape[1:] != value.shape[1:]:
                raise ValueError('Diagnostic dimensions changed: '+key)
            diagnostics[key][ix] = value
        rows.append(row)
    if np.any(arrays['p_null'] < 0) or np.any(arrays['p_null'] > 1):
        raise ValueError('NULL probabilities outside [0,1]')
    arrays.update(principal_mask=selection_mask(arrays['predicted'][:, 2], ids, allocation, .25, 2),
                  u_object_mse=np.square(arrays['actual_u']-arrays['mean_u']).mean(1))
    return arrays, rows, diagnostics


def _training_checks(models, manifest):
    folds = []
    for i, record in enumerate(manifest['folds']):
        configs, counts = [], {}
        for arm in ARMS[1:]:
            row = models[arm]['folds'][i]
            config, completion = row['training_configuration'], row['training']
            required = ('config', 'training_seed', 'fit_ids', 'validation_ids', 'loss', 'covariance')
            if any(key not in config for key in required):
                raise ValueError('Incomplete matched training configuration')
            configs.append({key: config[key] for key in required})
            if config.get('test_data_supplied', False) is not False:
                raise ValueError('Outer test data supplied to training')
            if any(completion.get(k, False) is not False for k in ('frozen_A_changed', 'objective_buffers_changed')):
                raise ValueError('A frozen model or objective buffer changed')
            if completion.get('disabled_equals_A', True) is not True:
                raise ValueError('Disabled new branch must restore frozen A')
            if config['config'].get('stage_epochs') != EPOCHS or config['config'].get('max_epochs') != 100:
                raise ValueError('New arm training schedule differs')
            counts[arm] = completion['trainable_parameters']
        if any(value != configs[0] for value in configs[1:]):
            raise ValueError('New arms differ in training scope, seed, loss or budget')
        if counts['STATE50'] != counts['STATE_BIO50']:
            raise ValueError('Primary state pair has unequal active capacity')
        folds.append(dict(fold=record['fold'], trainable_parameters=counts,
            primary_pair_equal_capacity=True, all_new_arms_same_budget=True,
            template_capacity_matched_to_state=False))
    return dict(folds=folds, primary_comparison=list(PRIMARY),
        contrast='same state-conditioned architecture; X-derived relations versus target/MoA relations',
        not_nested_additive_biology_ablation=True, frozen_A_actual_epoch=30,
        new_arms_actual_epoch=EPOCHS)


def _subset_scores(arrays, subset):
    if not subset.any():
        return dict(n=0)
    actual = arrays['actual'][subset, 2]
    return dict(u_mse=float(arrays['u_object_mse'][subset].mean()),
        gamma_crps=float(arrays['utility_crps'][subset, 2].mean()),
        **_metrics(actual, arrays['predicted'][subset, 2], arrays['p_null'][subset, 2]),
        policy_restricted_from_full_cohort=_policy(actual, arrays['principal_mask'][subset]),
        selection_not_reranked_within_subset=True)


def _subset_compare(a, b, subset, weights):
    null = (a['actual'][:, 2] <= 0).astype(float)
    mask = subset.astype(float)
    ma, mb = a['principal_mask']*mask, b['principal_mask']*mask
    values = dict(u_mse=a['u_object_mse']-b['u_object_mse'],
        gamma_crps=a['utility_crps'][:, 2]-b['utility_crps'][:, 2],
        null_brier=(a['p_null'][:, 2]-null)**2-(b['p_null'][:, 2]-null)**2,
        net_gain_per_eligible=(a['principal_mask']-b['principal_mask'])*a['actual'][:, 2])
    result = {key: interval(value*mask, mask, weights) for key, value in values.items()}
    for key, value in (('net_gain_per_selected', a['actual'][:, 2]), ('fdp', null)):
        result[key] = (difference(ma*value, ma, mb*value, mb, weights) if ma.sum() and mb.sum()
            else dict(estimate=None, interval95=None, valid_resamples=0))
    return result


def support_checks(data, diagnostics, weights):
    base = data['A_FROZEN']['mean_u']
    for arm in ARMS[1:]:
        if not np.array_equal(diagnostics[arm]['baseline_mean'], base):
            raise ValueError('New arm baseline differs from identical saved A')
    support = diagnostics['STATE_BIO50']['support'].any(1)
    if not np.array_equal(diagnostics['STATE_BIO50']['support'], diagnostics['BIO_TEMPLATE50']['support']):
        raise ValueError('Biological support differs between template and state biology')
    for arm in ('BIO_TEMPLATE50', 'STATE_BIO50'):
        if not np.array_equal(data[arm]['mean_u'][~support], base[~support]):
            raise ValueError('Biology changed A without biological support')
        for key in ARRAY_SHAPES:
            if not np.array_equal(data[arm][key][~support], data['A_FROZEN'][key][~support]):
                raise ValueError('Unsupported biological scores differ from shared-draw A: '+key)
    subsets = {}
    for name, subset in (('supported', support), ('unsupported', ~support)):
        subsets[name] = dict(models={a: _subset_scores(data[a], subset) for a in ARMS},
            comparisons={a+'__minus__'+b: _subset_compare(data[a], data[b], subset, weights) for a, b in PAIRS})
    return dict(supported_n=int(support.sum()), unsupported_n=int((~support).sum()),
        channel_supported_n=diagnostics['STATE_BIO50']['support'].sum(0).tolist(),
        support_equal_template_and_state_biology=True, biological_unsupported_exact_A=True,
        state_only_can_update_biologically_unsupported=True,
        subset_definition='same fixed target/MoA support subset applied to all models; no reranking', subsets=subsets)


def direction_checks(data, ids, records, weights, support):
    """Exact final-mean error decomposition, never an oracle training target."""
    result = {}
    for left, right in PAIRS:
        a, b = data[left], data[right]
        residual, delta = b['actual_u']-b['mean_u'], a['mean_u']-b['mean_u']
        energy = np.square(delta).mean(1)
        alignment = 2*(residual*delta).mean(1)
        observed = a['u_object_mse']-b['u_object_mse']
        if not np.allclose(observed, energy-alignment, rtol=1e-8, atol=1e-12):
            raise ValueError('Exact mean-direction MSE decomposition failed')
        cohorts = {'all': np.ones(len(ids), bool), 'biology_supported': support,
                   'biology_unsupported': ~support}
        cohorts.update({f"fold_{r['fold']}": np.isin(np.arange(len(ids)), r['test']) for r in records})
        result[left+'__minus__'+right] = {name: dict(n=int(mask.sum()),
            direction_energy=interval(energy*mask, mask.astype(float), weights),
            twice_residual_alignment=interval(alignment*mask, mask.astype(float), weights),
            observed_mse_change=interval(observed*mask, mask.astype(float), weights),
            final_correction_rms=float(np.sqrt(energy[mask].mean())) if mask.any() else None)
            for name, mask in cohorts.items()}
    return dict(formula='MSE(new)-MSE(reference)=mean(delta^2)-2mean(realized_reference_residual*delta)',
        coordinate_scope='same original fold-TRAIN-standardized nine Gram coordinates',
        interpretation='realized future residual includes noise; descriptive alignment, not per-object oracle direction',
        comparisons=result)


def _selection_change(a, b, ids):
    ma, mb = a['principal_mask'].astype(bool), b['principal_mask'].astype(bool)
    return dict(left_only=[dict(id=str(ids[i]), actual_gamma=float(a['actual'][i, 2]),
        null=bool(a['actual'][i, 2] <= 0)) for i in np.flatnonzero(ma & ~mb)],
        right_only=[dict(id=str(ids[i]), actual_gamma=float(b['actual'][i, 2]),
        null=bool(b['actual'][i, 2] <= 0)) for i in np.flatnonzero(mb & ~ma)],
        identical=np.array_equal(ma, mb), overlap=int(np.sum(ma & mb)))


def _report(root, result):
    lines = ['# LINCS：状态条件kernel，三个新分支各50轮', '',
        f"{result['n']}个已开放对象，原五个化学分组折。冻结A复用原第30轮；三个新分支固定第50轮比较。",
        'A_FROZEN是HR加旧信息路径，不是纯HR。BIO_TEMPLATE50是仅由靶点/MoA关系产生修正的旧结构，重训50轮。',
        'STATE50和STATE_BIO50使用相同状态条件架构和容量：前者关系来自第一孔形态/范数，后者关系来自靶点/MoA。',
        '**主比较是STATE_BIO50−STATE50：不同关系信息在同容量状态条件kernel中的表现，不是严格的“原模型只加生物信息”嵌套消融。**', '',
        '| 模型 | 几何MSE↓ | Γ fair CRPS↓ | NULL Brier↓ | Γ Spearman | 选中对象/新增孔 | 选中实际Γ↑ | NULL数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        model = result['models'][arm]
        action, policy = model['actions'][2], model['principal_policy']
        rho = 'NA' if action['spearman'] is None else f"{action['spearman']:.4f}"
        gain = 'NA' if policy['per_selected_net_gain'] is None else f"{policy['per_selected_net_gain']:.6f}"
        lines.append(f"|{arm}|{model['u_mse']:.6f}|{action['gamma_crps']:.6f}|{action['null_brier']:.6f}|"
            f"{rho}|{policy['selected_n']}/{policy['used_wells']}|{gain}|{policy['selected_null_count']}|")
    lines += ['', '预算为25%的额外物理孔，每个ADD_TWO占两孔，不是选择25%的化合物。', '',
        '## 主配对比较', '', '| 指标：STATE_BIO50−STATE50 | 差值与95%区间 |', '|---|---|']
    primary = result['comparisons']['STATE_BIO50__minus__STATE50']
    for key, label in (('u_mse', '几何MSE↓'), ('gamma_crps', 'Γ CRPS↓'), ('null_brier', 'NULL Brier↓'),
                       ('net_gain_per_selected', '选中实际Γ↑'), ('fdp', 'FDP↓')):
        lines.append(f"|{label}|{_format_stat(primary[key])}|")
    lines += ['', '区间是条件于已拟合折外预测的化学分组bootstrap；布局分组敏感性另存summary.json。',
        '区间跨零不能解释成胜出或两模型完全相同。逐对象选择变化、全部逐折指标及最终修正的方向分解均已保留。', '',
        '## 生物支持分层', '',
        f"有靶点或MoA参考支持：{result['support_analysis']['supported_n']}个；无支持：{result['support_analysis']['unsupported_n']}个。",
        '两个生物分支在无支持时精确退回冻结A。STATE50仍可利用第一孔，因此主比较也包含稀疏关系与较密形态关系覆盖的差异。',
        '分层沿用全队列已选名单，不在支持子集重新排预算。', '',
        '## 容量、目标与解释范围', '']
    for fold in result['training_checks']['folds']:
        lines.append(f"- 折{fold['fold']}可训练参数："+'；'.join(f"{a}={v}" for a, v in fold['trainable_parameters'].items())+'。')
    lines += ['', '损失仍为几何MSE＋标准化联合Γ CRPS＋修正幅度惩罚，联合误差协方差不变；并非新增收益损失。',
        'BIO_TEMPLATE50与新状态架构容量不同，其对比仅作结构参考，不能归因于单独增加状态或生物信息。',
        '50轮是固定比较点，不自动意味着收敛；验证历史只作描述，不挑更好的检查点。',
        '结果属于已反复使用的开发对象。共享板、固定孔位和训练重叠仍限制统计外推；不发放新认证（noCERT）。',
        '原终点、七项合同、FINAL、第五重复及历史实验结果均未修改。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(output):
    root = Path(output)
    manifest = _json(root/'run_manifest.json')
    ids, allocation = validate_manifest(manifest)
    # Never read partial folds or publish a partial success report.
    if any(not (root/'folds'/f"fold_{record['fold']}"/'complete.json').is_file() for record in manifest['folds']):
        raise ValueError('All five folds are not complete')
    sources = _source_checks(root, manifest)
    data, models, diagnostics = {}, {}, {}
    for arm in ARMS:
        data[arm], rows, diagnostics[arm] = read_arm(root, manifest, ids, allocation, arm)
        if arm != 'A_FROZEN':
            for key in ('actual', 'actual_u'):
                if not np.array_equal(data[arm][key], data['A_FROZEN'][key]):
                    raise ValueError('Arms have different realized targets: '+key)
            if not np.array_equal(data[arm]['covariance_u'], data['A_FROZEN']['covariance_u']):
                raise ValueError('The original frozen joint covariance changed')
        for row in rows:
            row['outer_evaluation'] = _subset_scores(data[arm], allocation == row['fold'])
        models[arm] = dict(_model_scores(data[arm], ids, manifest['folds']), folds=rows)
    training = _training_checks(models, manifest)
    seed = manifest['config']['seed']
    weights = bootstrap_weights(manifest['groups'], BOOTSTRAP, seed+711)
    layout = [str(unit['layout_block']) for unit in manifest['dataset']['units']]
    layout_weights = bootstrap_weights(layout, BOOTSTRAP, seed+712)
    comparisons, layout_comparisons = {}, {}
    for a, b in PAIRS:
        name = a+'__minus__'+b
        comparisons[name] = compare(a, b, data, weights, manifest['folds'])
        comparisons[name]['selection_changes'] = _selection_change(data[a], data[b], ids)
        layout_comparisons[name] = compare(a, b, data, layout_weights, manifest['folds'])
    support = support_checks(data, diagnostics, weights)
    supported = diagnostics['STATE_BIO50']['support'].any(1)
    support['folds'] = [dict(fold=r['fold'], subsets={name: dict(
        n=int((mask & (allocation == r['fold'])).sum()),
        models={a: _subset_scores(data[a], mask & (allocation == r['fold'])) for a in ARMS})
        for name, mask in (('supported', supported), ('unsupported', ~supported))}) for r in manifest['folds']]
    directions = direction_checks(data, ids, manifest['folds'], weights, supported)
    actual = data['A_FROZEN']['actual'][:, 2]
    result = dict(complete=True, n=len(ids), arms=list(ARMS), fixed_epoch=EPOCHS, samples=SAMPLES,
        models=models, comparisons=comparisons, layout_block_sensitivity=layout_comparisons,
        support_analysis=support, direction_analysis=directions, sources=sources,
        training_checks=training, primary_comparison=list(PRIMARY),
        baseline_population=dict(mean_gamma=float(actual.mean()), null_count=int((actual<=0).sum()),
            null_rate=float((actual<=0).mean()), positive_count=int((actual>=.005).sum())),
        uncertainty_scope=SCOPE, bootstrap_repeats=BOOTSTRAP,
        principal_policy=dict(extra_physical_well_budget_fraction=.25, action_cost_wells=2,
            ranking='descending expected Gamma within each original outer fold', null_threshold=0),
        covariance='unchanged original fold RIDGE OOF covariance',
        predictive_sampling='original matched 10000-draw evaluation protocol; saved means and NULL probabilities verified against all draws',
        formal_certificate=False, certificate_status='noCERT', original_endpoint_changed=False,
        original_contract_changed=False, final_opened=False, fifth_repeat_opened=False)
    for arm in ARMS:
        numeric = {f'diagnostic_{key}': value for key, value in diagnostics[arm].items()}
        np.savez_compressed(root/f'{arm}_oof.npz', ids=ids, fold=allocation, **data[arm], **numeric)
    _report(root, result)
    write_json(root/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    summarize(parser.parse_args().output)
