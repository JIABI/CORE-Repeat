"""Read saved five-arm scale-calibration outputs without fitting or selecting.

The original three arms are checked against their immutable source artifacts.
All reported policies retain the original foldwise physical-well budget.
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
from .geometry_kernel_summary import _overlap
from . import independent_biology_summary as original
from .independent_biology_summary import (
    EPOCHS, SAMPLES, SCOPE, _fixed_and_random, _format_stat, _subset_scores, read_arm,
)
from .lincs_biology_summary import bootstrap_weights, compare, difference, interval


ARMS = (*original.ARMS, 'A_PLUS_OLD_SCALED', 'A_PLUS_BIO_SCALED')
SCALED_ARMS = ARMS[3:]
PAIRS = (('A_PLUS_OLD_SCALED', 'A_PLUS_OLD'),
         ('A_PLUS_BIO_SCALED', 'A_PLUS_BIO'),
         ('A_PLUS_BIO_SCALED', 'A_PLUS_OLD_SCALED'),
         ('A_PLUS_BIO_SCALED', 'A_FROZEN'),
         ('A_PLUS_BIO', 'A_PLUS_OLD'))
BOOTSTRAP = 2000
PLAN = 'protocols/historical/SCALE_CALIBRATION_PLAN_20260915.md'


def validate_manifest(manifest):
    if manifest.get('arms') != list(ARMS):
        raise ValueError('All five declared scale-comparison arms are required')
    ids, allocation = original.validate_manifest(dict(manifest, arms=list(original.ARMS)))
    for key in ('covariance_updated_by_branch', 'reference_selection_changed',
                'independent_new_holdout', 'formal_certificate', 'endpoint_changed'):
        if manifest.get(key, False) is not False:
            raise ValueError('Protected scale-comparison scope changed: '+key)
    if not isinstance(manifest.get('reference_run'), str) or not manifest['reference_run'].strip():
        raise ValueError('The original independent-biology reference_run is required')
    return ids, allocation


def _source_checks(root, manifest):
    source = Path(manifest['reference_run']).expanduser()
    source = (source if source.is_absolute() else root/source).resolve()
    if source == root.resolve() or source in root.resolve().parents:
        raise ValueError('Write scale summaries only in a separate new run directory')
    old = json.loads((source/'run_manifest.json').read_text())
    original.validate_manifest(old)
    for key in ('ids', 'groups', 'folds', 'scopes', 'dataset', 'config'):
        if manifest.get(key) != old.get(key):
            raise ValueError('Original source cohort/scopes/config differ: '+key)
    checked = []
    for record in manifest['folds']:
        for arm in original.ARMS:
            relative = Path('folds')/f"fold_{record['fold']}"/'arms'/arm
            names = ['evaluation/metrics.json', 'evaluation/predictions.npz',
                     'evaluation/u_predictions.npz', 'model_diagnostics.npz']
            if arm != 'A_FROZEN':
                names += ['training_complete.json', 'training_config.json', 'history.jsonl']
            for name in names:
                copied, saved = root/relative/name, source/relative/name
                if not copied.is_file() or not saved.is_file() or not filecmp.cmp(copied, saved, shallow=False):
                    raise ValueError('Original copied arm artifact changed: '+str(relative/name))
            checked.append(dict(fold=record['fold'], arm=arm, source=str(source/relative),
                                identical_artifacts=names))
    return dict(reference_run=str(source), originals_identical=True, artifacts=checked,
                protocol=str(root/'PROTOCOL.md'), protocol_source_name=PLAN,
                frozen_base_reference_run=manifest.get('frozen_base_reference_run'),
                data_reference='https://github.com/broadinstitute/lincs-cell-painting')


def _training_checks(root, manifest, models):
    calibration = {arm: [] for arm in SCALED_ARMS}
    for i, record in enumerate(manifest['folds']):
        configs, counts = [], []
        for arm in ARMS[1:]:
            folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
            config = json.loads((folder/'training_config.json').read_text())
            row = models[arm]['folds'][i]
            completion = row['training']
            counts.append(completion['trainable_parameters'])
            for key in ('frozen_A_changed', 'objective_buffers_changed'):
                if completion.get(key, False) is not False:
                    raise ValueError('Frozen training state changed: '+key)
            if completion.get('disabled_equals_A', True) is not True:
                raise ValueError('Disabling the branch must restore frozen A')
            required = ('config', 'training_seed', 'fit_ids', 'validation_ids', 'loss', 'covariance')
            if any(key not in config for key in required):
                raise ValueError('Incomplete matched training configuration')
            configs.append({key: config[key] for key in required})
            if config.get('test_data_supplied', False) is not False:
                raise ValueError('Outer test data entered branch training')
            row['training_configuration'] = config
            if arm in SCALED_ARMS:
                path = folder/'aggregation_scale.json'
                metadata = json.loads(path.read_text())
                _validate_scale(metadata, config['fit_ids'])
                calibration[arm].append(dict(fold=record['fold'], source=str(path), metadata=metadata))
                with np.load(folder/'model_diagnostics.npz', allow_pickle=False) as saved:
                    rms = {}
                    for key in ('raw_basis_rms', 'scaled_basis_rms', 'raw_local_rms', 'scaled_local_rms',
                                'raw_initial_local_rms', 'scaled_initial_local_rms'):
                        if key in saved:
                            value = original._finite(saved[key], (2,), key)
                            if np.any(value < 0):
                                raise ValueError('Negative channel RMS: '+key)
                            rms[key] = value.tolist()
                    row['saved_test_channel_rms_descriptive'] = rms
                    descriptive = dict(rms)
                    for key in ('saturation_fraction', 'inherited_hr_saturation_fraction', 'raw_increment_max'):
                        if key in saved:
                            descriptive[key] = float(original._finite(saved[key], (), key))
                    calibration[arm][-1]['diagnostics'] = descriptive
            for name in ('parameter_trajectory.jsonl', 'parameter_history.jsonl'):
                path = folder/name
                if path.is_file():
                    row[name.removesuffix('.jsonl')] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(set(counts)) != 1:
            raise ValueError('The four fitted arms have unequal active capacity')
        if any(value != configs[0] for value in configs[1:]):
            raise ValueError('Fitted-arm scope, seed, objective or training budget differs')
    return calibration


def _validate_scale(metadata, fit_ids):
    if metadata.get('aggregation_scaling') != 'train_fixed' or metadata.get('fitted') is not True:
        raise ValueError('A fitted TRAIN-only aggregation scale is required')
    if metadata.get('fit_ids') != fit_ids:
        raise ValueError('Scale calibration FIT identities differ from training')
    if metadata.get('scale_max_gain') != 32. or metadata.get('scale_floor') != 1./32.:
        raise ValueError('Aggregation gain ceiling must retain the declared value 32')
    count = np.asarray(metadata.get('count'))
    if count.shape != (2,) or not np.issubdtype(count.dtype, np.integer) or np.any(count < 0) or np.any(count > len(fit_ids)):
        raise ValueError('Two valid FIT support counts are required for scale calibration')
    raw = original._finite(metadata.get('raw_s'), (2,), 'calibration raw_s')
    gain = original._finite(metadata.get('gain'), (2,), 'calibration gain')
    capped = np.asarray(metadata.get('capped'))
    if np.any(raw < 0) or np.any(gain <= 0) or capped.shape != (2,) or capped.dtype != bool:
        raise ValueError('Invalid fixed aggregation scale values')
    if np.any((count == 0) != (raw == 0)):
        raise ValueError('Zero-scale channels must use the recorded empty-channel policy')
    positive = (count > 0) & (raw > 0)
    expected = np.where(positive, 1./np.maximum(raw, 1./32.), 1.)
    if not np.allclose(gain, expected, rtol=1e-12, atol=0) or not np.array_equal(capped, positive & (raw < 1./32.)):
        raise ValueError('Saved aggregation gains do not match the declared fixed calibration rule')


def _subset_comparisons(data, subset, weights):
    if not subset.any():
        return {}
    denominator = subset.astype(float)
    result = {}
    for left, right in PAIRS:
        a, b = data[left], data[right]
        actual, null = a['actual'][:, 2], a['actual'][:, 2] <= 0
        ma, mb = a['principal_mask']*denominator, b['principal_mask']*denominator
        quantities = dict(u_mse=a['u_object_mse']-b['u_object_mse'],
            gamma_crps=a['utility_crps'][:, 2]-b['utility_crps'][:, 2],
            null_brier=np.square(a['p_null'][:, 2]-null)-np.square(b['p_null'][:, 2]-null),
            net_gain_per_eligible=(ma-mb)*actual)
        row = {key: interval(value*denominator, denominator, weights) for key, value in quantities.items()}
        for key, value in (('net_gain_per_selected', actual), ('fdp', null)):
            row[key] = (difference(ma*value, ma, mb*value, mb, weights) if ma.sum() and mb.sum()
                        else dict(estimate=None, interval95=None, valid_resamples=0))
        result[left+'__minus__'+right] = row
    return result


def support_checks(data, diagnostics, ids, weights, layout_weights):
    baseline = data['A_FROZEN']['mean_u']
    for arm in ARMS[1:]:
        if not np.array_equal(diagnostics[arm]['baseline_mean'], baseline):
            raise ValueError('A fitted branch does not preserve the identical saved A baseline')
    for scaled, unscaled in zip(SCALED_ARMS, ('A_PLUS_OLD', 'A_PLUS_BIO')):
        if not np.array_equal(diagnostics[scaled]['support'], diagnostics[unscaled]['support']):
            raise ValueError('Aggregation scale changed support: '+scaled)
    support = diagnostics['A_PLUS_BIO']['support'].any(1)
    unsupported, recovery = ~support, {}
    for arm in ('A_PLUS_BIO', 'A_PLUS_BIO_SCALED'):
        change = data[arm]['mean_u']-baseline
        if not np.array_equal(data[arm]['mean_u'][unsupported], baseline[unsupported]):
            raise ValueError('Biology changed frozen A without biological support: '+arm)
        for key in ARRAY_SHAPES:
            if not np.array_equal(data[arm][key][unsupported], data['A_FROZEN'][key][unsupported]):
                raise ValueError('Unsupported biology saved predictions/scores differ from A: '+arm+'/'+key)
        recovery[arm] = dict(exact_zero_change_without_support=True,
            unsupported_max_absolute_mean_change=float(np.abs(change[unsupported]).max()) if unsupported.any() else 0.,
            supported_mean_change_rms=float(np.sqrt(np.square(change[support]).mean())) if support.any() else 0.)
    subsets = {}
    for name, subset in (('supported', support), ('unsupported', unsupported)):
        scores = {arm: _subset_scores(data[arm], subset) for arm in ARMS}
        if subset.any():
            for arm in ARMS:
                arrays = data[arm]
                action = _metrics(arrays['actual'][subset, 2], arrays['predicted'][subset, 2], arrays['p_null'][subset, 2])
                scores[arm].update(gamma_spearman=action['spearman'],
                    rank_scope='descriptive pooled subgroup association across fitted outer-fold models')
        subsets[name] = dict(ids=ids[subset].tolist(), models=scores,
            comparisons=_subset_comparisons(data, subset, weights),
            layout_block_sensitivity=_subset_comparisons(data, subset, layout_weights))
    return dict(supported_n=int(support.sum()), unsupported_n=int(unsupported.sum()),
        channel_supported_n=diagnostics['A_PLUS_BIO']['support'].sum(0).astype(int).tolist(),
        support_unchanged_by_scale=True, exact_zero_change_without_support=True,
        recovery=recovery, subsets=subsets,
        subset_definition='positive target or MoA support in the unchanged TRAIN bank; identical original BIO mask for every arm',
        selection_not_reranked_within_subset=True)


def _training_trajectory(models):
    keys = ('fit_u_mse', 'fit_gamma_crps', 'fit_normalized_gamma_crps', 'fit_incremental_mse',
            'validation_u_mse', 'validation_gamma_crps', 'validation_normalized_gamma_crps', 'validation_incremental_mse')
    by_arm = {}
    for arm in ARMS[1:]:
        rows = []
        for fold in models[arm]['folds']:
            history = fold['history']
            first = next((row for row in history if row['epoch'] == 0), {})
            final = next(row for row in history if row['epoch'] == EPOCHS)
            values = {key: dict(epoch0=first[key], epoch30=final[key], change=final[key]-first[key])
                      for key in keys if isinstance(first.get(key), (int, float)) and isinstance(final.get(key), (int, float))}
            for scope in ('fit', 'validation'):
                terms = [scope+suffix for suffix in ('_u_mse', '_normalized_gamma_crps', '_incremental_mse')]
                if all(key in values for key in terms):
                    metric = {epoch: sum(values[key][epoch]*weight for key, weight in zip(terms, (1., 1., .1)))
                              for epoch in ('epoch0', 'epoch30')}
                    metric['change'] = metric['epoch30']-metric['epoch0']
                    values[scope+'_monitored_objective'] = metric
            rows.append(dict(fold=fold['fold'], metrics=values,
                             best_validation_epoch_descriptive=fold['training'].get('best_epoch')))
        by_arm[arm] = rows
    pairs = {}
    for scaled, unscaled in zip(SCALED_ARMS, ('A_PLUS_OLD', 'A_PLUS_BIO')):
        rows = []
        for a, b in zip(by_arm[scaled], by_arm[unscaled]):
            common = a['metrics'].keys() & b['metrics'].keys()
            rows.append(dict(fold=a['fold'], metrics={key: dict(
                epoch30_difference=a['metrics'][key]['epoch30']-b['metrics'][key]['epoch30'],
                epoch0_to_30_change_difference=a['metrics'][key]['change']-b['metrics'][key]['change']) for key in sorted(common)}))
        common = set.intersection(*(set(row['metrics']) for row in rows))
        averages = {key: dict(equal_fold_mean_epoch30_difference=float(np.mean([row['metrics'][key]['epoch30_difference'] for row in rows])),
            folds_with_lower_epoch30_value=sum(row['metrics'][key]['epoch30_difference'] < 0 for row in rows)) for key in sorted(common)}
        pairs[scaled+'__minus__'+unscaled] = dict(folds=rows, descriptive_equal_fold_averages=averages)
    return dict(models=by_arm, comparisons=pairs, checkpoint_selection=False,
                scope='fixed monitoring draws, descriptive per-fold epoch0-to-30 trajectories; averages weight folds equally, not independent replications')


def _fmt(value, digits=6):
    return 'NA' if value is None else f'{value:.{digits}f}'


def _report(root, result):
    lines = ['# LINCS：聚合尺度校准，固定第30轮五臂比较', '',
        f"{result['n']}个对象、五个原化学分组折。原三臂逐文件匹配保存来源；新增OLD_SCALED和BIO_SCALED各训练30轮。",
        '四个修正臂每折可训练参数量、训练范围、种子和目标一致。新臂仅增加FIT估计的固定聚合尺度；联合误差仍使用原RIDGE协方差。', '',
        '| 模型 | 几何MSE↓ | Γ fair CRPS↓ | NULL Brier↓ | Γ Spearman | 对象/新增孔 | 选中实际Γ↑ | NULL数 | FDP↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        model = result['models'][arm]
        action, policy = model['actions'][2], model['principal_policy']
        lines.append(f"|{arm}|{model['u_mse']:.6f}|{action['gamma_crps']:.6f}|{action['null_brier']:.6f}|"
            f"{_fmt(action['spearman'], 4)}|{policy['selected_n']}/{policy['used_wells']}|{_fmt(policy['per_selected_net_gain'])}|"
            f"{policy['selected_null_count']}|{_fmt(policy['fdp'], 4)}|")
    lines += ['', '主策略为每折25%的额外物理孔预算，ADD_TWO每对象两孔；Γ已含原成本，NULL仍为Γ≤0。', '',
        '## 配对差值与选孔变化', '', '差值为左减右，区间按化学组配对重采样。', '',
        '| 对比 | 几何MSE差及95%区间 | Γ CRPS差及95%区间 | 选中Γ差及95%区间 | FDP差及95%区间 | 选择名单变化数 |',
        '|---|---|---|---|---|---:|']
    for left, right in PAIRS:
        row = result['comparisons'][left+'__minus__'+right]
        lines.append('|'+left+'−'+right+'|'+'|'.join(_format_stat(row[key]) for key in
            ('u_mse', 'gamma_crps', 'net_gain_per_selected', 'fdp'))+f"|{row['changed_selected_objects']}|")
    lines += ['', '板布局分组敏感性、Brier差、逐折结果和新增/移出名单见summary.json。', '',
        '## 尺度是否改善优化，以及是否转化为验证和决策收益', '']
    for scaled, unscaled in zip(SCALED_ARMS, ('A_PLUS_OLD', 'A_PLUS_BIO')):
        name = scaled+'__minus__'+unscaled
        trajectory = result['training_trajectory']['comparisons'][name]['descriptive_equal_fold_averages']
        descriptions = []
        for key, label in (('fit_monitored_objective', 'FIT监测目标'), ('validation_monitored_objective', '验证监测目标'),
                           ('fit_u_mse', 'FIT几何MSE'), ('validation_u_mse', '验证几何MSE')):
            if key in trajectory:
                value = trajectory[key]
                descriptions.append(f"{label}第30轮差{value['equal_fold_mean_epoch30_difference']:+.6f}（{value['folds_with_lower_epoch30_value']}/5折下降）")
        lines.append(f"- {scaled}相对{unscaled}："+('；'.join(descriptions) if descriptions else '保存记录未提供完整监测目标，训练轨迹见JSON')+'。')
        comparison = result['comparisons'][name]
        gain = comparison['net_gain_per_selected']
        confidence = gain['interval95']
        if confidence is not None and confidence[0] > 0:
            decision = '在固定预测下的化学组配对区间中，选中Γ增加'
        elif confidence is not None and confidence[1] < 0:
            decision = '在固定预测下的化学组配对区间中，选中Γ下降'
        else:
            decision = '选中Γ的配对区间未区分出稳定的决策收益'
        lines.append(f"  {decision}；实际选中NULL数差{comparison['selected_null_count_difference']:+d}。")
    lines += ['', '训练/验证差值按折等权描述；固定监测目标=几何MSE+归一化Γ CRPS+0.1×增量MSE。验证最佳轮次只作描述，全部评估使用实际第30轮。', '']
    lines += ['尺度按两通道分别从FIT输入估计，增益上限固定为32；触及上限的通道不保证单位RMS。', '']
    for arm in SCALED_ARMS:
        records = result['aggregation_scale'][arm]
        gain = np.asarray([row['metadata']['gain'] for row in records])
        capped = np.asarray([row['metadata']['capped'] for row in records])
        channels = records[0]['metadata'].get('channels', ['通道1', '通道2'])
        lines.append(f"- {arm}："+'；'.join(f"{channel}增益{gain[:, j].min():.3f}–{gain[:, j].max():.3f}，{int(capped[:, j].sum())}/5折触及上限" for j, channel in enumerate(channels))+'。')
    lines.append('')
    support = result['support_analysis']
    lines += ['## 相同生物支持子集', '',
        f"有支持{support['supported_n']}个，无支持{support['unsupported_n']}个；尺度未改变两通道支持名单。原BIO和缩放BIO在无支持对象上均精确恢复A及其保存评分。", '',
        '| 模型（有支持子集） | 几何MSE↓ | Γ CRPS↓ | Brier↓ | Spearman | 选中实际Γ↑ | 选中NULL数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        row = support['subsets']['supported']['models'][arm]
        if not row['n']:
            continue
        policy = row['policy_restricted_from_full_cohort']
        lines.append(f"|{arm}|{row['u_mse']:.6f}|{row['gamma_crps']:.6f}|{row['null_brier']:.6f}|"
            f"{_fmt(row['gamma_spearman'], 4)}|{_fmt(policy['per_selected_net_gain'])}|{policy['selected_null_count']}|")
    random = result['same_budget_random']
    lines += ['', '子集沿用全队列选择名单，不重新排序。两套分组区间均保存在JSON。', '',
        f"同预算随机：{random['selected_n']}对象/{random['used_wells']}孔，选中Γ均值{random['mean_selected_gamma']:.6f}，平均NULL数{random['expected_null_count']:.2f}。",
        'STOP、全部Z1、全部Z2及全部Z1Z2的成本和收益保存在JSON；随机范围为观测队列条件下的分配分布。', '',
        '## 解释范围与来源', '',
        '本轮是已使用开发数据上的尺度校准比较，不进行原合同的统计认证（noCERT）。化学组及布局组区间不包含重新拟合、反复开发选择或Monte Carlo不确定性；固定孔位与共享板仍限制跨批次外推。',
        '原Γ、NULL、成本和七项合同保持不变；未打开受保护FINAL或第五重复。尺度、门控或读出变化本身不证明生物机制。',
        f"实验方案：[PROTOCOL.md](PROTOCOL.md)（源文件{PLAN}）。原三臂来源：{result['sources']['reference_run']}。",
        '每折aggregation_scale.json的FIT身份、原尺度及增益完整保存在summary.json；训练和参数轨迹仅按保存记录描述。',
        '数据与注释来源：[官方LINCS Cell Painting](https://github.com/broadinstitute/lincs-cell-painting)。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = validate_manifest(manifest)
    sources = _source_checks(root, manifest)
    data, models, diagnostics = {}, {}, {}
    for arm in ARMS:
        data[arm], rows, diagnostics[arm] = read_arm(root, manifest, ids, allocation, arm)
        if arm != 'A_FROZEN':
            for key in ('actual', 'actual_u'):
                if not np.array_equal(data[arm][key], data['A_FROZEN'][key]):
                    raise ValueError('Arms have different realized targets: '+key)
        models[arm] = dict(_model_scores(data[arm], ids, manifest['folds']), folds=rows)
    calibration = _training_checks(root, manifest, models)
    seed = int(manifest['config']['seed'])
    weights = bootstrap_weights(manifest['groups'], BOOTSTRAP, seed+701)
    layout_weights = bootstrap_weights([str(unit['layout_block']) for unit in manifest['dataset']['units']], BOOTSTRAP, seed+702)
    comparisons = {a+'__minus__'+b: compare(a, b, data, weights, manifest['folds']) for a, b in PAIRS}
    layout_comparisons = {a+'__minus__'+b: compare(a, b, data, layout_weights, manifest['folds']) for a, b in PAIRS}
    for a, b in PAIRS:
        comparisons[a+'__minus__'+b]['selection_overlap'] = _overlap(data[a]['principal_mask'], data[b]['principal_mask'], ids)
    support = support_checks(data, diagnostics, ids, weights, layout_weights)
    actual = data['A_FROZEN']['actual']
    fixed, random = _fixed_and_random(actual, manifest, seed+703)
    result = dict(complete=True, n=len(ids), arms=list(ARMS), fixed_epoch=EPOCHS, samples=SAMPLES,
        models=models, comparisons=comparisons, layout_block_sensitivity=layout_comparisons,
        support_analysis=support, aggregation_scale=calibration, training_trajectory=_training_trajectory(models),
        sources=sources, fixed_plans=fixed, same_budget_random=random,
        baseline_population=dict(mean_gamma=float(actual[:, 2].mean()), null_count=int((actual[:, 2] <= 0).sum()),
            null_rate=float((actual[:, 2] <= 0).mean()), positive_count=int((actual[:, 2] >= .005).sum())),
        uncertainty_scope=SCOPE, bootstrap_repeats=BOOTSTRAP,
        principal_policy=dict(extra_physical_well_budget_fraction=.25, action_cost_wells=2,
            ranking='descending expected Gamma within each original outer fold', null_threshold=0),
        covariance='unchanged original fold RIDGE OOF covariance', formal_certificate=False, certificate_status='noCERT',
        original_endpoint_changed=False, original_contract_changed=False, final_opened=False, fifth_repeat_opened=False)
    # Publish only after every saved arm and source consistency check passes.
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
