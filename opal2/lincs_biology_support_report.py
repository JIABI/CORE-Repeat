"""Retrospective saved-artifact audit of biological support and branch activity.

No data loader, fit, checkpoint selection, policy selection or Monte Carlo draw
is used. Vocabulary size, query annotation coverage, actual reference overlap,
and nonzero model readout are deliberately reported as different quantities.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .biology_kernel_evaluation import write_json
from .hierarchical_geometry_summary import _folds


ARMS = dict(A_OLD_GENERIC='old_generic', B_OLD_STRUCTURED='old_structured',
            C_BIO_GENERIC='bio_generic', D_BIO_STRUCTURED='bio_structured')
CHANNELS = ['target_profile_overlap', 'moa_profile_overlap', 'product_of_both_overlaps']


def _json(path):
    return json.loads(Path(path).read_text())


def require_complete(root, manifest):
    """Inspect completeness before creating either retrospective report file."""
    if (manifest.get('arms') != ['HR', *ARMS] or manifest.get('fixed_epochs') != 30
            or manifest['config']['folds'] != 5 or len(manifest['folds']) != 5):
        raise ValueError('Expected the declared five-fold, four-branch, fixed-30 experiment')
    ids, allocation = _folds(manifest)
    required = []
    for record in manifest['folds']:
        folder = Path(root)/'folds'/f"fold_{record['fold']}"
        required.extend(folder/name for name in ('complete.json', 'scope.json', 'biological_support.json',
            'bank.pt', 'gamma_objective_state.pt', 'HR_fit/best.pt'))
        for arm in ARMS:
            required.extend(folder/'arms'/arm/name for name in ('training_complete.json',
                'training_config.json', 'epoch0.pt', 'epoch30.pt', 'model_diagnostics.npz'))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Experiment is incomplete; no report written. Missing: '+', '.join(missing[:6]))
    for record in manifest['folds']:
        complete = _json(Path(root)/'folds'/f"fold_{record['fold']}"/'complete.json')
        if complete.get('fold') != record['fold'] or complete.get('arms') != ['HR', *ARMS] or complete.get('branch_epochs') != 30:
            raise ValueError('Fold completion record does not confirm all four epoch30 branches')
    return ids, allocation


def _active_parameters(state, anchors=64, hidden=16):
    d = 2*anchors+3
    shapes = {'local_coefficients': (d, 3), 'conditioner.0.weight': (hidden, d),
        'conditioner.0.bias': (hidden,), 'conditioner.2.weight': (9, hidden),
        'conditioner.2.bias': (9,), 'output.weight': (9, d)}
    actual = set(state)-{key for key in state if key.startswith(('bank.', 'base_hr.'))}-{'descriptor_block'}
    if actual != set(shapes):
        raise ValueError('Unexpected active or unclassified model state keys')
    for key, shape in shapes.items():
        if tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
            raise ValueError('Active parameter shape/value differs: '+key)
    parameters = {key: state[key] for key in shapes}
    if sum(value.numel() for value in parameters.values()) != 3837:
        raise ValueError('The four branches must each retain 3837 active parameters')
    return parameters


def _frozen_equal(state, reference, prefix):
    actual = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if set(actual) != set(reference) or any(not torch.equal(actual[k], reference[k]) for k in reference):
        raise ValueError('Frozen artifact changed: '+prefix)


def _tensor_mapping_equal(actual, expected, label):
    if set(actual) != set(expected) or any(not torch.equal(actual[k], expected[k]) for k in expected):
        raise ValueError('Frozen tensor mapping changed: '+label)


def activation_statistics(arrays):
    """Pool by objects/coordinates, never average unequal-fold RMS values."""
    result = {}
    for name in ('biological_readout', 'old_information_readout', 'kernel_raw', 'increment'):
        values = np.concatenate([np.asarray(row[name], float) for row in arrays])
        if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
            raise ValueError('Finite object-by-nine readouts are required: '+name)
        result[name] = dict(rms=float(np.sqrt(np.square(values).mean())),
            max_absolute=float(np.abs(values).max()),
            nonzero_object_count=int(np.any(values != 0, axis=1).sum()))
    result['objects'] = int(sum(len(row['kernel_raw']) for row in arrays))
    result['raw_decomposition_max_error'] = float(max(np.abs(
        row['biological_readout']+row['old_information_readout']-row['kernel_raw']).max() for row in arrays))
    result['biological_readout_present'] = result['biological_readout']['nonzero_object_count'] > 0
    result['interpretation'] = 'activation amplitude only; not variance explained, causal attribution, or decision improvement'
    return result


def normalized_biology(values, support, scale, mode):
    values, support = np.asarray(values, float), np.asarray(support)
    if (values.ndim != 3 or values.shape[-1] != 3 or support.shape != values.shape or support.dtype != bool
            or not np.isfinite(values).all() or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid saved biological overlap, support or fitted scale')
    if mode == 'generic':
        response = np.exp(-.5*((values-1.)/.5)**2)-np.exp(-2.)
    elif mode == 'structured':
        response = values
    else:
        raise ValueError('Unknown biological response function')
    result = np.where(support, response, 0.)/scale
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite normalized biological response')
    return result


def _range(values):
    values = np.asarray(values, float)
    return dict(minimum=float(values.min()), maximum=float(values.max()),
        rms=float(np.sqrt(np.square(values).mean())),
        absolute_quantiles=dict(zip(('50%', '90%', '99%', '100%'), np.quantile(np.abs(values), [.5, .9, .99, 1.]).tolist())))


def summarize(output):
    root = Path(output).resolve()
    manifest = _json(root/'run_manifest.json')
    ids, allocation = require_complete(root, manifest)
    groups = np.asarray(manifest['groups'], str)
    if groups.shape != ids.shape:
        raise ValueError('Connectivity groups must align with every unique OOF identity')
    rows, activations, normalized = [], {arm: [] for arm in ARMS}, {mode: [] for mode in ('generic', 'structured')}
    coverage = {name: np.zeros(3, dtype=int) for name in ('annotated_reference_queries', 'nonzero_reference_queries', 'nonzero_reference_pairs')}
    annotated = np.zeros(2, dtype=int)
    vocabulary = None
    for record in manifest['folds']:
        f = record['fold'];folder = root/'folds'/f'fold_{f}'
        scope, support_report = _json(folder/'scope.json'), _json(folder/'biological_support.json')
        fit, valid, test = [np.asarray(record[key], int) for key in ('fit', 'inner_validation', 'test')]
        if not np.array_equal(np.sort(np.concatenate((fit, valid, test))), np.arange(len(ids))):
            raise ValueError('Full fit/validation/test partition changed')
        sets = [set(groups[ii]) for ii in (fit, valid, test)]
        if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Connectivity identity crosses an outer partition')
        supervised = list(scope['commonbranchfit_ids']);references = list(scope['reference_ids'])
        if (len(set(supervised)) != len(supervised) or len(references) != 64 or len(set(references)) != 64
                or not set(supervised+references).issubset(set(ids[fit]))):
            raise ValueError('Invalid supervised/reference identities')
        if set(groups[np.isin(ids, supervised)]) & set(groups[np.isin(ids, references)]):
            raise ValueError('A reference connectivity group enters branch supervision')
        bank_payload = torch.load(folder/'bank.pt', map_location='cpu', weights_only=True)
        bank_state, bank_config = bank_payload['state_dict'], bank_payload['config']
        if bank_config['fitting_ids'] != supervised or bank_config['anchor_ids'] != references:
            raise ValueError('Mechanism bank fitting/reference identity differs')
        names = (bank_config['target_names'], bank_config['moa_names'])
        if vocabulary is None:vocabulary = names
        elif names != vocabulary:raise ValueError('Biological vocabulary changes between folds')
        hr = torch.load(folder/'HR_fit/best.pt', map_location='cpu', weights_only=True)['state_dict']
        objective = torch.load(folder/'gamma_objective_state.pt', map_location='cpu', weights_only=True)
        foldrow = dict(fold=f, original_fit_n=len(fit), supervised_n=len(supervised),
            validation_n=len(valid), test_n=len(test), reference_n=64,
            identity_and_connectivity_disjoint=True, source_scope=str((folder/'scope.json').relative_to(root)),
            biological_training_rms=bank_config['fitting_biological_rms'],
            biological_applied_scales=bank_config['applied_biological_scales'], arms={})
        first_parameters = None;shared_biology = None
        for arm, mode in ARMS.items():
            branch = folder/'arms'/arm
            complete, training = _json(branch/'training_complete.json'), _json(branch/'training_config.json')
            initial = torch.load(branch/'epoch0.pt', map_location='cpu', weights_only=True)
            final = torch.load(branch/'epoch30.pt', map_location='cpu', weights_only=True)
            expected_steps = 30*math.ceil(len(supervised)/training['config']['batch_size'])
            if (complete['actual_checkpoint_epoch'] != 30 or complete['final_epoch'] != 30
                    or complete['trainable_parameters'] != 3837 or complete['optimizer_steps'] != expected_steps
                    or final['epoch'] != 30 or final['actual_checkpoint_epoch'] != 30
                    or final['optimizer_steps'] != expected_steps
                    or initial['epoch'] != 0 or initial['actual_checkpoint_epoch'] != 0
                    or initial['optimizer_steps'] != 0):
                raise ValueError('Actual checkpoint epoch/capacity/update budget differs')
            if (training['fit_ids'] != supervised or training['validation_ids'] != ids[valid].tolist()
                    or training['config'] != manifest['config']
                    or training['training_seed'] != record['seed']+manifest['config']['branch_seed_offset']
                    or final['model_config']['mode'] != mode or final['model_config']['hidden_dim'] != 16
                    or final['model_config']['bank_config'] != bank_config):
                raise ValueError('Training membership or biological architecture differs')
            for payload in (initial, final):
                if payload['fit_ids'] != supervised or payload['validation_ids'] != ids[valid].tolist():
                    raise ValueError('Checkpoint fit/validation identity differs')
                _frozen_equal(payload['state_dict'], hr, 'base_hr.')
                _frozen_equal(payload['state_dict'], bank_state, 'bank.')
                _tensor_mapping_equal(payload['gamma_objective_state_dict'], objective, 'joint covariance/objective')
            active0, active30 = _active_parameters(initial['state_dict']), _active_parameters(final['state_dict'])
            if first_parameters is None:first_parameters = active0
            elif any(not torch.equal(active0[k], first_parameters[k]) for k in active0):
                raise ValueError('Four-arm active parameter initialization differs')
            optimizer_states = final['optimizer_state_dict']['state']
            if len(optimizer_states) != len(active30) or any(int(value['step']) != expected_steps for value in optimizer_states.values()):
                raise ValueError('Actual AdamW parameter-update counters differ')
            with np.load(branch/'model_diagnostics.npz', allow_pickle=False) as saved:
                if not np.array_equal(saved['ids'], ids[test]):
                    raise ValueError('OOF activation IDs/order differ')
                readout = {key: saved[key].copy() for key in ('biological_readout', 'old_information_readout', 'kernel_raw', 'increment')}
                if any(value.shape != (len(test), 9) for value in readout.values()):
                    raise ValueError('Saved activation dimensions differ')
                if mode.startswith('bio_'):
                    values, known = saved['biological_values'].copy(), saved['biological_support'].copy()
                    if values.shape != (len(test), 64, 3) or known.shape != values.shape or known.dtype != bool:
                        raise ValueError('Saved biological-reference support shape differs')
                    if shared_biology is None:
                        shared_biology = (values, known)
                    elif not np.array_equal(values, shared_biology[0]) or not np.array_equal(known, shared_biology[1]):
                        raise ValueError('C/D biological information differs')
                    response_mode = mode.removeprefix('bio_')
                    basis = normalized_biology(values, known, float(bank_state['biology_'+response_mode+'_scale']), response_mode)
                    normalized[response_mode].append(basis)
                elif np.any(readout['biological_readout'] != 0):
                    raise ValueError('Old-information comparator has a nonzero biological readout')
            activations[arm].append(readout)
            foldrow['arms'][arm] = dict(epoch=30, optimizer_steps=expected_steps, trainable_parameters=3837,
                same_initialization=True, frozen_hr_exact=True, frozen_bank_exact=True, frozen_objective_exact=True,
                membership_exact=True, activation=activation_statistics([readout]),
                trainable_parameter_delta_rms=float(torch.cat([(active30[k]-active0[k]).flatten() for k in active0]).square().mean().sqrt()))
        values, known = shared_biology
        nonzero = (values > 0) & known
        counts = dict(annotated_reference_queries=known.any(1).sum(0),
            nonzero_reference_queries=nonzero.any(1).sum(0), nonzero_reference_pairs=nonzero.sum((0, 1)))
        test_support = support_report['test']
        if test_support['n'] != len(test):raise ValueError('Biological support record denominator differs')
        for name, expected_name in (('annotated_reference_queries', 'known_queries_by_channel'),
                                   ('nonzero_reference_queries', 'shared_reference_queries_by_channel'),
                                   ('nonzero_reference_pairs', 'nonzero_reference_pairs_by_channel')):
            if not np.array_equal(counts[name], test_support[expected_name]):
                raise ValueError('Saved support summary differs from actual C/D diagnostic arrays')
            coverage[name] += counts[name]
        annotated += [test_support['target_known'], test_support['moa_known']]
        foldrow['oof_support'] = {name: value.tolist() for name, value in counts.items()}
        foldrow['oof_annotation_known'] = dict(target=int(test_support['target_known']), moa=int(test_support['moa_known']))
        foldrow['normalized_biological_ranges'] = {mode: _range(normalized[mode][-1]) for mode in normalized}
        rows.append(foldrow)
    pooled = {arm: activation_statistics(values) for arm, values in activations.items()}
    no_support = [arm for arm in ('C_BIO_GENERIC', 'D_BIO_STRUCTURED') if not pooled[arm]['biological_readout_present']]
    result = dict(complete=True, audited_source=str(root), n=len(ids), fold_count=5, completed_branch_fits=20,
        fixed_checkpoint_epoch=30, trainable_parameters_per_branch=3837, information_matched_C_D=True,
        vocabulary_sizes=dict(target=len(vocabulary[0]), moa=len(vocabulary[1])),
        query_annotation_known=dict(target=int(annotated[0]), moa=int(annotated[1])),
        channels=CHANNELS, oof_support={name: value.tolist() for name, value in coverage.items()},
        oof_support_fractions={name: (value/len(ids)).tolist() for name, value in coverage.items() if name.endswith('queries')},
        normalized_biological_ranges={mode: _range(np.concatenate(values)) for mode, values in normalized.items()},
        activation_by_arm=pooled, biological_arms_with_zero_readout=no_support,
        zero_readout_interpretation='no observed nonzero biological readout; not a training failure or a model-selection rule',
        folds=rows, every_id_exactly_once=True, parameters_and_frozen_artifacts_verified=True,
        new_training=False, new_data_read=False, covariance_refit=False, performance_claim=False,
        scope='saved OOF support and implementation integrity only; RMS is not causal attribution or explained variance')
    lines = ['# 生物信息支持与实现核验', '',
        f'已核验五折、20个实际30轮分支；{len(ids)}个对象各出现一次折外读出。每分支3,837个可训练参数。',
        '四臂初始化一致，HR、参考bank及联合协方差/目标尺度逐张量保持不变；训练身份与实际更新次数一致。', '',
        '## 注释覆盖不等于参考支持', '',
        f"词表包含{len(vocabulary[0])}个靶点项与{len(vocabulary[1])}个MoA项；这不是有机制信息的化合物数量。",
        f'有靶点注释的对象{annotated[0]}/{len(ids)}；有MoA注释的对象{annotated[1]}/{len(ids)}。',
        '| 通道 | 至少一个非零参考重叠的对象 | 非零对象—参考对 |',
        '|---|---:|---:|']
    for i, name in enumerate(('靶点profile重叠', 'MoA profile重叠', '两者乘积')):
        lines.append(f"|{name}|{coverage['nonzero_reference_queries'][i]}/{len(ids)}|{coverage['nonzero_reference_pairs'][i]}|")
    lines += ['', '## 模型实际读出', '', '| 分支 | 生物读出RMS | 旧信息读出RMS | 非零生物读出对象 |', '|---|---:|---:|---:|']
    for arm, value in pooled.items():
        lines.append(f"|{arm}|{value['biological_readout']['rms']:.6g}|{value['old_information_readout']['rms']:.6g}|{value['biological_readout']['nonzero_object_count']}|")
    lines += ['', '上述幅度在最终tanh之前，不是可解释方差、因果贡献或收益改善。最终均值的增量RMS、各折尺度和逐项完整核验见JSON。',
        'C/D使用相同的靶点/MoA重叠值及缺失mask，只改变新增生物响应函数。生物读出全零时如实标记，不替换模型或重新选规则。',
        '本报告不加载新的逐孔数据、不重训、不更换检查点、不改协方差，也不根据这些诊断选模型。']
    write_json(root/'BIOLOGY_SUPPORT_AND_INTEGRITY.json', result)
    (root/'BIOLOGY_SUPPORT_AND_INTEGRITY.md').write_text('\n'.join(lines)+'\n')
    return result


report = summarize


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--output', required=True)
    summarize(parser.parse_args().output)
