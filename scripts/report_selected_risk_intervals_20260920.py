"""Report absolute selected-set risk ranges from existing prediction caches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.decision_region_calibration import ARMS
from opal2.selected_risk_rate_intervals import summarize_selected_risk_rates

SOURCE = PROJECT / 'runs/selected_risk_replay_20260920_v1'
OLD_REPORT = PROJECT / 'reports/selected_risk_replay_20260920_v1'
DEST = PROJECT / 'reports/selected_risk_intervals_20260920_v1'
EXPECTED = {'EU': (5, 904), 'JUMP': (5, 639),
            'LINCS': (10, 1188), 'RxRx3': (40, 10410)}
RATE_NAMES = ('observed_query_rate', 'base_score_case_mix_rate',
              'empirical_cal_forecast_rate', 'jeffreys_cal_forecast_rate')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stat_input(name, arrays, arm):
    return dict(cell=name, cal_null=(arrays['cal_actual'] <= 0).astype(float),
        cal_selected=arrays[arm+'__cal_selected_L02'],
        cal_groups=arrays['cal_groups'], cal_layout=arrays['cal_layout'],
        query_null=(arrays['query_actual'] <= 0).astype(float),
        query_selected=arrays[arm+'__query_selected_FIXED_02'],
        query_probability=arrays[arm+'__query_p'],
        query_groups=arrays['query_groups'], query_layout=arrays['query_layout'])


def read_dataset(dataset, old_cells):
    result = []
    for path in sorted((SOURCE/dataset).glob('*/replay_predictions.npz')):
        with np.load(path, allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        name = path.parent.name
        if set(arrays['cal_groups']) & set(arrays['query_groups']):
            raise ValueError('CAL/QUERY chemical-group overlap: '+dataset+'/'+name)
        for arm in ARMS:
            row = old_cells[(dataset, name, arm)]
            cs = arrays[arm+'__cal_selected_L02'].astype(bool)
            qs = arrays[arm+'__query_selected_FIXED_02'].astype(bool)
            if int(cs.sum()) != row['cal_selected'] or int(qs.sum()) != row['query_selected']:
                raise ValueError('Saved selection count changed')
            if int((arrays['cal_actual'][cs] <= 0).sum()) != row['cal_NULL']:
                raise ValueError('CAL event count mismatch')
            if int((arrays['query_actual'][qs] <= 0).sum()) != row['actual_NULL']:
                raise ValueError('QUERY event count mismatch')
            if not np.isclose(arrays[arm+'__query_p'][qs].sum(), row['BASE_predicted_NULL'],
                              rtol=0, atol=1e-11):
                raise ValueError('Original model probabilities changed')
        result.append((name, arrays))
    count, n = EXPECTED[dataset]
    ids = np.concatenate([a['query_ids'] for _, a in result])
    if len(result) != count or len(ids) != n or len(set(ids)) != n:
        raise ValueError('Incomplete or duplicated dataset: '+dataset)
    return result


def interval_text(stat, *, cell=False):
    if stat['ci95'] is None:
        return '不可估'
    if cell and stat['noninformative_boundary']:
        return '退化：不能据此排除未见事件'
    text = '%.2f–%.2f%%' % tuple(100*x for x in stat['ci95'])
    if stat['noninformative_boundary'] or stat['contains_boundary_cells']:
        text += '*'
    return text


def flatten_stat(identity, quantity, stat, support):
    row = dict(**identity, quantity=quantity)
    for key, value in stat.items():
        if key == 'ci95':
            row['percentile_025'] = None if value is None else value[0]
            row['percentile_975'] = None if value is None else value[1]
        elif isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False)
        else:
            row[key] = value
    row['support_json'] = json.dumps(support, ensure_ascii=False)
    for role in ('cal', 'query'):
        for key, value in support[role].items():
            row[role+'_'+key] = value
    return row


def render(result):
    aggregate, units = [], []
    for item in result['results']:
        identity = {k: item[k] for k in ('dataset', 'arm', 'block')}
        stats = item['statistics']
        for quantity, stat in stats['summary'].items():
            aggregate.append(flatten_stat(identity, quantity, stat, stats['summary_support']))
        for cell in stats['cells']:
            for quantity, stat in cell['rates'].items():
                units.append(flatten_stat(dict(**identity, cell=cell['cell'],
                    fixed_query_weight=cell['fixed_query_weight']), quantity, stat, cell['support']))
    save_csv(DEST/'aggregate_rates.csv', aggregate)
    save_csv(DEST/'unit_rates.csv', units)

    lines = ['# 选中区域风险：绝对风险率与支持量', '',
        '四份开发数据、60个部署单元、五种接口均复用原固定 λ=0.2 名单。模型、概率和名单没有改变。以下为2,000次块重采样的描述性百分位范围，不是经验证的95%覆盖保证。', '',
        '## 1. CORE：风险预报与实际比例分开报告', '',
        '| 数据 | 块单位 | CAL活动块/事件块 | 原模型预计率 | CAL重放预计率 | CAL重放范围 | 实际率 | 实际率组成敏感性 |',
        '|---|---|---:|---:|---:|---|---:|---|']
    for item in result['results']:
        if item['arm'] != 'CORE':
            continue
        rates = item['statistics']['summary']
        base = rates['base_score_case_mix_rate']
        cal = rates['empirical_cal_forecast_rate']
        obs = rates['observed_query_rate']
        support = item['statistics']['summary_support']['cal']
        lines.append('| %s | %s | %d/%d | %.2f%% | %.2f%% | %s | %.2f%% | %s |' % (
            item['dataset'], item['block'], support['active_blocks'], support['null_bearing_blocks'],
            100*base['estimate'], 100*cal['estimate'],
            interval_text(cal), 100*obs['estimate'], interval_text(obs)))
    lines += ['', '*表示该列对应的CAL或QUERY角色中至少一个单元为零/全事件，经验重采样无法刻画该单元未观察到的事件类型。稀少布局产生粗粒度敏感性范围，不等同于化学组分析，也不能合并成双向依赖保证；例如JUMP只有两个布局，范围较窄不能被读成精确风险已确定。', '',
        'CAL范围围绕CAL集合重放估计，不是原模型概率的参数区间。原固定名单上的原模型预计率是固定点；对其分数重采样只测对象组成变化。实际率已经观测，其重采样范围也不是对未来NULL计数的预报。', '',
        '## 2. 强直接模型：同样报告，不为CORE单独挑选有利区间', '',
        '| 数据 | 接口 | 原模型预计率 | CAL重放预计率 | CAL范围（化学组） | 实际率 |',
        '|---|---|---:|---:|---|---:|']
    for item in result['results']:
        if item['arm'] == 'CORE' or item['block'] != 'groups':
            continue
        rates = item['statistics']['summary']
        lines.append('| %s | %s | %.2f%% | %.2f%% | %s | %.2f%% |' % (
            item['dataset'], item['arm'], 100*rates['base_score_case_mix_rate']['estimate'],
            100*rates['empirical_cal_forecast_rate']['estimate'],
            interval_text(rates['empirical_cal_forecast_rate']),
            100*rates['observed_query_rate']['estimate']))
    lines += ['', '## 3. 解释与用途', '',
        '区间不能制造新证据：零事件、很少的事件块、很少的布局、无效抽样和退化范围在逐单元文件中逐项列出。点估计与计数保留；不以“20个事件”为统一隐藏点值的门槛。', '',
        'CAL预报汇总使用原始查询预算权重；QUERY观测率与原分数组成敏感性则使用重采样QUERY权重。CAL预报和区间不使用QUERY标签；但跨折开发身份重复出现，因此汇总不属于独立确认。QUERY绝对率不因CAL分母为零而丢弃有效抽样。', '',
        '这次补报未包含模型、径向律、校准系数或λ的重拟合；未重放新campaign的top-k，未覆盖模型设定错误。实际比例落入某一范围并不说明校准成功。原配对误差分析仍保留。', '',
        '与联合分布的一致性限制只针对单独修改P(NULL)而保留原Gamma分布的后处理接口；不是原CORE内部的缺陷，也不意味着所有一致校准都必须重训。空神谕结论限于逐实现事后优化。', '',
        '完整输出：`summary.json`、`aggregate_rates.csv`、`unit_rates.csv`、`PER_UNIT.md`。所有五种接口和两种块单位都保留。CPU数值计算墙钟时间：%.2f秒。'%result['elapsed_seconds']]
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')

    lines = ['# 逐部署单元风险报告', '',
        '区间为固定模型、固定名单下的描述性块重采样范围；CAL风险预报、QUERY组成敏感性分别计算。支持元数据和有效次数全部见CSV/JSON。', '']
    for item in result['results']:
        lines += ['## %s / %s / %s' % (item['dataset'], item['arm'], item['block']), '',
            '| 单元 | CAL NULL/n | CAL活动块/事件块 | 原模型预计率 | CAL预计率 | CAL范围 | CAL有效次数 | QUERY NULL/k | QUERY活动块 | QUERY范围 |',
            '|---|---:|---:|---:|---:|---|---:|---:|---:|---|']
        for cell in item['statistics']['cells']:
            r = cell['rates']
            cal = r['empirical_cal_forecast_rate']
            obs = r['observed_query_rate']
            sc, sq = cell['support']['cal'], cell['support']['query']
            lines.append('| %s | %d/%d | %d/%d | %.2f%% | %.2f%% | %s | %d | %d/%d | %d | %s |' % (
                cell['cell'],sc['null_events'],sc['selected_objects'],sc['active_blocks'],
                sc['null_bearing_blocks'],100*r['base_score_case_mix_rate']['estimate'],
                100*cal['estimate'],interval_text(cal, cell=True),cal['valid_replicates'],
                sq['null_events'],sq['selected_objects'],sq['active_blocks'],
                interval_text(obs, cell=True)))
        lines.append('')
    (DEST/'PER_UNIT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-only', action='store_true')
    args = parser.parse_args()
    if args.reports_only:
        render(json.loads((DEST/'summary.json').read_text()))
        return
    if not (DEST/'PROTOCOL.md').exists():
        raise RuntimeError('Reporting protocol is required before computation')
    old = json.loads((OLD_REPORT/'summary.json').read_text())['results']
    old_cells = {(r['dataset'],r['cell'],r['arm']): r for r in old['risk_cells']}
    old_summary = {(r['dataset'],r['arm'],r['estimator']): r for r in old['risk']}
    start = time.perf_counter()
    results = []
    with threadpool_limits(limits=1):
        for dataset in EXPECTED:
            cells = read_dataset(dataset, old_cells)
            for arm in ARMS:
                data = [stat_input(name, arrays, arm) for name, arrays in cells]
                for block in ('groups', 'layout'):
                    stats = summarize_selected_risk_rates(data, block=block,
                                                         replicates=2000, seed=20260920)
                    for quantity, estimator in (
                        ('base_score_case_mix_rate','BASE'),
                        ('empirical_cal_forecast_rate','EMPIRICAL'),
                        ('jeffreys_cal_forecast_rate','JEFFREYS')):
                        saved = old_summary[(dataset,arm,estimator)]
                        if not np.isclose(stats['summary'][quantity]['estimate'],
                                          saved['predicted_NULL']/saved['k'],rtol=0,atol=1e-12):
                            raise ValueError('Rate no longer matches original point estimate')
                    results.append(dict(dataset=dataset, arm=arm, block=block, statistics=stats))
            print(dataset+' complete', flush=True)
    result = dict(complete=True,source=str(SOURCE),datasets=list(EXPECTED),
        cells=60,interfaces=list(ARMS),policy='original fixed lambda=0.2 per-interface list',
        new_model_fits=0,new_future_measurement_draws=0,protected_measurements_opened=False,
        full_parameter_uncertainty=False,finite_sample_guarantee=False,
        elapsed_seconds=time.perf_counter()-start,results=results)
    save_json(DEST/'summary.json', result)
    render(result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__ == '__main__':
    main()
