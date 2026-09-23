"""Outcome-preserving localization of existing four-well Gram predictions.

No model is fitted or sampled here. The eight task coordinates retain all six
pairwise cosines and the two acquisition amplitudes relative to X. Only the
independently irrelevant validation-vector norm is removed. All comparisons
use saved joint draws and the original three half-cosine utilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from .biology_kernel_evaluation import write_json
from .biology_kernel_experiment import _load_study_data
from .gram_evaluation import free_entries, paired_energy_score, fit_score_scale
from .gram_geometry import _gram, profiles_to_gram, gram_gains, COSINE_PAIRS
from .objective_analysis import fair_crps


PROJECT = Path(__file__).resolve().parents[1]
PARTITIONS = ("validation", "evaluation", "calibration")
ARMS = ("GLOBAL_GEOMETRY", "RIDGE_GEOMETRY", "L_GRAM", "G_DIRECT")
ACTIONS = ("ADD_ONE_Z1", "ADD_ONE_Z2", "ADD_TWO")
PRIMITIVES = ("cos_X_Z1", "cos_X_Z2", "cos_X_V", "cos_Z1_Z2", "cos_Z1_V",
              "cos_Z2_V", "log_r_Z1", "log_r_Z2", "log_r_V")
SPACES = ("original_free9", "matched_cos_log9", "task8", "angles6", "acquisition_amplitudes2")
CONFIG = dict(samples=2000, object_chunk=16, n_bootstrap=2000, seed=20260914, threads=2)


def primitive_coordinates(gram):
    """Return six cosines and three log norms relative to X, not raw norms."""
    g = _gram(gram)
    diagonal = g.diagonal(dim1=-2, dim2=-1)
    if bool((diagonal <= 0).any()):
        raise ValueError("All four profile norms must be positive")
    cosines = torch.stack([g[..., i, j]/(diagonal[..., i]*diagonal[..., j]).sqrt()
                           for i, j in COSINE_PAIRS], -1)
    log_ratios = .5*(diagonal[..., 1:]/diagonal[..., :1]).log()
    return torch.cat((cosines, log_ratios), -1)


def validation_norm_quotient(gram):
    """D G D with D=diag(1,1,1,1/sqrt(Gvv)); acquisition ratios unchanged.

    Congruence by positive D preserves PSD. Existing roundoff-tolerant PSD
    validation is used, without requiring an inverse Schur transformation.
    """
    g = _gram(gram)
    vv = g[..., 3, 3]
    if bool((vv <= 0).any()):
        raise ValueError("Validation norm must be positive")
    d = torch.ones_like(g[..., 0, :])
    d[..., 3] = vv.rsqrt()
    return _gram(d.unsqueeze(-1)*g*d.unsqueeze(-2))


def cosine_terms(gram):
    """Return [..., action, (after_A,before_B)] directly from the Gram."""
    g = _gram(gram)
    w = g.new_tensor(((.5, .5, 0, 0), (.5, 0, .5, 0), (1/3, 1/3, 1/3, 0)))
    norm2 = torch.einsum("ki,...ij,kj->...k", w, g, w)
    before_denominator = (g[..., 0, 0]*g[..., 3, 3]).sqrt()
    if bool((norm2 <= 0).any()) or bool((before_denominator <= 0).any()):
        raise ValueError("Original acquisition or validation cosine is undefined")
    after = torch.einsum("ki,...i->...k", w, g[..., :, 3]) / (norm2*g[..., 3, 3, None]).sqrt()
    before = (g[..., 0, 3]/before_denominator)[..., None].expand_as(after)
    return torch.stack((after, before), -1)


def fit_task_scales(train_grams, original_scale):
    """TRAIN score scaling only; no conditional prediction model is fitted."""
    p = primitive_coordinates(torch.as_tensor(train_grams, dtype=torch.float64)).numpy()
    if p.ndim != 2 or len(p) < 2:
        raise ValueError("Scales require at least two TRAIN objects")
    std = p.std(0)
    scale = np.where(std >= 1e-6, std, 1.)
    original_scale = np.asarray(original_scale, dtype=np.float64)
    if original_scale.shape != (9,) or not np.allclose(original_scale, fit_score_scale(train_grams), rtol=1e-12, atol=1e-14):
        raise ValueError("Original nine-entry TRAIN score scale changed")
    return dict(original_free9=original_scale, matched_cos_log9=scale, task8=scale[:8],
                angles6=scale[:6], acquisition_amplitudes2=scale[6:8],
                primitive_train_mean=p.mean(0), primitive_train_std=std,
                constant_coordinate_scale_fallback=[PRIMITIVES[i] for i in np.flatnonzero(std < 1e-6)])


def _space_arrays(grams, p):
    return dict(original_free9=free_entries(grams), matched_cos_log9=p,
                task8=p[..., :8], angles6=p[..., :6], acquisition_amplitudes2=p[..., 6:8])


def _cov(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.
    return float(np.mean((a-np.mean(a))*(b-np.mean(b))))


def _distribution_summary(a, b):
    va = float(np.var(a)) if np.ptp(a) else 0.
    vb = float(np.var(b)) if np.ptp(b) else 0.
    cab = _cov(a, b)
    vd = float(np.var(a-b)) if np.ptp(a-b) else 0.
    return dict(mean_A=float(np.mean(a)), mean_B=float(np.mean(b)), mean_delta=float(np.mean(a-b)),
        variance_A=va, variance_B=vb, covariance_AB=cab, variance_delta=vd,
        variance_identity_error=vd-(va+vb-2*cab),
        covariance_cancellation_fraction=2*cab/(va+vb) if va+vb > 0 else None)


def _prediction_metrics(actual, predicted, train_mean):
    mse = float(np.mean(np.square(actual-predicted)))
    reference = float(np.mean(np.square(actual-train_mean)))
    varied = np.ptp(actual) > 0 and np.ptp(predicted) > 0
    return dict(mse=mse, r2_vs_train_mean=1-mse/reference if reference > 0 else None,
                pearson=float(np.corrcoef(actual, predicted)[0, 1]) if varied else None,
                spearman=float(spearmanr(actual, predicted).statistic) if varied else None,
                actual_mean=float(actual.mean()), predicted_mean=float(predicted.mean()),
                actual_sd=float(actual.std()), predicted_mean_sd=float(predicted.std()))


def cancellation_decomposition(actual_terms, predicted_terms, train_terms_mean,
                               within_predictive_covariance=None):
    """Exact across-object identities for two terms, their errors and delta.

    These are properties of the supplied fitted predictions and observed rows,
    not a decomposition of all information available in X.
    """
    actual = np.asarray(actual_terms, dtype=np.float64)
    predicted = np.asarray(predicted_terms, dtype=np.float64)
    if actual.ndim != 2 or actual.shape[1] != 2 or actual.shape != predicted.shape:
        raise ValueError("Term decomposition expects matching [N,2] A/B arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Every object's terms must be finite")
    a, b = actual.T
    ma, mb = predicted.T
    ea, eb = ma-a, mb-b
    mse_a, mse_b, error_product = float(np.mean(ea**2)), float(np.mean(eb**2)), float(np.mean(ea*eb))
    mse_delta = float(np.mean((ea-eb)**2))
    cross = dict(predA_actualA=_cov(ma, a), predB_actualB=_cov(mb, b),
                 predA_actualB=_cov(ma, b), predB_actualA=_cov(mb, a))
    matched = cross["predA_actualA"]+cross["predB_actualB"]
    subtracted = cross["predA_actualB"]+cross["predB_actualA"]
    delta_covariance = _cov(ma-mb, a-b)
    result = dict(actual=_distribution_summary(a, b), fitted_means=_distribution_summary(ma, mb),
        errors=_distribution_summary(ea, eb),
        mse=dict(A=mse_a, B=mse_b, error_product=error_product, delta=mse_delta,
                 identity_error=mse_delta-(mse_a+mse_b-2*error_product), gamma=mse_delta/4),
        prediction_actual_covariance=dict(**cross, matched_sum=matched, subtracted_sum=subtracted,
            delta_covariance=delta_covariance, identity_error=delta_covariance-(matched-subtracted),
            cancellation_fraction=subtracted/matched if matched != 0 else None),
        accuracy=dict(A=_prediction_metrics(a, ma, train_terms_mean[0]),
                      B=_prediction_metrics(b, mb, train_terms_mean[1]),
                      delta=_prediction_metrics(a-b, ma-mb, train_terms_mean[0]-train_terms_mean[1])),
        interpretation="fitted-mean/observed/error arithmetic, not an information-theoretic signal or noise fraction")
    if within_predictive_covariance is not None:
        c = np.asarray(within_predictive_covariance, dtype=np.float64).mean(0)
        if c.shape != (2, 2):
            raise ValueError("Within-predictive covariances must have shape [N,2,2]")
        result["mean_within_model_MC_covariance"] = dict(variance_A=float(c[0, 0]), variance_B=float(c[1, 1]),
            covariance_AB=float(c[0, 1]), variance_delta=float(c[0, 0]+c[1, 1]-2*c[0, 1]))
    return result


def analyze_samples(samples, actual_grams, scales, train_terms_mean, *, object_chunk=16):
    """Blockwise score computation on all stored draws, with no draw generation."""
    samples, actual_grams = np.asarray(samples, dtype=np.float64), np.asarray(actual_grams, dtype=np.float64)
    if (samples.ndim != 4 or samples.shape[1:] != actual_grams.shape or len(samples) < 2
            or actual_grams.shape[-2:] != (4, 4) or object_chunk < 1):
        raise ValueError("Expected samples [S,N,4,4], targets [N,4,4] and a positive object chunk")
    if not np.allclose(samples[..., 0, 0], 1., atol=1e-10, rtol=1e-10):
        raise ValueError("The stored Gram must already have common X-norm normalization")
    n = len(actual_grams)
    actual_p = primitive_coordinates(torch.from_numpy(actual_grams)).numpy()
    actual_terms = cosine_terms(torch.from_numpy(actual_grams)).numpy()
    actual_gain = gram_gains(torch.from_numpy(actual_grams)).numpy()
    actual_spaces = _space_arrays(actual_grams, actual_p)
    energy = {key:np.empty(n) for key in SPACES}
    traces = dict(actual_primitives=actual_p, primitive_mean=np.empty((n, 9)),
                  primitive_crps=np.empty((n, 9)), actual_terms=actual_terms,
                  term_mean=np.empty((n, 3, 2)), term_crps=np.empty((n, 3, 2)),
                  term_MC_covariance=np.empty((n, 3, 2, 2)),
                  actual_gamma=actual_gain, predicted_gamma=np.empty((n, 3)),
                  gamma_crps=np.empty((n, 3)), p_null=np.empty((n, 3)))
    checks = dict(max_gamma_quotient_difference=0., max_task8_quotient_difference=0.,
                  max_term_gamma_difference=0., quotient_null_flips=0, checked_draw_objects=0,
                  all_original_and_quotient_grams_PSD=True, removed_dimension="validation-vector relative norm only")
    for start in range(0, n, object_chunk):
        stop = min(n, start+object_chunk)
        g = torch.from_numpy(samples[:, start:stop])
        primitive = primitive_coordinates(g).numpy()
        terms = cosine_terms(g).numpy()
        gains = gram_gains(g).numpy()
        quotient = validation_norm_quotient(g)
        quotient_gains = gram_gains(quotient).numpy()
        quotient_p = primitive_coordinates(quotient).numpy()[..., :8]
        check_gain = float(np.max(np.abs(quotient_gains-gains)))
        checks["max_gamma_quotient_difference"] = max(checks["max_gamma_quotient_difference"], check_gain)
        checks["max_task8_quotient_difference"] = max(checks["max_task8_quotient_difference"],
                                                     float(np.max(np.abs(quotient_p-primitive[..., :8]))))
        checks["quotient_null_flips"] += int(np.count_nonzero((quotient_gains <= 0) != (gains <= 0)))
        reconstructed = .5*(terms[..., 0]-terms[..., 1])-np.array([.01, .01, .02])
        checks["max_term_gamma_difference"] = max(checks["max_term_gamma_difference"],
                                                  float(np.max(np.abs(reconstructed-gains))))
        checks["checked_draw_objects"] += len(samples)*(stop-start)
        if check_gain > 1e-11 or not np.isfinite(primitive).all():
            raise ValueError("V-length quotient failed original-utility preservation")
        traces["primitive_mean"][start:stop] = primitive.mean(0)
        traces["primitive_crps"][start:stop] = fair_crps(primitive, actual_p[start:stop])
        mean_terms = terms.mean(0)
        centered = terms-mean_terms[None]
        traces["term_mean"][start:stop] = mean_terms
        traces["term_crps"][start:stop] = fair_crps(terms, actual_terms[start:stop])
        traces["term_MC_covariance"][start:stop] = np.einsum("snai,snaj->naij", centered, centered)/len(samples)
        traces["predicted_gamma"][start:stop] = gains.mean(0)
        traces["gamma_crps"][start:stop] = fair_crps(gains, actual_gain[start:stop])
        traces["p_null"][start:stop] = (gains <= 0).mean(0)
        for name, values in _space_arrays(samples[:, start:stop], primitive).items():
            energy[name][start:stop] = paired_energy_score(values/np.asarray(scales[name]),
                                    actual_spaces[name][start:stop]/np.asarray(scales[name]))
    traces.update({"energy_"+key:value for key, value in energy.items()})
    primitive_report = []
    for j, name in enumerate(PRIMITIVES):
        primitive_report.append(dict(name=name, crps=float(traces["primitive_crps"][:, j].mean()),
            standardized_crps=float(traces["primitive_crps"][:, j].mean()/scales["matched_cos_log9"][j]),
            **_prediction_metrics(actual_p[:, j], traces["primitive_mean"][:, j], scales["primitive_train_mean"][j])))
    decompositions = {name:cancellation_decomposition(actual_terms[:, j], traces["term_mean"][:, j],
        train_terms_mean[j], traces["term_MC_covariance"][:, j]) for j, name in enumerate(ACTIONS)}
    report = dict(n=n, samples=len(samples), energy={key:float(value.mean()) for key,value in energy.items()},
        primitives=primitive_report, cancellation=decompositions, invariance_checks=checks,
        gamma_crps=traces["gamma_crps"].mean(0).tolist(),
        geometry_scores_are_not_additively_decomposable=True, no_information_upper_bound=True)
    return report, traces


def paired_comparison(left, right, indices):
    metrics = {**{name:(left["energy_"+name]-right["energy_"+name]) for name in SPACES},
               **{"crps_"+name:left["primitive_crps"][:, j]-right["primitive_crps"][:, j]
                  for j,name in enumerate(PRIMITIVES)},
               **{"gamma_crps_"+name:left["gamma_crps"][:,j]-right["gamma_crps"][:,j]
                  for j,name in enumerate(ACTIONS)}}
    return {name:dict(mean=float(values.mean()),
                     interval95=np.quantile(values[indices].mean(1), [.025,.975]).tolist())
            for name,values in metrics.items()}


def _source_paths(gram_run, simple_run, arm, part):
    if arm in ("G_DIRECT", "L_GRAM"):
        path = gram_run/"arms"/arm/part
    else:
        suffix = "calibration_forward_diagnostic" if arm == "RIDGE_GEOMETRY" and part == "calibration" else part
        path = simple_run/"arms"/arm/suffix
    return path, dict(G_validation_used_for_checkpoint_selection=arm == "G_DIRECT" and part == "validation",
                      ridge_calibration_forward_diagnostic=arm == "RIDGE_GEOMETRY" and part == "calibration")


def _markdown(summary):
    lines = ["# 原收益相关的几何定位：结果", "", "本轮不训练、不重新抽样；仅重评四模型原有的2000份联合预测。",
        "原Gram已去掉所有四孔的共同尺度。本轮八维保留六个余弦和两个追加孔相对幅度，另去掉V的独立长度。", "",
        "## 1. G 相对 GLOBAL：优势落在哪里", "",
        "表中为G−GLOBAL（负数表示该空间评分改善），括号内为95%配对区间。不同空间的绝对energy不能相减解释信息比例。", "",
        "| 分区 | 原自由Gram9D | 匹配cos/log9D | 任务8D | 六cos | 两个追加幅度 |",
        "|---|---|---|---|---|---|"]
    for part in PARTITIONS:
        p = summary["paired"][part]["G_DIRECT__minus__GLOBAL_GEOMETRY"]
        fmt = lambda r: f"{r['mean']:+.5f} [{r['interval95'][0]:+.5f}, {r['interval95'][1]:+.5f}]"
        lines.append("| "+part+" | "+" | ".join(fmt(p[key]) for key in SPACES)+" |")
    lines += ["", "匹配9D和任务8D使用相同的八个保留坐标及TRAIN尺度，区别只有V长度；原9D到匹配9D还改变了坐标和距离度量，不能归因于删掉一维。", "",
              "## 2. 余弦、追加幅度与验证长度的分项CRPS", "",
              "表中为TRAIN标准差单位下的逐坐标CRPS均值，不是联合energy的可加归因。", "",
              "| 分区 | 模型 | 六cos平均CRPS | 追加log幅度平均CRPS | V log幅度CRPS | Γ ADD_TWO CRPS |",
              "|---|---|---:|---:|---:|---:|"]
    for part in PARTITIONS:
        for arm in ARMS:
            r = summary["results"][part][arm]
            values = [q["standardized_crps"] for q in r["primitives"]]
            label = arm+(" *" if r["scope"]["ridge_calibration_forward_diagnostic"] else "")
            lines.append(f"| {part} | {label} | {np.mean(values[:6]):.5f} | {np.mean(values[6:8]):.5f} | {values[8]:.5f} | {r['gamma_crps'][2]:.5f} |")
    lines += ["", "## 3. 两项余弦相减：ADD_TWO", "",
              "A=cos(追加平均,V)，B=cos(X,V)，Δ=A−B。R²相对TRAIN真实均值；Γ的成本项不改变差分误差。", "",
              "| 分区 | 模型 | A 的R² | B 的R² | Δ 的R² | 拟合均值方差抵消比例 | 真实值方差抵消比例 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    show = lambda v: "未定义" if v is None else f"{v:.4f}"
    for part in PARTITIONS:
        for arm in ARMS:
            r = summary["results"][part][arm]["cancellation"]["ADD_TWO"]
            lines.append("| "+part+" | "+arm+" | "+" | ".join(show(v) for v in (
                r["accuracy"]["A"]["r2_vs_train_mean"], r["accuracy"]["B"]["r2_vs_train_mean"],
                r["accuracy"]["delta"]["r2_vs_train_mean"], r["fitted_means"]["covariance_cancellation_fraction"],
                r["actual"]["covariance_cancellation_fraction"]))+" |")
    lines += ["", "抵消比例=2Cov(A,B)/(Var(A)+Var(B))；拟合均值一列只描述现有预测器。常量预测的近零方差不能用该比例推断信息。完整JSON还列出误差协方差、MSE差分恒等式、预测与真实Δ协方差的四项分解以及模型内部MC协方差。", "",
              "## 4. 复算与解释范围", "",
              f"全部{summary['checks']['draw_objects']}个样本-对象组合均保留。V长度归一后的最大Γ差为{summary['checks']['max_gamma_quotient_difference']:.3g}；NULL翻转{summary['checks']['quotient_null_flips']}次。",
              f"与原保存报告的原9D energy最大差{summary['checks']['max_original_energy_reproduction_difference']:.3g}；原Γ CRPS最大差{summary['checks']['max_gamma_crps_reproduction_difference']:.3g}。",
              "", "* RIDGE calibration只包含原参数/原样本的前向数值补充；原严格逆Schur STOP未改变。G的validation用过Γ CRPS选点。所有分区都是此前反复使用的DEV，共享批次也未改变。",
              "", "这些结果能定位已拟合模型的评分优势与差分结构，不能证明所有可预测信息被消掉、确定信息上限，或由一个评分失败宣布整条模型路线不可能。", "",
              "本轮未改终点、风险合同、预算策略、FINAL或第五重复。逐对象分项与复算信息在各分区NPZ/JSON，全部模型对照和区间在summary.json。"]
    return "\n".join(lines)+"\n"


def execute(output):
    root = Path(output).resolve()
    if root != PROJECT/"runs/gram_task_geometry_20260914_v1":
        raise ValueError("Use this task's declared new output directory")
    if not (root/"PROTOCOL.md").is_file() or (root/"summary.json").exists():
        raise ValueError("A prewritten protocol and a not-yet-completed analysis are required")
    started = time.monotonic()
    torch.set_num_threads(CONFIG["threads"])
    gram_run, simple_run = PROJECT/"runs/gram_probability_20260914_v1", PROJECT/"runs/gram_simple_20260914_v1"
    manifest = json.loads((gram_run/"run_manifest.json").read_text())
    if Path(manifest["data_directory"]).resolve() != PROJECT/"data/source5_primary_fullcontrols":
        raise ValueError("The fixed opened export changed")
    for key in ("final_opened", "fifth_repeat_opened", "original_endpoint_changed", "original_contract_changed", "original_split_changed"):
        if manifest[key] is not False:
            raise ValueError("Original data/endpoint scope changed")
    ds, split, scope = _load_study_data(manifest["data_directory"])
    if ds.Y.shape != (639,4,3617) or {k:ds.ids[ix].tolist() for k,ix in split.items()} != manifest["compound_ids"]:
        raise ValueError("Original cohort or split changed")
    old_stats = json.loads((gram_run/"preprocessing.json").read_text())
    actual = profiles_to_gram(torch.as_tensor(ds.Y, dtype=torch.float64)).numpy()
    train = np.asarray(split["train"])
    scales = fit_task_scales(actual[train], old_stats["score_scale"])
    train_terms_mean = cosine_terms(torch.from_numpy(actual[train])).numpy().mean(0)
    write_json(root/"scales.json", dict(**scales, train_ids=ds.ids[train].tolist(), train_terms_mean=train_terms_mean))
    shutil.copy2(__file__, root/"gram_task_geometry_diagnostic.py")
    run_manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
        protocol=str(root/"PROTOCOL.md"), gram_run=str(gram_run), simple_run=str(simple_run),
        data_directory=manifest["data_directory"], compound_ids=manifest["compound_ids"], split_scope=scope,
        model_fits=0, new_draws=0, final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False,
        ridge_primary_calibration="NUMERICAL_STOP retained", G_validation_selected=True)
    write_json(root/"run_manifest.json", run_manifest)
    summary = dict(results={}, paired={}, config=CONFIG,
        checks=dict(draw_objects=0, max_gamma_quotient_difference=0., quotient_null_flips=0,
                    max_original_energy_reproduction_difference=0., max_gamma_crps_reproduction_difference=0.),
        score_comparison_scope="same coordinate space only; no additive energy attribution",
        interval_scope="fixed-model paired compound bootstrap; shared batches, MC integration and model search not covered",
        final_opened=False, fifth_repeat_opened=False, formal_certificate=False)
    for part in PARTITIONS:
        ix = np.asarray(split[part]); ids = ds.ids[ix]
        boot = np.random.default_rng(CONFIG["seed"]).integers(len(ix), size=(CONFIG["n_bootstrap"],len(ix)))
        summary["results"][part], per_arm = {}, {}
        for arm in ARMS:
            source, flags = _source_paths(gram_run,simple_run,arm,part)
            print(json.dumps(dict(state="ANALYZING", arm=arm, partition=part)),flush=True)
            with np.load(source/"joint_grams.npz",allow_pickle=False) as saved:
                if not np.array_equal(saved["ids"],ids) or not np.array_equal(saved["actual_grams"],actual[ix]):
                    raise ValueError("Stored sample IDs or original observed geometry changed")
                samples = saved["grams"]
            if samples.shape != (CONFIG["samples"],len(ix),4,4):
                raise ValueError("Original sample count changed")
            report,traces = analyze_samples(samples,actual[ix],scales,train_terms_mean,
                                            object_chunk=CONFIG["object_chunk"])
            prior = json.loads((source/"metrics.json").read_text())
            energy_error = abs(report["energy"]["original_free9"]-prior["joint_geometry_energy_score"])
            crps_error = float(np.max(np.abs(np.array(report["gamma_crps"])-[q["crps"] for q in prior["utility"]])))
            with np.load(source/"predictions.npz",allow_pickle=False) as previous:
                for new,old in (("predicted_gamma","predicted"),("p_null","p_null"),("gamma_crps","utility_crps")):
                    if not np.allclose(traces[new],previous[old],rtol=1e-12,atol=1e-12):
                        raise ValueError("Saved original predictive utility changed: "+new)
            if max(energy_error,crps_error) > 1e-10:
                raise ValueError("Original geometry/utility scores did not reproduce")
            report.update(scope=flags,source=str(source),original_energy_reproduction_difference=energy_error,
                          gamma_crps_reproduction_difference=crps_error)
            folder = root/part/arm;folder.mkdir(parents=True,exist_ok=False)
            write_json(folder/"diagnostic.json",report)
            np.savez_compressed(folder/"per_object.npz",ids=ids,**traces)
            summary["results"][part][arm]=report;per_arm[arm]=traces
            check=summary["checks"]
            check["draw_objects"]+=report["invariance_checks"]["checked_draw_objects"]
            check["max_gamma_quotient_difference"]=max(check["max_gamma_quotient_difference"],report["invariance_checks"]["max_gamma_quotient_difference"])
            check["quotient_null_flips"]+=report["invariance_checks"]["quotient_null_flips"]
            check["max_original_energy_reproduction_difference"]=max(check["max_original_energy_reproduction_difference"],energy_error)
            check["max_gamma_crps_reproduction_difference"]=max(check["max_gamma_crps_reproduction_difference"],crps_error)
            del samples
        summary["paired"][part] = {}
        for left,right in (("G_DIRECT","GLOBAL_GEOMETRY"),("RIDGE_GEOMETRY","GLOBAL_GEOMETRY"),
                           ("L_GRAM","GLOBAL_GEOMETRY"),("RIDGE_GEOMETRY","G_DIRECT"),
                           ("RIDGE_GEOMETRY","L_GRAM"),("G_DIRECT","L_GRAM")):
            summary["paired"][part][left+"__minus__"+right]=paired_comparison(per_arm[left],per_arm[right],boot)
    summary.update(analysis_complete=True,elapsed_seconds=time.monotonic()-started,
                   new_model_fits=0,new_MC_draws=0)
    write_json(root/"summary.json",summary)
    (root/"REPORT.md").write_text(_markdown(summary))
    print(json.dumps(dict(state="COMPLETE",seconds=summary["elapsed_seconds"],report=str(root/"REPORT.md"))),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True)
    execute(parser.parse_args().output)


if __name__ == "__main__":
    main()
