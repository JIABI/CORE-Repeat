"""One-pass R1 EU diagnostics; reuse previous CORE predictions without refitting.

Run from the project with its existing Python environment. Only the previously
opened complete-case FIT dataset and saved run artefacts are accepted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import traceback
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.eu_core_experiment import partitions
from opal2.eu_fit_dataset import read_rows

PHASE = PROJECT / "reports/eu_core_development_20260917_v1"
DATA = PHASE / "prepared_data_cc904"
PREVIOUS = PROJECT / "runs/eu_core_cc904_20260917_v1"
OUTPUT = PROJECT / "runs/r1_completion_20260917_v1"
SEED = 20260917
TARGETS = ("log_W", "asinh_S", "Gamma")
ARMS = ("CONSTANT", "AMPLITUDE", "AMPLITUDE_COUNT", "STATE8_COUNT")


def encode(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False)+"\n")


def gamma_from_gram(gram, x, v):
    observed = [i for i in range(4) if i != v]
    averaged_norm2 = gram[:, observed][:, :, observed].sum((1, 2))/9
    averaged_dot_v = gram[:, observed, v].sum(1)/3
    old = gram[:, x, v]/np.sqrt(gram[:, x, x]*gram[:, v, v])
    return .5*(averaged_dot_v/np.sqrt(averaged_norm2*gram[:, v, v])-old)-.02


def observables(y):
    n, _, d = y.shape
    gram = np.einsum("nrd,nsd->nrs", y, y)
    diag = np.diagonal(gram, axis1=1, axis2=2)
    if np.any(diag <= 0): raise ValueError("A measured well has zero norm")
    pairs = list(itertools.combinations(range(4), 2))
    distance = np.column_stack([(diag[:, i]+diag[:, j]-2*gram[:, i, j])/d for i,j in pairs])
    cosine = np.column_stack([gram[:, i,j]/np.sqrt(diag[:, i]*diag[:, j]) for i,j in pairs])
    future = y[:, 1:]
    average = future.mean(1)
    w = np.square(future-average[:, None]).sum((1,2))/(2*d)
    s = np.square(average).mean(1)-w/3
    np.testing.assert_allclose(w, distance[:, 3:].mean(1)/2, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(s+w, np.square(future).mean((1,2)), rtol=1e-11, atol=1e-11)
    if np.any(w <= 0): raise ValueError("Dispersion is not positive; do not add a hidden floor")
    roles = [(x,v) for x in range(4) for v in range(4) if x != v]
    gamma = np.column_stack([gamma_from_gram(gram,x,v) for x,v in roles])
    return dict(gram=gram, pair_indices=np.asarray(pairs), pair_distance=distance,
                pair_cosine=cosine, W=w, S=s, role_indices=np.asarray(roles), role_gamma=gamma)


def metric(y, prediction, baseline):
    err = np.square(prediction-y)
    base = np.square(baseline-y)
    association = None if np.ptp(prediction) == 0 else float(spearmanr(prediction,y).statistic)
    return dict(mse=float(err.mean()), train_constant_mse=float(base.mean()),
                r2_vs_train_constant=float(1-err.mean()/base.mean()), spearman=association)


def ci(values, weights):
    simulated = weights @ values
    return dict(difference=float(np.mean(values)), ci95=np.quantile(simulated,[.025,.975]))


def role_summary(obs, layout):
    roles, values = obs["role_indices"], obs["role_gamma"]
    primary = int(np.flatnonzero((roles == [0,3]).all(1))[0])
    base = values[:, primary]
    rows = []
    for col,(x,v) in enumerate(roles):
        a = values[:, col]
        rows.append(dict(x=int(x), v=int(v), z=[int(i) for i in range(4) if i not in (x,v)],
                         mean_gamma=float(a.mean()), sd_gamma=float(a.std(ddof=1)),
                         null_n=int((a<=0).sum()), positive_n=int((a>=.005).sum()),
                         gray_n=int(((a>0)&(a<.005)).sum()),
                         null_flip_vs_primary=float(np.mean((a<=0)!=(base<=0))),
                         spearman_vs_primary=float(spearmanr(a,base).statistic),
                         layout_mean_gamma={str(k):float(a[layout==k].mean()) for k in np.unique(layout)}))
    fixed_x = values[:, roles[:,0] == 0]
    switches = lambda a: np.any(a<=0,axis=1)&np.any(a>0,axis=1)
    return dict(primary_x=0, primary_v=3, assignments=rows,
                fixed_original_X_any_null_flip=float(switches(fixed_x).mean()),
                all_12_any_null_flip=float(switches(values).mean()),
                fixed_X_objectwise_role_sd_median=float(np.median(fixed_x.std(1,ddof=1))),
                all_12_objectwise_role_sd_median=float(np.median(values.std(1,ddof=1))),
                independence_claim=False, selection_of_best_role=False)


def run():
    if OUTPUT.exists():
        status = json.loads((OUTPUT/"status.json").read_text()) if (OUTPUT/"status.json").exists() else {}
        if status.get("state") == "COMPLETE":
            print("Already complete; returning saved results without rerunning.", flush=True)
            return
        raise FileExistsError("Existing partial run requires inspection, not an automatic duplicate")
    OUTPUT.mkdir(parents=True)
    started = time.monotonic()
    def status(state, **extra):
        write_json(OUTPUT/"status.json",dict(state=state,elapsed_seconds=time.monotonic()-started,**extra))
        print(state, extra, flush=True)
    try:
        status("RUNNING",stage="load opened FIT artefact")
        metadata = json.loads((DATA/"metadata.json").read_text())
        if metadata["confirmation_data_loaded"] or metadata["n"] != 904:
            raise ValueError("Unexpected data release")
        with np.load(DATA/"data.npz",allow_pickle=False) as a:
            data = {k:a[k] for k in a.files}
        with np.load(PREVIOUS/"AMP_EMP_LOCAL.npz",allow_pickle=False) as a:
            saved = {k:a[k] for k in ("ids","fold","actual")}
        np.testing.assert_array_equal(data["ids"],saved["ids"])
        if len(np.unique(data["groups"])) != 904: raise ValueError("Unexpected compound grouping")
        plan = read_rows(PHASE/"identity_split_plan.csv")
        split = [partitions(data["ids"],data["groups"],plan,f,
                 excluded_ids=metadata["excluded_incomplete_ids"]) for f in range(5)]
        np.testing.assert_array_equal(np.sort(np.concatenate([p["DEV_EVAL"] for p in split])),np.arange(904))
        status("RUNNING",stage="four-well observables and all declared role assignments")
        obs = observables(data["Y"])
        primary = gamma_from_gram(obs["gram"],0,3)
        np.testing.assert_allclose(primary,saved["actual"],rtol=1e-9,atol=1e-10)
        np.savez_compressed(OUTPUT/"observables.npz",ids=data["ids"],groups=data["groups"],layout=data["layout"],
                            cell_count=data["cell_count"],well_ids=data["well_ids"],**obs)
        write_json(OUTPUT/"roles.json",role_summary(obs,data["layout"]))
        pairs=[]
        for j,(i,k) in enumerate(obs["pair_indices"]):
            d,c = obs["pair_distance"][:,j],obs["pair_cosine"][:,j]
            pairs.append(dict(roles=[int(i),int(k)],mean_squared_distance=float(d.mean()),
                              median_squared_distance=float(np.median(d)),mean_cosine=float(c.mean()),
                              per_layout={str(l):dict(n=int((data["layout"]==l).sum()),
                                  mean_squared_distance=float(d[data["layout"]==l].mean()),
                                  mean_cosine=float(c[data["layout"]==l].mean())) for l in np.unique(data["layout"])}))
        write_json(OUTPUT/"pair_summary.json",pairs)
        x=data["Y"][:,0]; norms=np.linalg.norm(x,axis=1); direction=x/norms[:,None]
        count=data["cell_count"][:,0]; valid_count=np.isfinite(count)&(count>0)
        target=np.full((904,3),np.nan); predictions={arm:np.full_like(target,np.nan) for arm in ARMS}
        baseline=np.full_like(target,np.nan); membership=np.full(904,-1,int)
        fold_records=[]
        for f,part in enumerate(split):
            status("RUNNING",stage="fixed diagnostic estimators",fold=f)
            train,query=part["TRAIN"],part["DEV_EVAL"]
            np.testing.assert_array_equal(saved["fold"][query],np.full(len(query),f))
            sscale=float(np.median(np.abs(obs["S"][train])))
            if sscale<=0: raise ValueError("Degenerate S transform")
            y=np.column_stack((np.log(obs["W"]),np.arcsinh(obs["S"]/sscale),primary))
            count_fill=float(np.median(count[train][valid_count[train]]))
            logcount=np.log(np.where(valid_count,count,count_fill))
            pca=PCA(n_components=8,svd_solver="full").fit(direction[train])
            pcs=pca.transform(direction)
            amp=np.log(norms)[:,None]
            ampcount=np.column_stack((amp,logcount,~valid_count))
            features={"CONSTANT":np.empty((904,0)),"AMPLITUDE":amp,
                      "AMPLITUDE_COUNT":ampcount,"STATE8_COUNT":np.column_stack((ampcount,pcs))}
            ycenter=y[train].mean(0)
            baseline[query]=ycenter;target[query]=y[query];membership[query]=f
            state=dict(train_ids=data["ids"][train],query_ids=data["ids"][query],
                       pca_mean=pca.mean_,pca_components=pca.components_,pca_variance_ratio=pca.explained_variance_ratio_,
                       S_transform_scale=sscale,count_fill=count_fill,target_train_center=ycenter)
            frec=dict(fold=f,n_train=len(train),n_query=len(query),models={})
            for arm,z in features.items():
                center=z[train].mean(0)
                scale=z[train].std(0)
                scale=np.where(scale>1e-12,scale,1.)
                zt=(z[train]-center)/scale
                coef=np.linalg.solve(zt.T@zt+np.eye(z.shape[1]),zt.T@(y[train]-ycenter)) if z.shape[1] else np.empty((0,3))
                predictions[arm][query]=((z[query]-center)/scale)@coef+ycenter
                state.update({arm+"__center":center,arm+"__scale":scale,arm+"__coef":coef})
                frec["models"][arm]={t:metric(y[query,j],predictions[arm][query,j],baseline[query,j]) for j,t in enumerate(TARGETS)}
            np.savez_compressed(OUTPUT/f"fold_{f}_diagnostic_state.npz",**state)
            fold_records.append(frec)
        if any(not np.isfinite(p).all() for p in predictions.values()) or not np.isfinite(target).all():
            raise ValueError("Incomplete OOF predictions")
        np.testing.assert_array_equal(membership,saved["fold"])
        np.savez_compressed(OUTPUT/"oof_predictions.npz",ids=data["ids"],groups=data["groups"],layout=data["layout"],
                            fold=membership,target_names=np.array(TARGETS),target=target,baseline=baseline,
                            actual_Gamma=primary,actual_W=obs["W"],actual_S=obs["S"],**predictions)
        status("RUNNING",stage="paired uncertainty and reusable summaries")
        rng=np.random.default_rng(SEED)
        boot_compound=rng.multinomial(904,np.full(904,1/904),size=2000)/904
        layouts=np.unique(data["layout"]); labels=np.searchsorted(layouts,data["layout"])
        group_draw=rng.multinomial(len(layouts),np.full(len(layouts),1/len(layouts)),size=2000)
        boot_layout=group_draw[:,labels].astype(float);boot_layout/=boot_layout.sum(1,keepdims=True)
        results={arm:{t:metric(target[:,j],p[:,j],baseline[:,j]) for j,t in enumerate(TARGETS)} for arm,p in predictions.items()}
        paired={}
        for arm,ref in [("AMPLITUDE","CONSTANT"),("AMPLITUDE_COUNT","AMPLITUDE"),("STATE8_COUNT","AMPLITUDE_COUNT")]:
            paired[arm+"_minus_"+ref]={}
            for j,t in enumerate(TARGETS):
                delta=(predictions[arm][:,j]-target[:,j])**2-(predictions[ref][:,j]-target[:,j])**2
                paired[arm+"_minus_"+ref][t]=dict(compound=ci(delta,boot_compound),layout=ci(delta,boot_layout))
        result=dict(n=904,space_dimension=data["Y"].shape[-1],folds=fold_records,models=results,paired_mse=paired,
                    cell_count_available=int(valid_count.sum()),source_data=str(DATA),source_core=str(PREVIOUS),
                    reused_core_without_refitting=True,new_neural_training=False,new_monte_carlo_sampling=False,
                    confirmation_opened=False,primary_endpoint_matches_previous=True,
                    results_sections=["R1","R2 simple Gamma diagnostic reuse","R4 layout reuse"],
                    caution="S is common-energy proxy, not identified shared noise; bootstrap conditions on fitted models; role endpoints overlap")
        write_json(OUTPUT/"summary.json",result)
        roles=json.loads((OUTPUT/"roles.json").read_text())
        text=["# R1 EU diagnostic results", "", "Fixed CORE models were not retrained or resampled.","",
              "| Input | R² log W vs TRAIN constant | R² transformed S | R² Gamma | Gamma Spearman |",
              "|---|---:|---:|---:|---:|"]
        for arm,rr in results.items():
            text.append(f"| {arm} | {rr['log_W']['r2_vs_train_constant']:.6f} | {rr['asinh_S']['r2_vs_train_constant']:.6f} | {rr['Gamma']['r2_vs_train_constant']:.6f} | {rr['Gamma']['spearman'] if rr['Gamma']['spearman'] is not None else 'NA'} |")
        text += ["",f"NULL label changes across the three roles at fixed original X: {roles['fixed_original_X_any_null_flip']:.2%}.",
                 f"Across all twelve ordered X/V assignments: {roles['all_12_any_null_flip']:.2%}.",
                 "These are overlapping diagnostic endpoints, not independent observations or a physical noise decomposition.","",
                 "See summary.json for fold metrics and paired compound/layout intervals; see observables.npz and oof_predictions.npz for reuse."]
        (OUTPUT/"REPORT.md").write_text("\n".join(text)+"\n")
        status("COMPLETE",completed_folds=5,report=str(OUTPUT/"REPORT.md"))
    except Exception as exc:
        status("FAILED",error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.parse_args()
    with threadpool_limits(limits=1):
        run()
