"""Figures from immutable snapshots of completed development evidence.

The optional --snapshot action extracts existing aggregate results once.
Subsequent plotting reads only source_data CSV files, never live run outputs.
No fitting, resampling, Monte Carlo, or synthetic generation is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import pandas as pd

from release_paths import ROOT, DATA, OUT, QA, RESEARCH
SOURCE = RESEARCH


COL = {"teal": "#0072B2", "blue": "#0072B2", "ochre": "#D55E00",
       "ink": "#182838", "gray": "#69737C", "light": "#D9DEE1"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.frameon": False, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "text.color": COL["ink"], "axes.labelcolor": COL["ink"],
    "xtick.color": COL["ink"], "ytick.color": COL["ink"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def jread(rel):
    return json.loads((SOURCE / rel).read_text())


def write_csv(name, rows):
    path = DATA / name
    if path.exists():
        raise FileExistsError(f"Snapshot already exists: {name}; never silently refresh a working-paper snapshot")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def snapshot():
    DATA.mkdir(exist_ok=True)
    common = pd.read_csv(SOURCE / "reports/r1_closure_20260918_v1/common_observed_metrics.csv")
    common["source"] = "reports/r1_closure_20260918_v1/common_observed_metrics.csv"
    write_csv("fig1_observed_cohorts.csv", common.to_dict("records"))

    rel = "runs/r1_completion_20260917_v1/summary.json"
    x = jread(rel)
    rows = []
    for model in ["AMPLITUDE", "AMPLITUDE_COUNT", "STATE8_COUNT"]:
        for target in ["log_W", "Gamma"]:
            rows.append(dict(model=model, target=target, aggregation="pooled", fold=-1,
                             n=x["n"], r2=x["models"][model][target]["r2_vs_train_constant"], source=rel))
            for fold in x["folds"]:
                rows.append(dict(model=model, target=target, aggregation="fold", fold=fold["fold"],
                                 n=fold["n_query"], r2=fold["models"][model][target]["r2_vs_train_constant"], source=rel))
    write_csv("fig1_predictability.csv", rows)

    rel = "runs/lincs_empirical_radial_20260916_v1/summary.json"
    x = jread(rel)
    rows = []
    for arm in ["GAUSSIAN", "EMP_GLOBAL"]:
        for level in x["levels"]:
            observed = x["metrics"][arm]["regions"][str(level)]["joint_coverage_by_level"]
            rows.append(dict(arm=arm, nominal=level, observed=observed,
                             deviation_pp=100*(observed-level), n=x["n"], source=rel))
    write_csv("fig1_radial_coverage.csv", rows)

    closure = "reports/r2_four_dataset_closure_20260918_v1/"
    m = pd.read_csv(SOURCE / (closure + "main_paired_intervals.csv"))
    c = pd.read_csv(SOURCE / (closure + "calibrated_risk_interface_paired.csv"))
    m = m.loc[m["contrast"].eq("DIRECT_ACCESS_MATCHED_HISTGB_COHERENT minus CORE_ORIGINAL")
              & m["metric"].isin(["crps", "value_per_candidate", "false_activation_per_candidate"])].copy()
    m["interface"] = "Coherent distribution"
    c = c.loc[c["metric"].isin(["value_per_candidate", "false_activation_per_candidate"])].copy()
    c["interface"] = "Calibrated NULL classifier"
    paired = pd.concat([m, c], ignore_index=True)
    assert set(paired["dataset"]) == {"JUMP", "LINCS", "EU"}
    write_csv("fig2_paired_comparisons.csv", paired.to_dict("records"))
    a = pd.read_csv(SOURCE / (closure + "all_arms.csv"))
    a = a.loc[a["arm"].isin(["CORE_ORIGINAL", "DIRECT_ACCESS_MATCHED_HISTGB_COHERENT",
                            "DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL"])
              & a["policy"].eq("lambda_0.2")].copy()
    write_csv("fig2_selected_counts.csv", a.to_dict("records"))

    brel = "runs/biology_borrowing_diagnostic_20260916_v2/summary.json"
    rrel = "runs/biology_random_reference_20260916_v1/summary.json"
    b, r = jread(brel), jread(rrel)
    assert b["support_counts"] == {"target": 344, "moa": 259, "union": 418}
    rows, rr = [], []
    for family in ["TARGET", "MOA"]:
        for strength, suffix in [(0.25, "025"), (0.5, "050"), (1.0, "100")]:
            arm = f"{family}_A{suffix}"
            effect = b["comparisons"][arm]["supported"]
            rand = r["results"][arm]
            core_crps = rand["real_biology"]["supported"]["scores"]["crps"] - effect["crps"]["chemistry"]["difference"]
            for block in ["chemistry", "layout"]:
                q = effect["crps"][block]
                rows.append(dict(family=family, alpha=strength, block=block, n=effect["n"],
                                 difference=q["difference"], low=q["ci95"][0], high=q["ci95"][1], source=brel))
            for rep, val in enumerate(rand["random_replicates"]):
                rr.append(dict(family=family, alpha=strength, replicate=rep, n=effect["n"],
                               difference=val["supported"]["scores"]["crps"]-core_crps, source=rrel))
    write_csv("fig3_borrowing_paired.csv", rows)
    write_csv("fig3_matched_random.csv", rr)

    rel = "runs/null_scale_oracle_20260917_v1/summary.json"
    o = jread(rel)
    h = o["metrics"]["supported"]["H2"]
    rows = [dict(origin="Observed development data", realization="Fitting realization", replicate=-1,
                 n=h["n"], relative_gain=h["real_relative_gain"], source=rel)]
    for i in range(o["replicates"]):
        for kind, key in [("Fitting realization", "null_relative_gains"),
                          ("Independent second realization", "second_relative_gains")]:
            rows.append(dict(origin="CORE self-simulation", realization=kind, replicate=i,
                             n=h["n"], relative_gain=h[key][i], source=rel))
    write_csv("fig3_oracle_replications.csv", rows)


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=11, fontweight="normal")
    ax.text(-0.04/ax.get_position().width, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=10,
            va="bottom", ha="left")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    # Fixed canvas preserves declared physical dimensions; no tight-bbox cropping.
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def figure1():
    fig, axes = plt.subplots(2, 2, figsize=(7.2047244, 5.7086614))  # 183 × 145 mm
    fig.subplots_adjust(left=.095, right=.975, top=.91, bottom=.09, wspace=.38, hspace=.59)
    a,b,c,d = axes.flat
    panel(a, "a", "A declared follow-up decision")
    a.set_axis_off(); a.set_xlim(0,1); a.set_ylim(0,1)
    for x,label,colour in [(0.1,"X",COL["teal"]),(.41,"Z1",COL["blue"]),(.60,"Z2",COL["blue"]),(.90,"V",COL["ochre"])]:
        a.add_patch(Circle((x,.73),.075,facecolor="white",edgecolor=colour,lw=1.8))
        a.text(x,.73,label,ha="center",va="center",color=colour,fontsize=10)
    a.text(.10,.53,"Observed",ha="center",fontsize=7.5)
    a.text(.50,.53,"Acquire two wells",ha="center",fontsize=7.5)
    a.text(.90,.53,"Held out",ha="center",fontsize=7.5)
    a.add_patch(FancyArrowPatch((.19,.73),(.30,.73),arrowstyle="->",mutation_scale=9,lw=1,color=COL["gray"]))
    a.text(.50,.28,"Γ = ½ [cos(mean(X, Z1, Z2), V)",ha="center",fontsize=8.3)
    a.text(.50,.13,"− cos(X, V)] − 0.02",ha="center",fontsize=8.3)
    a.text(.50,-.035,"NULL: Γ ≤ 0     |     Action cost: 0.01 per well",ha="center",fontsize=7.5)

    panel(b, "b", "Realized benefit depends on measurement roles")
    q = pd.read_csv(DATA/"fig1_observed_cohorts.csv")
    y=np.arange(3)
    for off,col,colour,marker,label in [(-.12,"null_fraction",COL["blue"],"o","NULL in original roles"),
                                     (.12,"fixed_X_three_roles_any_null_flip_fraction",COL["ochre"],"s","Any NULL switch across roles")]:
        vals=100*q[col].to_numpy()
        b.scatter(vals,y+off,s=27,color=colour,marker=marker,zorder=3,label=label)
        for x0,y0 in zip(vals,y+off):b.text(x0+1.3,y0,f"{x0:.1f}",fontsize=7,va="center",color=colour)
    b.set_yticks(y,[f"JUMP (n = 639)",f"LINCS (n = 1,188)",f"EU (n = 904)"])
    b.invert_yaxis(); b.set_xlim(0,68); b.set_ylim(3.2,-.55)
    b.set_xlabel("Objects (%)"); b.legend(loc="lower left",fontsize=7,handletextpad=.4)
    b.grid(axis="x",color=COL["light"],lw=.5)

    panel(c, "c", "Initial state predicts variation better than gain")
    q=pd.read_csv(DATA/"fig1_predictability.csv"); order=["AMPLITUDE","AMPLITUDE_COUNT","STATE8_COUNT"]
    for target,colour,label in [("log_W",COL["blue"],"Repeat dispersion (log W)"),("Gamma",COL["ochre"],"Realized net gain (Γ)")]:
        part=q[q.target.eq(target)]
        for fold in range(5):
            line=part[part.fold.eq(fold)].set_index("model").loc[order]
            c.plot(range(3),line.r2,color=colour,alpha=.16,lw=.7,zorder=1)
        pooled=part[part.aggregation.eq("pooled")].set_index("model").loc[order]
        c.plot(range(3),pooled.r2,"o-",color=colour,lw=1.6,ms=4.5,label=label,zorder=3)
        for x0,val in enumerate(pooled.r2):c.text(x0+.05,val+.015,f"{val:.3f}",color=colour,fontsize=7)
    c.axhline(0,color=COL["gray"],lw=.7,ls=":")
    c.set_xticks(range(3),["Amplitude","+ Cell count","+ Direction"])
    c.set_xlim(-.2,2.45); c.set_ylim(-.10,.56); c.set_ylabel("Out-of-fold R²")
    c.legend(loc="upper left",fontsize=7,handlelength=1.5,handletextpad=.4)
    c.text(.98,.02,"EU: n = 904; five folds",transform=c.transAxes,ha="right",fontsize=7,color=COL["gray"])

    panel(d,"d","A radial law improves joint-region calibration")
    q=pd.read_csv(DATA/"fig1_radial_coverage.csv")
    for arm,colour,marker,label in [("GAUSSIAN",COL["gray"],"s","Gaussian"),("EMP_GLOBAL",COL["teal"],"o","Empirical radial")]:
        line=q[q.arm.eq(arm)].sort_values("nominal")
        d.plot(np.arange(5),line.deviation_pp,marker=marker,color=colour,lw=1.5,ms=4,label=label)
    d.axhline(0,color=COL["ink"],ls=":",lw=.8)
    d.set_xticks(range(5),["50","80","90","95","99"]);d.set_xlabel("Nominal joint coverage (%)")
    d.set_ylabel("Coverage error (percentage points)");d.set_ylim(-8,17);d.set_xlim(-.2,4.2)
    d.legend(loc="upper right",fontsize=7)
    d.text(.98,.03,"LINCS: n = 1,188",transform=d.transAxes,ha="right",fontsize=7,color=COL["gray"])
    save(fig,"fig1_measurement_structure")


def figure2():
    fig=plt.figure(figsize=(7.2047244,6.6929134))  # 183 × 170 mm
    top=fig.add_gridspec(1,3,left=.13,right=.975,top=.815,bottom=.51,wspace=.29)
    bottom=fig.add_gridspec(1,3,left=.13,right=.975,top=.31,bottom=.10,wspace=.29)
    axs=[fig.add_subplot(top[0,i]) for i in range(3)]
    risk_axes=[fig.add_subplot(bottom[0,i]) for i in range(3)]
    q=pd.read_csv(DATA/"fig2_paired_comparisons.csv")
    settings=[("crps",1000,"Γ distribution score",r"ΔΓ-CRPS (×10$^{-3}$)",(-1.5,4.8)),
              ("value_per_candidate",1000,"Same-budget net value",r"ΔNet value / candidate (×10$^{-3}$)",(-4.5,5.2)),
              ("false_activation_per_candidate",100,"Same-budget false activations","ΔNULL selections / candidate (pp)",(-3.1,2.1))]
    dsorder=["JUMP","LINCS","EU"]; ymap={"JUMP":6,"LINCS":3.5,"EU":1}
    for i,(metric,mult,title,xlabel,xlims) in enumerate(settings):
        ax=axs[i];panel(ax,chr(ord('a')+i),title)
        ax.axvline(0,color=COL["gray"],ls=":",lw=.8,zorder=0)
        ax.set_xlim(*xlims);ax.set_ylim(-1.6,7.1);ax.set_xlabel(xlabel,fontsize=7.5,labelpad=8)
        for ds in dsorder:
            for interface,off,colour,marker in [("Coherent distribution",.35,COL["teal"],"o"),("Calibrated NULL classifier",-.35,COL["blue"],"s")]:
                rows=q[q.dataset.eq(ds)&q.metric.eq(metric)&q.interface.eq(interface)]
                if rows.empty:continue
                base=ymap[ds]+(0 if metric=="crps" else off)
                for blockoff,blockstyle in [(.075,"chemical"),(-.075,"layout")]:
                    r=rows[rows.block.str.startswith(blockstyle)].iloc[0]
                    lo,hi,point=r.ci95_low*mult,r.ci95_high*mult,r.difference*mult
                    y=base+blockoff
                    ax.plot([lo,hi],[y,y],color=colour,lw=1.15 if blockstyle=="chemical" else .8,
                            ls="-" if blockstyle=="chemical" else "--",alpha=1 if blockstyle=="chemical" else .65,zorder=2)
                ax.scatter([point],[base],marker=marker,s=24,color=colour,zorder=3)
        ax.axhline(-.35,color=COL["light"],lw=.6)
        ax.text(.5,.055,"RxRx3: tabulated separately",transform=ax.transAxes,ha="center",color=COL["gray"],fontsize=7.5)
        ax.grid(axis="x",color=COL["light"],lw=.4,alpha=.5)
        ax.tick_params(axis='y',length=0)
        ax.set_yticks([6,3.5,1])
        if i:ax.set_yticklabels([])
    axs[0].set_yticks([6,3.5,1],["JUMP\nn = 639\nk = 79","LINCS\nn = 1,188\nk = 146","EU\nn = 904\nk = 110"])
    fig.text(.55,.968,"Distribution scores and allocation outcomes",ha="center",fontsize=10,fontweight="bold")
    fig.text(.55,.937,"Paired differences: direct HistGB − CORE",ha="center",fontsize=8)
    legend=[Line2D([],[],color=COL["teal"],marker="o",lw=0,label="Coherent distribution"),
            Line2D([],[],color=COL["blue"],marker="s",lw=0,label="Calibrated NULL classifier"),
            Line2D([],[],color=COL["gray"],lw=1.1,label="Chemical-group 95% interval"),
            Line2D([],[],color=COL["gray"],ls="--",lw=.8,label="Layout 95% interval")]
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.55,.918),ncol=2,frameon=False,
               fontsize=7.5,columnspacing=1.8,handlelength=1.6)
    for ax,label in zip(axs,["← Direct score lower","Direct value higher →","← Fewer direct NULL selections"]):
        box=ax.get_position()
        fig.text((box.x0+box.x1)/2,.436,label,ha="center",fontsize=7,color=COL["gray"])

    counts=pd.read_csv(DATA/"fig2_selected_counts.csv")
    arms=["CORE_ORIGINAL","DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL"]
    for i,(ax,ds) in enumerate(zip(risk_axes,dsorder)):
        part=counts[counts.dataset.eq(ds)].set_index("arm").loc[arms]
        assert len(part)==2 and part.activated.nunique()==1
        panel(ax,chr(ord('d')+i),f"{ds}: {int(part.activated.iloc[0])} selected objects")
        for row,(_,r) in enumerate(part.iterrows()):
            ax.plot([r.predicted_null_count,r.null_selected],[row,row],color=COL["gray"],lw=1,zorder=1)
            ax.scatter([r.predicted_null_count],[row],facecolors="white",edgecolors=COL["blue"],s=33,lw=1.25,zorder=3)
            ax.scatter([r.null_selected],[row],color=COL["ochre"],marker="s",s=27,zorder=2)
            ax.text(r.predicted_null_count,row-.17,f"{r.predicted_null_count:.2f}",ha="center",va="bottom",fontsize=7,color=COL["blue"])
            ax.text(r.null_selected,row+.17,f"{int(r.null_selected)}",ha="center",va="top",fontsize=7,color=COL["ochre"])
        ax.set_xlim(-2,42);ax.set_ylim(1.65,-.65)
        ax.set_xticks([0,10,20,30,40]);ax.set_xlabel("NULL objects in selected set",fontsize=7.5)
        ax.set_yticks([0,1],["CORE","HistGB:\nCalibrated NULL"] if i==0 else ["",""])
        ax.tick_params(axis="y",length=0)
        ax.grid(axis="x",color=COL["light"],lw=.5)
    fig.text(.55,.393,"Risk forecasts within each method's selected set",ha="center",fontsize=9,fontweight="bold")
    risk_legend=[Line2D([],[],marker="o",markerfacecolor="white",markeredgecolor=COL["blue"],color="none",label="Predicted probability sum"),
                 Line2D([],[],marker="s",color=COL["ochre"],lw=0,label="Observed NULL count")]
    fig.legend(handles=risk_legend,loc="upper center",bbox_to_anchor=(.55,.379),ncol=2,fontsize=7.5,
               handletextpad=.5,columnspacing=2)
    fig.text(.55,.032,"Fixed development selections; each selected object receives two additional wells",ha="center",fontsize=7.3)
    save(fig,"fig2_prediction_decision")


def figure3():
    fig=plt.figure(figsize=(7.2047244,5.7086614))  # 183 × 145 mm
    gs=fig.add_gridspec(2,2,left=.095,right=.975,top=.91,bottom=.105,hspace=.59,wspace=.36,height_ratios=[1,1])
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
    real=pd.read_csv(DATA/"fig3_borrowing_paired.csv");rand=pd.read_csv(DATA/"fig3_matched_random.csv")
    for idx,family in enumerate(["TARGET","MOA"]):
        ax=axes[idx];n=int(real[real.family.eq(family)].n.iloc[0])
        panel(ax,chr(ord('a')+idx),f"{'Target' if family=='TARGET' else 'MoA'} references (n = {n})")
        q=real[real.family.eq(family)&real.block.eq("chemistry")].sort_values("alpha")
        ax.axhline(0,color=COL["gray"],ls=":",lw=.8)
        random_mean=[0];random_lo=[0];random_hi=[0]
        for alpha in q.alpha:
            vals=1000*rand[rand.family.eq(family)&rand.alpha.eq(alpha)].difference.to_numpy()
            assert len(vals)==20
            random_mean.append(vals.mean());random_lo.append(vals.min());random_hi.append(vals.max())
        xx=np.r_[0,q.alpha.to_numpy()]
        ax.fill_between(xx,random_lo,random_hi,color=COL["gray"],alpha=.18,lw=0,label="Random-reference range")
        ax.plot(xx,random_mean,"s--",ms=3.5,lw=1.1,color=COL["gray"],label="Random-reference mean")
        yy=np.r_[0,1000*q.difference.to_numpy()]
        ax.plot(xx,yy,"o-",color=COL["teal"],ms=4,lw=1.5,label="Biological references")
        ax.errorbar(q.alpha,1000*q.difference,yerr=np.vstack([1000*(q.difference-q.low),1000*(q.high-q.difference)]),
                    fmt="none",ecolor=COL["teal"],capsize=2,lw=.9,zorder=3)
        ax.set_xticks([0,.25,.5,1],["0","0.25","0.5","1"]);ax.set_xlim(-.04,1.07);ax.set_ylim(-.22,1.65)
        ax.set_xlabel("Borrowing strength (α)");ax.set_ylabel(r"ΔΓ-CRPS versus CORE (×10$^{-3}$)")
        if idx==0:ax.legend(loc="upper left",fontsize=6.8,handletextpad=.5,handlelength=1.5,labelspacing=.3)
        ax.text(.97,.045,"Lower is better",transform=ax.transAxes,ha="right",fontsize=7,color=COL["gray"])

    ax=axes[2];panel(ax,"c","Outcome-selected scales fit one realization, not future uncertainty")
    q=pd.read_csv(DATA/"fig3_oracle_replications.csv");sim=q[q.origin.eq("CORE self-simulation")]
    first=sim[sim.realization.eq("Fitting realization")].sort_values("replicate").relative_gain.to_numpy()*100
    second=sim[sim.realization.eq("Independent second realization")].sort_values("replicate").relative_gain.to_numpy()*100
    assert len(first)==len(second)==20
    for i,(f,s) in enumerate(zip(first,second)):
        jitter=(i-9.5)*.005
        ax.plot([.4+jitter,1.4+jitter],[f,s],color=COL["gray"],lw=.7,alpha=.32,zorder=1)
        ax.scatter([.4+jitter],[f],color=COL["blue"],s=12,zorder=2)
        ax.scatter([1.4+jitter],[s],color=COL["ochre"],s=12,zorder=2)
    actual=100*q[q.origin.eq("Observed development data")].relative_gain.iloc[0]
    ax.scatter([-.55],[actual],marker="D",s=33,color=COL["teal"],zorder=3)
    ax.text(-.55,actual+1.8,f"{actual:.2f}%",ha="center",fontsize=7.5,color=COL["teal"])
    ax.axhline(0,color=COL["gray"],lw=.8,ls=":")
    ax.set_xticks([-.55,.4,1.4],["Observed-data\nhindsight gain","Self-simulation:\nfitting realization","Same scales:\nindependent second realization"])
    ax.set_xlim(-.95,1.9);ax.set_ylim(-9,23);ax.set_ylabel("Γ-CRPS improvement (%)")
    ax.text(.98,.95,"Two-scale search; n = 1,047\n20 model-self-simulation replications",transform=ax.transAxes,ha="right",va="top",fontsize=7.3)
    save(fig,"fig3_optional_information")


def figure_s1():
    fig,axes=plt.subplots(1,3,figsize=(7.2047244,3.1496063),sharey=True)  # 183 × 80 mm
    fig.subplots_adjust(left=.18,right=.975,bottom=.24,top=.80,wspace=.25)
    q=pd.read_csv(DATA/"fig2_selected_counts.csv")
    arms=["CORE_ORIGINAL","DIRECT_ACCESS_MATCHED_HISTGB_COHERENT","DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL"]
    for i,ds in enumerate(["JUMP","LINCS","EU"]):
        ax=axes[i];part=q[q.dataset.eq(ds)].set_index("arm").loc[arms]
        panel(ax,chr(ord('a')+i),f"{ds} (k = {int(part.activated.iloc[0])})")
        for row,(_,r) in enumerate(part.iterrows()):
            ax.plot([r.predicted_null_count,r.null_selected],[row,row],color=COL["gray"],lw=1,zorder=1)
            ax.scatter([r.predicted_null_count],[row],facecolors="white",edgecolors=COL["blue"],s=31,lw=1.2,zorder=3)
            ax.scatter([r.null_selected],[row],color=COL["ochre"],marker="s",s=25,zorder=2)
        ax.set_xlim(-1,40);ax.set_ylim(2.45,-.45);ax.set_xlabel("Selected NULL count")
        ax.set_xticks([0,10,20,30,40]);ax.grid(axis="x",color=COL["light"],lw=.5)
        ax.tick_params(axis="y",length=0)
    axes[0].set_yticks([0,1,2],["CORE","HistGB:\nCoherent law","HistGB:\nCalibrated NULL"])
    handles=[Line2D([],[],marker="o",markerfacecolor="white",markeredgecolor=COL["blue"],color="none",label="Predicted sum of NULL probabilities"),
             Line2D([],[],marker="s",color=COL["ochre"],lw=0,label="Observed NULL count")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,.98),ncol=2,fontsize=7.5,handletextpad=.5,columnspacing=1.5)
    fig.text(.57,.07,"Fixed development selections; counts are not independent-trial confidence intervals",ha="center",fontsize=7.1)
    save(fig,"figS1_selection_calibration")


if __name__=="__main__":
    figure_s1()
