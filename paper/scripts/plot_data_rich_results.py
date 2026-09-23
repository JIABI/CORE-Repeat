"""Saved observations, fixed policies: figures 2, 4 and 5. No fitting/sampling."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from release_paths import ROOT, DATA, OUT, QA, RESEARCH

REPO = RESEARCH
AUDIT = REPO/'reports/unused_findings_audit_20260921_v1'
from release_qa import audit_layout

C = dict(core='#0072B2', hist='#D55E00', bio='#8E44AD', gray='#69737C',
         faint='#C5CBD0', ink='#182838', grid='#DDE3E7')
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8, 'axes.labelsize':8, 'axes.titlesize':8.5,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
    'legend.frameon':False, 'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'xtick.major.size':3,'ytick.major.size':3,
    'text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['ink'],
    'ytick.color':C['ink'],'pdf.fonttype':42,'svg.fonttype':'none',
    'figure.facecolor':'white','savefig.facecolor':'white'})
DATASETS = ['EU','JUMP','LINCS','RxRx3']
STYLE = {'CORE':dict(color=C['core'],lw=1.8),
         'HistGB':dict(color=C['hist'],lw=1.5,ls=(0,(5,1.5))),
         'Constant score':dict(color=C['gray'],lw=1.2,ls=(0,(3,2))),
         'Random expectation':dict(color='#89949D',lw=1.3,ls=':')}

def title(ax, letter, text, x=-.13):
    ax.text(x,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
    ax.set_title(text,loc='left',fontweight='bold',pad=11)

def grid(ax,axis='y'):
    ax.grid(axis=axis,color=C['grid'],lw=.5);ax.set_axisbelow(True)

def export(fig,name,counts):
    issues=audit_layout(fig)
    QA.mkdir(exist_ok=True)
    (QA/(name+'_layout.json')).write_text(json.dumps({'issues':issues,'source_counts':counts},indent=2))
    fig.savefig(OUT/(name+'.pdf'),dpi=600)
    fig.savefig(OUT/(name+'.svg'),dpi=600)
    fig.savefig(OUT/(name+'.png'),dpi=600)
    fig.savefig(QA/(name+'_preview.png'),dpi=300)
    fig.savefig(QA/(name+'_grayscale_input.png'),dpi=300)
    from PIL import Image
    Image.open(QA/(name+'_grayscale_input.png')).convert('L').save(QA/(name+'_grayscale.png'))
    plt.close(fig)
    print(name,issues,flush=True)

def figure2_rank_profiles(q):
    """Descriptive rank profiles: pooled outcomes, not sampling intervals."""
    q = q.copy()
    q['unit_n'] = q.groupby(['dataset', 'deployment_unit']).candidate_id.transform('count')
    bounds = np.array([0., 5., 10., 15., 25., 50., 75., 100.])
    records = []
    for policy, rank in [('CORE', 'core_within_unit_rank'), ('HistGB', 'histgb_within_unit_rank')]:
        pct = 100 * (q[rank] - .5) / q.unit_n
        bins = pd.cut(pct, bounds, labels=False, include_lowest=True)
        for (dataset, bin_id), g in q.groupby([q.dataset, bins], observed=True, sort=False):
            v = g.actual_gamma
            j = int(bin_id)
            records.append(dict(policy=policy, dataset=dataset, bin_id=j,
                rank_pct_low=bounds[j], rank_pct_high=bounds[j+1],
                rank_pct_mid=(bounds[j]+bounds[j+1])/2, n=len(v),
                mean_gamma=v.mean(), null_count=int((v <= 0).sum()), null_rate=(v <= 0).mean(),
                q10=v.quantile(.1), q25=v.quantile(.25), q50=v.median(),
                q75=v.quantile(.75), q90=v.quantile(.9)))
    profiles = pd.DataFrame(records).sort_values(['policy', 'dataset', 'bin_id'])
    for policy in ['CORE', 'HistGB']:
        assert profiles[profiles.policy == policy].n.sum() == len(q)
    profiles.to_csv(QA/'data_rich_fig2_rank_profiles.csv', index=False)
    quotas = q.groupby(['dataset', 'deployment_unit'], sort=False).agg(
        n=('candidate_id', 'size'), k=('core_original_selected', 'sum')).reset_index()
    quotas['selected_fraction_percent'] = 100 * quotas.k / quotas.n
    quotas.to_csv(QA/'data_rich_fig2_rank_quotas.csv', index=False)
    return profiles, quotas

def figure2_full_heatmap(q):
    """Retain the displaced main-panel measurement map as an SI figure."""
    rows=[]; groups=[]; ends=[]; cuts=[]; row_data=[]
    width=int(q.groupby(['dataset','deployment_unit']).size().max())
    for dataset in DATASETS:
        start=len(rows)
        for unit,g in q[q.dataset==dataset].groupby('deployment_unit',sort=True):
            g=g.sort_values('core_within_unit_rank'); r=np.full(width,np.nan)
            values=g.actual_gamma.to_numpy();r[:len(values)]=values;rows.append(r)
            cuts.append(int(g.core_original_selected.sum()))
            row_data.append(dict(row=len(rows),dataset=dataset,deployment_unit=unit,n=len(g),k=cuts[-1]))
        groups.append((start+len(rows)-1)/2);ends.append(len(rows))
    matrix=np.array(rows);assert np.isfinite(matrix).sum()==len(q)==13141
    pd.DataFrame(row_data).to_csv(QA/'data_rich_fig2_heatmap_rows.csv',index=False)
    fig=plt.figure(figsize=(183/25.4,125/25.4))
    fig.text(.105,.97,'All realized follow-up gains, ordered by CORE score',fontsize=10,weight='bold',va='top')
    ax=fig.add_axes([.105,.20,.86,.67])
    cmap=LinearSegmentedColormap.from_list('net_gain',[C['hist'],'#FFFDFC',C['core']]);cmap.set_bad('#EEF1F3')
    im=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=cmap,
                 norm=TwoSlopeNorm(vmin=-.4,vcenter=0,vmax=.66),rasterized=True)
    ax.plot(np.array(cuts)-.5,np.arange(len(rows)),color=C['ink'],lw=.8,ls='--')
    for e in ends[:-1]: ax.axhline(e-.5,color='white',lw=1.2)
    ax.set(yticks=groups,yticklabels=DATASETS,xticks=[0,49,99,149,199,width-1],
           xticklabels=[1,50,100,150,200,width],xlabel='Within-unit rank by CORE score',
           ylabel='Deployment units')
    ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
    ca=fig.add_axes([.105,.075,.55,.022])
    cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[-.4,0,.3,.6])
    cb.set_label('Realized net Γ',labelpad=3);cb.ax.tick_params(labelsize=7.5);cb.outline.set_visible(False)
    fig.text(.705,.09,'Dashed: actual unit quota\nGrey: no observation',fontsize=8,va='center')
    export(fig,'figS_ranked_realized_gains',{'objects':len(q),'units':len(rows),
                                          'finite_cells':int(np.isfinite(matrix).sum())})

def figure2():
    q=pd.read_csv(DATA/'measurement_fig2_object_predictions.csv')
    b=pd.read_csv(DATA/'measurement_fig2_budget_curves.csv')
    u=pd.read_csv(DATA/'plot_inputs/fig2_deployment_unit_audit.csv')
    u.to_csv(QA/'data_rich_fig2_units.csv',index=False)
    profiles,quotas=figure2_rank_profiles(q)
    figure2_full_heatmap(q)
    fig=plt.figure(figsize=(183/25.4,190/25.4))
    fig.text(.08,.983,'Initial information concentrates follow-up value',fontsize=10,weight='bold',va='top')
    fig.legend(handles=[Line2D([],[],**s,label=k) for k,s in STYLE.items()],loc='upper center',
               bbox_to_anchor=(.52,.967),ncol=4,handlelength=2.4,columnspacing=1.1,handletextpad=.45)
    for letter,dataset,pos in zip('abcd',DATASETS,[(.09,.712,.385,.19),(.59,.712,.385,.19),
                                                    (.09,.432,.385,.19),(.59,.432,.385,.19)]):
        ax=fig.add_axes(pos); part=b[b.dataset==dataset]
        title(ax,letter,dataset)
        n=int(part.n.iloc[0]);ax.text(1,1.08,f'n = {n:,}',transform=ax.transAxes,ha='right',fontsize=7.5)
        for method in ['Random expectation','Constant score','HistGB','CORE']:
            s=part[part.method==method]
            ax.plot(s.selected_fraction*100,s.cumulative_net_gamma,**STYLE[method])
            r=s[s.original_budget].iloc[0]
            ax.scatter(r.selected_fraction*100,r.cumulative_net_gamma,s=20,
                       marker='D' if method=='HistGB' else 'o',color=STYLE[method]['color'],
                       edgecolor='white',lw=.5,zorder=5)
        s=part[part.method=='CORE']; total=s.iloc[-1].cumulative_net_gamma
        half=s[s.cumulative_net_gamma>=total/2].iloc[0]
        ax.plot([0,half.selected_fraction*100,half.selected_fraction*100],
                [total/2,total/2,0],color=C['core'],lw=.8,ls=':')
        ax.annotate(f'Half of full-library gain\nat {half.selected_fraction*100:.0f}% of candidates',
                    xy=(half.selected_fraction*100,total/2),xytext=(.39,.18),textcoords='axes fraction',
                    fontsize=7.5,color=C['core'],ha='left',va='center',
                    arrowprops=dict(arrowstyle='-',color=C['core'],lw=.65))
        ax.set(xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Candidates receiving two added wells (%)',
               ylabel='Cumulative net Γ')
        lo=min(0,float(part.cumulative_net_gamma.min())); hi=float(part.cumulative_net_gamma.max())
        ax.set_ylim(lo-(hi-lo)*.07,hi+(hi-lo)*.13);grid(ax)
    # Rank-bin profiles give each dataset equal visual weight. The ribbons are
    # realized-outcome quantiles, not confidence intervals for the bin means.
    fig.text(.040,.340,'e',fontweight='bold',fontsize=10,va='bottom')
    fig.text(.090,.340,'Realized outcomes along CORE ranks',fontweight='bold',fontsize=8.5,va='bottom')
    for j,dataset in enumerate(DATASETS):
        left=.090+j*.149
        a=fig.add_axes([left,.189,.126,.112])
        z=fig.add_axes([left,.079,.126,.074])
        v=profiles[(profiles.policy=='CORE')&(profiles.dataset==dataset)].sort_values('bin_id')
        ranks=v.rank_pct_mid.to_numpy();low=v.q10.to_numpy();high=v.q90.to_numpy()
        a.fill_between(ranks,low,high,color=C['core'],alpha=.18,lw=0,zorder=1)
        a.plot(ranks,v.mean_gamma,color=C['core'],lw=1.45,marker='o',ms=2.7,zorder=4)
        z.plot(ranks,v.null_rate*100,color=C['core'],lw=1.45,marker='s',ms=2.7,zorder=4)
        a.set_title(dataset,fontsize=8,fontweight='bold',pad=7)
        for axis in [a,z]:
            limits=quotas.loc[quotas.dataset==dataset,'selected_fraction_percent']
            for cutoff in sorted(limits.unique()):
                axis.axvline(cutoff,color=C['gray'],lw=.55,ls=(0,(2,2)),alpha=.55,zorder=2)
            axis.set_xlim(0,100);axis.set_xticks([0,50,100]);axis.tick_params(labelsize=7,length=2)
            grid(axis)
        a.axhline(0,color=C['gray'],lw=.65,zorder=2)
        a.set(ylim=(-.12,.46),yticks=[0,.2,.4],xticklabels=[])
        z.set(ylim=(0,60),yticks=[0,25,50])
        if j==0:
            a.set_ylabel('Realized Γ',fontsize=7.5,labelpad=3)
            z.set_ylabel('NULL (%)',fontsize=7.5,labelpad=3)
        else:
            a.set_yticklabels([]);z.set_yticklabels([])
    fig.text(.377,.039,'Within-unit rank percentile (best → worst)',ha='center',fontsize=7.5)
    fig.text(.090,.011,'Ribbon: 10–90% of outcomes; dashed: actual unit quotas.',fontsize=7)
    ax=fig.add_axes([.775,.111,.20,.19])
    fig.text(.717,.340,'f',fontweight='bold',fontsize=10,va='bottom')
    fig.text(.775,.340,'Unit-level value',fontweight='bold',fontsize=8.5,va='bottom')
    shapes={'EU':'o','JUMP':'s','LINCS':'^','RxRx3':'D'}
    for ds,marker in shapes.items():
        v=u[u.dataset==ds]
        ax.scatter(v.core_minus_random/v.k,v.histgb_minus_random/v.k,s=26,marker=marker,
                   facecolors='none' if ds=='RxRx3' else C['core'],edgecolors=C['core'],lw=.8,label=ds,zorder=3)
    span=(-.045,.185); ax.plot(span,span,color=C['gray'],lw=.8,ls='--')
    ax.axvline(0,color=C['gray'],lw=.6);ax.axhline(0,color=C['gray'],lw=.6)
    ax.set(xlim=span,ylim=span,xticks=[0,.08,.16],yticks=[0,.08,.16],
           xlabel='CORE − random / action',ylabel='HistGB − random / action')
    ax.text(.04,.95,'Each beats random:\n59 / 60 units',transform=ax.transAxes,fontsize=7,va='top')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.29),ncol=2,fontsize=7,
              columnspacing=.6,handletextpad=.3,handlelength=1,borderpad=.2)
    export(fig,'fig2_prediction_decision',{'objects':len(q),'units':len(u),'rank_bins_per_dataset':7,
        'rank_profile_candidates':int(profiles[profiles.policy=='CORE'].n.sum()),
        'quantile_ribbon':'10th–90th percentiles of observed outcomes, not confidence intervals',
        'quota_fraction_range_percent':[quotas.selected_fraction_percent.min(),quotas.selected_fraction_percent.max()]})

def selection_legend(fig,y):
    fig.legend(handles=[Line2D([],[],marker='o',color=C['faint'],lw=0,ms=4,label='All observed'),
                        Line2D([],[],marker='o',color=C['core'],lw=0,ms=4,label='CORE selected'),
                        Line2D([],[],marker='o',mfc='none',color=C['hist'],lw=0,ms=4.5,label='HistGB selected')],
               loc='upper center',bbox_to_anchor=(.52,y),ncol=3,columnspacing=2,handletextpad=.5)

def scatter_all(ax,q,x,y):
    valid=np.isfinite(q[[x,y]]).all(axis=1)
    ax.scatter(q.loc[valid,x],q.loc[valid,y],color=C['faint'],s=7,alpha=.5,lw=0,rasterized=True)
    for key,color,face,size in [('CORE_selected',C['core'],C['core'],13),
                               ('HISTGB_CAL_selected',C['hist'],'none',22)]:
        ix=valid&q[key]
        ax.scatter(q.loc[ix,x],q.loc[ix,y],facecolors=face,edgecolors=color,s=size,lw=.65,
                   alpha=.88,rasterized=True)
    return int(valid.sum())

def figure4():
    q=pd.read_csv(DATA/'measurement_fig4_objects.csv')
    fig=plt.figure(figsize=(183/25.4,174/25.4))
    fig.text(.08,.982,'Targeted repeats improve cross-site phenotypic agreement',fontsize=10,weight='bold',va='top')
    fig.text(.08,.95,'Frozen confirmation: 1,539 candidates · 192 actions per policy · 384 added wells',fontsize=8)
    selection_legend(fig,.927)
    counts={}
    for ds,letter,pos in [('MEDINA','a',[.10,.553,.38,.302]),('USC','b',[.60,.553,.38,.302])]:
        ax=fig.add_axes(pos);title(ax,letter,ds)
        counts[ds]=scatter_all(ax,q,ds+'_before',ds+'_after')
        limits=(-.58,.82);ax.plot(limits,limits,lw=.9,ls='--',color=C['gray'],zorder=1)
        ax.set(xlim=limits,ylim=limits,xticks=[-.4,0,.4,.8],yticks=[-.4,0,.4,.8],
               xlabel='Agreement before additional wells',ylabel='Agreement after additional wells')
        ax.text(.03,.95,f'n = {counts[ds]:,} observed pairs',transform=ax.transAxes,va='top',fontsize=7.5)
        ax.text(.64,.13,'No change',transform=ax.transAxes,fontsize=7,color=C['gray'],rotation=40)
    ax=fig.add_axes([.10,.145,.38,.277]);title(ax,'c','Distribution of cross-site gains')
    for col,label,color,ls in [(None,'All observed',C['gray'],':'),
                              ('CORE_selected','CORE selected',C['core'],'-'),
                              ('HISTGB_CAL_selected','HistGB selected',C['hist'],'--')]:
        vals=q.loc[np.isfinite(q.delta)&(q[col] if col else True),'delta'].sort_values().to_numpy()
        ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',lw=1.6,color=color,ls=ls,label=f'{label} (n={len(vals):,})')
    ax.axvline(0,color=C['gray'],lw=.8,ls=':')
    ax.set(xlim=(-.3,.65),ylim=(0,1),xticks=[-.2,0,.2,.4,.6],yticks=[0,.25,.5,.75,1],
           xlabel='Paired-site agreement gain',ylabel='Cumulative fraction of objects')
    ax.legend(loc='lower right',fontsize=7,handlelength=1.5)
    ax.text(.50,.43,'CORE: +0.137 selected\n+0.047 not selected',transform=ax.transAxes,fontsize=7.5,
            color=C['core'],va='top');grid(ax)
    ax=fig.add_axes([.60,.145,.38,.277]);title(ax,'d','Gain across the two external sites')
    counts['paired_sites']=scatter_all(ax,q,'MEDINA_delta','USC_delta')
    ax.axhline(0,color=C['gray'],lw=.7,ls=':');ax.axvline(0,color=C['gray'],lw=.7,ls=':')
    ax.set(xlim=(-.45,.85),ylim=(-.45,.85),xticks=[-.4,0,.4,.8],yticks=[-.4,0,.4,.8],
           xlabel='MEDINA agreement gain',ylabel='USC agreement gain')
    ax.text(.035,.95,'All 1,515 paired observations',transform=ax.transAxes,fontsize=7.5,va='top')
    fig.text(.10,.061,'Paired-site endpoint available: CORE 189/192; HistGB 191/192.',fontsize=7.5)
    fig.text(.10,.034,'Selected/non-selected curves describe the frozen choices; paired-site bounds retain missing outcomes (Table 2).',fontsize=7)
    export(fig,'fig4_confirmation',counts)

def identified_bar(ax,x,low,high,color,width=.55):
    ax.bar(x,low,width=width,color=color,zorder=3)
    ax.bar(x,high-low,bottom=low,width=width,color='white',edgecolor=color,hatch='////',lw=.85,zorder=3)

def c3_risk_strip(fig, data, y, height, title_y, legend_y, letter='g', own_rank=False):
    """Descriptive, object-weighted errors; lines join ordered rank bands."""
    from matplotlib.ticker import FixedLocator
    model_styles = {
        'core': dict(color=C['core'], marker='o', lw=1.65, ms=3.4, ls='-'),
        'raw': dict(color=C['hist'], marker='s', lw=1.2, ms=3.0, ls='--'),
        'cal': dict(color=C['bio'], marker='^', lw=1.2, ms=3.3, ls=':'),
    }
    labels = {'core':'CORE', 'raw':'Quantile RAW', 'cal':'Quantile CAL'}
    heading = 'Risk across each policy’s own rank bands' if own_rank else 'Risk across fixed CORE rank bands'
    fig.text(.05, title_y, letter, fontsize=10, weight='bold', va='center')
    fig.text(.10, title_y, heading, fontsize=8.5, weight='bold', va='center')
    fig.legend(handles=[Line2D([],[],**model_styles[a],label=labels[a]) for a in model_styles],
        loc='center right', bbox_to_anchor=(.982, legend_y), ncol=3, fontsize=7,
        handlelength=1.7, handletextpad=.4, columnspacing=.85)
    for j, ds in enumerate(DATASETS):
        ax = fig.add_axes([.10 + .227*j, y, .192, height])
        d = data[data.dataset == ds]
        n = int(d[d.arm == 'core'].n.sum())
        for arm, style in model_styles.items():
            v = d[d.arm == arm].sort_values('bin_id')
            assert len(v) == 5
            ax.plot(v.bin_id, v.forecast_minus_observed_pp, **style, clip_on=False,
                markerfacecolor='white' if arm != 'core' else style['color'], markeredgewidth=.85)
        ax.axhline(0, color=C['gray'], lw=.65, zorder=0)
        ax.set_title(f'{ds} (n = {n:,})', fontsize=7.2, loc='left', fontweight='bold', pad=4)
        ax.set(xlim=(-.2,4.2), ylim=(-25,20), xticks=range(5),
               xticklabels=['0–5','5–10','10–15','15–25','≥25'], yticks=[-20,0,20])
        ax.tick_params(axis='x', labelsize=7, length=2, pad=2)
        ax.tick_params(axis='y', labelsize=7, length=2, pad=2)
        ax.yaxis.set_major_locator(FixedLocator([-20,0,20]))
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Forecast − observed\nNULL rate (pp)', fontsize=7, labelpad=3)
        ax.set_xlabel('Score-rank band (%)', fontsize=7, labelpad=3)
        grid(ax)


def figureS_c3_risk(own):
    fig = plt.figure(figsize=(183/25.4,78/25.4))
    c3_risk_strip(fig, own, y=.31, height=.37, title_y=.94, legend_y=.83,
                  letter='', own_rank=True)
    fig.text(.10,.09,'Each curve ranks its own population; observed NULL rates therefore differ between curves.',
             fontsize=7)
    export(fig,'figS_c3_own_rank_risk',{'observations_per_method':13141,'units':60,
               'bins_per_dataset':5,'population':'own-policy rank; not shared between curves'})


def figure5():
    fixed = pd.read_csv(DATA/'c3_fixed_core_rank_risk.csv')
    own = pd.read_csv(DATA/'c3_own_rank_risk.csv')
    q=pd.read_csv(DATA/'measurement_fig2_object_predictions.csv')
    risk=[]
    for (ds,unit),g in q.groupby(['dataset','deployment_unit'],sort=False):
        for method in ['core','histgb']:
            selected=g[method+'_original_selected'];v=g[selected]
            risk.append(dict(dataset=ds,unit=unit,method=method,n=len(v),
                             forecast=v[method+'_p_null'].mean()*100,observed=(v.actual_gamma<=0).mean()*100))
    risk=pd.DataFrame(risk);risk.to_csv(QA/'data_rich_fig5_unit_risk.csv',index=False)
    fig=plt.figure(figsize=(183/25.4,190/25.4))
    fig.text(.08,.983,'Selection value, risk forecasts and setup costs are distinct',fontsize=10,weight='bold',va='top')
    markers={'EU':'o','JUMP':'s','LINCS':'^','RxRx3':'D'}
    for letter,method,pos in [('a','core',[.10,.758,.37,.172]),('b','histgb',[.61,.758,.37,.172])]:
        ax=fig.add_axes(pos);name='CORE' if method=='core' else 'HistGB';color=C[method if method=='core' else 'hist']
        title(ax,letter,name+': risk in each selected set')
        for ds,m in markers.items():
            v=risk[(risk.dataset==ds)&(risk.method==method)]
            ax.scatter(v.forecast,v.observed,s=25,marker=m,facecolors='none' if ds=='RxRx3' else color,
                       edgecolors=color,lw=.8,label=ds,zorder=3)
        ax.plot([0,65],[0,65],ls='--',lw=.85,color=C['gray'])
        ax.set(xlim=(-2,65),ylim=(-2,65),xticks=[0,20,40,60],yticks=[0,20,40,60],
               xlabel='Forecast NULL rate (%)',ylabel='Observed NULL rate (%)')
        ax.text(.035,.94,'60 development units',transform=ax.transAxes,fontsize=7,va='top')
        if method=='histgb':ax.legend(loc='lower right',ncol=2,fontsize=7,columnspacing=.8,handletextpad=.2)
    values=pd.read_csv(DATA/'discovery_fig4_gain_risk.csv').set_index('policy')
    ax=fig.add_axes([.10,.505,.37,.158]);title(ax,'c','Confirmation: total net value')
    for i,key,color in [(0,'CORE',C['core']),(1,'HISTGB_CAL',C['hist']),(2,'UNIFORM_RANDOM_EXPECTATION',C['gray'])]:
        r=values.loc[key];identified_bar(ax,i,r.lower,r.upper,color)
        ax.text(i,r.upper+.9,f'{r.lower:.1f}–{r.upper:.1f}',ha='center',fontsize=7.5,color=color)
    ax.set(xticks=[0,1,2],xticklabels=['CORE','HistGB','Random'],ylim=(0,27),ylabel='Total net Γ')
    ax.tick_params(axis='x',length=0);grid(ax)
    ax=fig.add_axes([.61,.505,.37,.158]);title(ax,'d','Confirmation: forecast versus NULL',x=-.24)
    for i,key,color in [(0,'CORE',C['core']),(1,'HISTGB_CAL',C['hist'])]:
        r=values.loc[key]
        ax.bar(i-.18,r.predicted_null,width=.3,facecolor='white',edgecolor=color,lw=1.4,zorder=3)
        identified_bar(ax,i+.18,r.observed_null_lower,r.observed_null_upper,color,width=.3)
        ax.text(i-.18,r.predicted_null+1.2,f'{r.predicted_null:.2f}',ha='center',fontsize=7.5,color=color)
        ax.text(i+.18,r.observed_null_upper+1.2,f'{r.observed_null_lower:.0f}–{r.observed_null_upper:.0f}',
                ha='center',fontsize=7.5,color=color)
    ax.set(xticks=[0,1],xticklabels=['CORE','HistGB'],ylim=(0,42),yticks=[0,20,40],ylabel='NULL objects / 192 selected')
    ax.legend(handles=[Patch(facecolor='white',edgecolor=C['gray'],label='Forecast'),
                       Patch(facecolor=C['gray'],label='Observed')],loc='upper left',fontsize=7,handlelength=1)
    ax.tick_params(axis='x',length=0);grid(ax)
    ax=fig.add_axes([.10,.298,.37,.117])
    ax.text(-.13,1.43,'e',transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
    ax.set_title('Net utility after setup costs',loc='left',fontweight='bold',pad=27)
    costs=pd.read_csv(DATA/'r4_deployment_cost_sensitivity.csv')
    rows=['existing_resources','new_REF','new_REF_CAL','new_all_fitting_resources']
    periods=[1,2,5,10]
    z=np.array([[costs[(costs.policy=='CORE')&(costs.scenario==r)&(costs.amortization_campaigns==t)]
                 .total_net_value_lower.iloc[0] for t in periods] for r in rows])
    assert z.shape==(4,4) and np.isfinite(z).all()
    ax.axhline(0,color='#8A9299',lw=.75,zorder=0)
    scenarios=[('Existing',C['gray'],'s','--'),('REF',C['core'],'^',':'),
               ('REF+CAL',C['core'],'D','--'),('All fitting',C['core'],'o','-')]
    for values,(label,color,marker,ls) in zip(z,scenarios):
        ax.plot(periods,values,label=label,color=color,marker=marker,ls=ls,lw=1.2,ms=3.0,
                mfc='white' if label=='Existing' else color,mew=.7,zorder=2)
    ax.set(xlim=(.65,10.35),ylim=(-21,24),xticks=periods,yticks=[-20,0,20])
    ax.set_xlabel('Campaigns sharing setup',fontsize=7.5,labelpad=3)
    ax.set_ylabel('Net Γ (lower bound)',fontsize=7.5,labelpad=2)
    ax.tick_params(labelsize=7.5,pad=2)
    ax.legend(loc='lower left',bbox_to_anchor=(-.045,1.07),ncol=4,fontsize=7,
              handlelength=1.0,handletextpad=.3,columnspacing=.8,borderaxespad=0)
    ax=fig.add_axes([.61,.310,.37,.120]);title(ax,'f','Risk assigned to selected objects')
    risk_objects=pd.read_csv(DATA/'data_rich_fig5_selected_probabilities.csv')
    for y,key,color in [(1,'CORE',C['core']), (0,'HistGB',C['hist'])]:
        v=risk_objects[risk_objects.policy==key]
        ids=v.id.to_numpy();p=v.p_null.to_numpy();g=v.actual_gamma.to_numpy();assert len(p)==192
        # Fixed ID-based vertical jitter; no outcome-based point placement.
        ranks=np.argsort(np.argsort(ids,kind='stable'),kind='stable')
        yy=y+((ranks*0.61803398875)%1-.5)*.48
        known=np.isfinite(g);null=known&(g<=0);missing=~known
        ax.scatter(p*100,yy,s=8,color=color,alpha=.53,lw=0,zorder=2)
        ax.scatter(p[null]*100,yy[null],s=23,color=C['ink'],marker='x',lw=.9,zorder=4)
        ax.scatter(p[missing]*100,yy[missing],s=22,facecolors='white',edgecolors=C['ink'],marker='D',lw=.8,zorder=5)
    ax.set(yticks=[1,0],yticklabels=['CORE','HistGB'],ylim=(-.5,1.6),xlabel='Forecast P(NULL) (%)')
    ax.set_xlim(-2,max(45,risk_objects.p_null.max()*100+3));ax.tick_params(axis='y',length=0)
    ax.legend(handles=[Line2D([],[],color=C['ink'],marker='x',lw=0,ms=4,label='Known NULL'),
                       Line2D([],[],color=C['ink'],mfc='white',marker='D',lw=0,ms=3.5,label='Missing')],
              loc='upper right',ncol=2,fontsize=7,handlelength=.7,columnspacing=.6,handletextpad=.3)
    c3_risk_strip(fig, fixed, y=.060, height=.133, title_y=.247, legend_y=.225)
    export(fig,'fig5_risk_cost',{'development_units_per_method':60,'confirmation_candidates':1539,
         'selected_per_policy':192,'c3_development_observations_per_method':13141,
         'panel_g_population':'original CORE score-rank bins, identical rows for all methods',
         'panel_e_cost_values':int(z.size),'panel_e_quantity':'conservative net-utility lower bound',
         'panel_e_campaign_counts':periods})
    figureS_c3_risk(own)

if __name__=='__main__':
    chosen=sys.argv[1:] or ['2','4','5']
    for name in chosen:{'2':figure2,'4':figure4,'5':figure5}[name]()
