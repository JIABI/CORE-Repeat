"""Fig. 3: association versus predictive increment; no new fitting/resampling.

Figure contract: actual pair/object observations, all 20 reference assignments,
and all 20 pairs of simulated realizations. No universal learnability claim.
183 x 185 mm; Python; editable PDF/SVG and 600 dpi PNG.

Sources relative to sibling opal2_measurement_model_r2:
a: scripts/diagnose_r3_relation_variance_20260921.py::load_units, read-only
recovery from runs/r3_crossdose_response_20260920_v1/unit_*/predictions.npz.
Only approved metadata are accessed in prepared data.npz; Y is never decoded.
b: source_data/discovery_fig3_covariance.csv (ADJUSTED_BOTH, saved intervals).
c: reports/r3_signed_program_dose_20260921_v1/jointdose_aggregate_scores.npz;
all pair-level BIO_CONTEXT_COMPONENT_CAL minus GENERIC profile MSE (index 0).
d: runs/biology_borrowing_diagnostic_20260916_v2/{CORE,TARGET_A100,MOA_A100}.npz.
e: source_data/fig3_{matched_random,borrowing_paired}.csv, all saved assignments.
f: source_data/fig3_oracle_replications.csv, all saved simulation pairs;
runs/direct_risk_scale_20260917_v1/summary.json, separate real-data OOF fit.

Panel a is raw/unadjusted/unweighted; b is an adjusted equal-unit coefficient.
c ECDF uses equal-pair weights; the text estimate is the saved equal-group
comparison. These are different estimands. All observations are retained.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import numpy as np
import pandas as pd

from release_paths import ROOT, DATA, OUT, QA, RESEARCH
REPO = RESEARCH

C = dict(blue='#0072B2', orange='#D55E00', bio='#8E44AD', gray='#69737C',
         ink='#25313A', light='#E2E6E9')
plt.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8, 'axes.labelsize':8, 'axes.titlesize':8.5,
    'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':7.5,
    'legend.frameon':False, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.linewidth':.65, 'xtick.major.size':2.5, 'ytick.major.size':2.5,
    'svg.fonttype':'none', 'pdf.fonttype':42, 'pdf.compression':9,
    'figure.facecolor':'white', 'savefig.facecolor':'white',
    'text.color':C['ink'], 'axes.labelcolor':C['ink'],
    'xtick.color':C['ink'], 'ytick.color':C['ink'],
})

def recover_pair_products():
    q=pd.read_csv(DATA/'plot_inputs/fig6_residual_pairs.csv')
    assert len(q)==93558 and int((q.target_overlap>0).sum())==3088
    return q.target_overlap.to_numpy(), q.cross_dose_residual_product.to_numpy()

def panel(fig,ax,letter,title):
    p=ax.get_position()
    fig.text(p.x0-.063,p.y1+.022,letter,fontsize=11,weight='bold',va='bottom')
    fig.text(p.x0,p.y1+.024,title,fontsize=8.7,weight='bold',va='bottom')

def grid(ax,axis='y'):
    ax.grid(axis=axis,color=C['light'],linewidth=.45)
    ax.set_axisbelow(True)

def ecdf(ax,values,**kwargs):
    x=np.sort(np.asarray(values))
    ax.step(x,100*np.arange(1,len(x)+1)/len(x),where='post',**kwargs)

def numeric_label(value, specification):
    return format(value, specification).replace('-', '−')

def swarm_offsets(values,step=.045,width=.23,bin_width=1.1):
    """Deterministic vertical packing; no random jitter/subsampling."""
    bins=np.floor(np.asarray(values)/bin_width).astype(int)
    out=np.zeros(len(values))
    for block in np.unique(bins):
        idx=np.flatnonzero(bins==block)
        idx=idx[np.argsort(np.asarray(values)[idx],kind='stable')]
        n=np.arange(len(idx)); offsets=((n+1)//2)*np.where(n%2,1.,-1.)
        out[idx]=offsets*min(step,width/max(1.,np.max(np.abs(offsets))))
    return out

def draw():
    target,products=recover_pair_products()
    cov=pd.read_csv(DATA/'discovery_fig3_covariance.csv')
    response=pd.read_csv(DATA/'discovery_fig3_response.csv')
    paired=pd.read_csv(DATA/'fig3_borrowing_paired.csv')
    paired=paired.loc[paired.block=='chemistry']
    random=pd.read_csv(DATA/'fig3_matched_random.csv')
    oracle=pd.read_csv(DATA/'fig3_oracle_replications.csv')
    fig=plt.figure(figsize=(7.204724409,7.283464567))  # 183 x 185 mm
    left,right,width=.115,.625,.337
    a=fig.add_axes([left,.744,width,.178]); b=fig.add_axes([right,.744,width,.178])
    c=fig.add_axes([left,.438,width,.184]); d=fig.add_axes([right,.438,width,.184])
    e=fig.add_axes([left,.104,width,.214]); f=fig.add_axes([right,.104,width,.214])

    panel(fig,a,'a','Real residual pairs · RxRx3')
    no=target<=0
    a.scatter(target[no],products[no],s=1,color=C['gray'],alpha=.13,edgecolors='none')
    a.scatter(target[~no],products[~no],s=2,color=C['bio'],alpha=.26,edgecolors='none')
    a.axhline(0,color=C['ink'],lw=.6,zorder=0)
    a.set_yscale('symlog',linthresh=.2,linscale=.5)
    a.set_xlim(-.035,1.035); a.set_ylim(-40,100)
    a.set_xticks([0,.5,1],['0','0.5','1'])
    a.set_yticks([-10,-1,0,1,10,100],['−10','−1','0','1','10','100'])
    a.yaxis.set_minor_locator(NullLocator())
    a.set_xlabel('Shared-target similarity')
    a.set_ylabel('Pair residual product\n(symmetric-log scale)')
    a.text(0,1.01,'93,558 dependent pairs; 380 groups',transform=a.transAxes,
           ha='left',va='bottom',fontsize=7.1)

    panel(fig,b,'b','Adjusted target association')
    b.axvline(0,color=C['gray'],lw=.75,ls='--')
    for y,task,marker in [(1,'SAME','o'),(0,'CROSS','s')]:
        row=cov.loc[cov.task==task].iloc[0]
        assert row.model=='ADJUSTED_BOTH'
        b.errorbar(row.raw,y,xerr=[[row.raw-row.low],[row.high-row.raw]],fmt=marker,
                   ms=5,color=C['bio'],capsize=3,elinewidth=1.6)
        b.text(.99,y+.22,f'{numeric_label(row.raw,".3f")} [{numeric_label(row.low,".3f")}, {numeric_label(row.high,".3f")}]',
               ha='right',fontsize=7.5)
    b.set_yticks([1,0],['Same dose','Cross dose'])
    b.set_ylim(-.5,1.7); b.set_xlim(-.12,1.36); b.set_xticks([0,.5,1])
    b.set_xlabel('Target-kernel coefficient\n(response coordinate²)')
    b.text(.97,.91,'Morphology + technical + amplitude controls',
           transform=b.transAxes,ha='right',fontsize=7.1)
    b.spines['left'].set_visible(False); b.tick_params(axis='y',length=0); grid(b,'x')

    panel(fig,c,'c','Response prediction · RxRx3')
    q=pd.read_csv(DATA/'plot_inputs/fig6_response_changes.csv')
    delta=q[['same_dose_mse_change','cross_dose_mse_change']].to_numpy().T
    assert delta.shape==(2,8935)
    assert np.isfinite(delta).all()
    for i,label,color,ls in [(0,'Same dose',C['blue'],'-'),(1,'Cross dose',C['orange'],'--')]:
        ecdf(c,delta[i],color=color,lw=1.45,ls=ls,label=label)
    c.axvline(0,color=C['gray'],lw=.6,zorder=0)
    c.set_xscale('symlog',linthresh=.001,linscale=.5)
    c.set_xlim(-1,1); c.set_ylim(0,100)
    c.xaxis.set_major_locator(FixedLocator([-.1,0,.1]))
    c.xaxis.set_major_formatter(FixedFormatter(['−0.1','0','0.1']))
    c.xaxis.set_minor_locator(NullLocator()); c.set_yticks([0,25,50,75,100])
    c.set_xlabel('MSE change: biology − generic\n(symmetric-log scale; lower is better)')
    c.set_ylabel('Dose pairs at or below (%)')
    c.legend(loc='upper left',bbox_to_anchor=(.01,.86),handlelength=1.5,borderaxespad=0,labelspacing=.25)
    c.text(.02,.98,'All 8,935 dose pairs',transform=c.transAxes,va='top',fontsize=7.5)
    gain=response.loc[response.task=='CROSS'].iloc[0]
    c.text(.98,.06,f'Cross-dose mean benefit\n{gain.estimate:.4f}%\n95% CI:\n{numeric_label(gain.lower,".4f")} to {numeric_label(gain.upper,".4f")}%',
           ha='right',transform=c.transAxes,fontsize=7.2); grid(c)

    panel(fig,d,'d','Object-level borrowing · LINCS')
    counts={}
    borrow=pd.read_csv(DATA/'plot_inputs/fig6_borrowing_changes.csv')
    for family,y,marker in [('TARGET',1,'o'),('MOA',0,'D')]:
        values=borrow.loc[borrow.family==family,'display_change_x1000'].to_numpy()
        row=paired.loc[(paired.family==family)&(paired.alpha==1)].iloc[0]
        assert len(values)==int(row.n) and np.isfinite(values).all()
        np.testing.assert_allclose(values.mean(),row.difference*1000)
        d.scatter(values,y+swarm_offsets(values),s=7,marker=marker,color=C['bio'],alpha=.47,linewidth=0,zorder=2)
        d.errorbar(row.difference*1000,y+.36,
            xerr=[[(row.difference-row.low)*1000],[(row.high-row.difference)*1000]],
            fmt=marker,color=C['ink'],ms=4,elinewidth=1.6,capsize=2.5,zorder=4)
        improved=int((values<0).sum());counts[family]=dict(n=len(values),improved=improved)
        d.text(.98,(y+.10)/2.2+.05,f'{improved}/{len(values)} improve',transform=d.transAxes,fontsize=7.2,ha='right')
    d.axvline(0,color=C['gray'],lw=.75,ls='--')
    d.set_xlim(-22,24); d.set_ylim(-.4,1.8); d.set_xticks([-20,-10,0,10,20])
    d.set_yticks([1,0],['Target\n344 objects','MoA\n259 objects'])
    d.set_xlabel('Δ Γ-CRPS × 1,000\n(biology − CORE; lower is better)')
    d.text(.97,.97,'Full borrowing strength\nPoints: objects; black: mean + 95% CI',
           ha='right',va='top',transform=d.transAxes,fontsize=7.1)
    d.tick_params(axis='y',length=0); d.spines['left'].set_visible(False); grid(d,'x')

    panel(fig,e,'e','Real versus matched random references')
    cols=[(family,alpha) for family in ['TARGET','MOA'] for alpha in [.25,.5,1.]]
    matrix=np.zeros((21,6))
    for j,(family,alpha) in enumerate(cols):
        row=paired.loc[(paired.family==family)&(paired.alpha==alpha)].iloc[0]
        matrix[0,j]=row.difference*1000
        r=random.loc[(random.family==family)&(random.alpha==alpha)].sort_values('replicate')
        assert len(r)==20 and list(r.replicate)==list(range(20))
        matrix[1:,j]=r.difference.to_numpy()*1000
    positions=np.array([0.,1.,2.,3.6,4.6,5.6])
    e.axhline(0,color=C['ink'],lw=.8,ls='--')
    for j,xpos in enumerate(positions):
        values=matrix[1:,j]
        offsets=swarm_offsets(values,step=.075,width=.24,bin_width=.05)
        e.scatter(xpos+offsets,values,s=11,color=C['gray'],alpha=.62,
                  linewidth=0,zorder=2)
        e.plot([xpos-.28,xpos+.28],[values.mean()]*2,color=C['ink'],lw=1.1,zorder=3)
        e.scatter(xpos,matrix[0,j],s=35,marker='D',color=C['bio'],
                  edgecolor='white',linewidth=.65,zorder=4)
    e.axvline(2.8,color=C['light'],lw=.9)
    e.set_xlim(-.55,6.15); e.set_ylim(-.045,1.62)
    e.set_yticks([0,.4,.8,1.2]); e.set_ylabel('Δ Γ-CRPS × 1,000\n(reference borrowing − CORE)')
    e.set_xticks(positions,['.25','.5','1','.25','.5','1'])
    e.tick_params(axis='x',length=0,pad=3)
    e.text(1,1.39,'Target',ha='center',fontsize=8,weight='bold')
    e.text(4.6,1.39,'MoA',ha='center',fontsize=8,weight='bold')
    e.legend(handles=[Line2D([],[],marker='D',color=C['bio'],ls='none',ms=4,label='Biology'),
                      Line2D([],[],marker='o',color=C['gray'],ls='none',ms=3,label='20 random assignments')],
             loc='upper left',bbox_to_anchor=(-.03,1.055),ncol=2,
             fontsize=7,columnspacing=.5,handletextpad=.3,borderaxespad=0)
    e.set_xlabel('Borrowing strength',labelpad=5)
    grid(e)
    fig.text(left+width/2,.031,'All assignments shown; black ticks = random mean',
             ha='center',fontsize=7,color=C['gray'])

    panel(fig,f,'f','Hindsight versus held-out performance')
    sim=oracle.loc[oracle.origin=='CORE self-simulation']
    wide=sim.pivot(index='replicate',columns='realization',values='relative_gain')*100
    first=wide['Fitting realization'].to_numpy(); second=wide['Independent second realization'].to_numpy()
    assert len(wide)==20 and np.isfinite(wide).all().all()
    offsets=np.linspace(-.15,.15,20)
    for j,(v1,v2) in enumerate(zip(first,second)):
        f.plot([v1,v2],[2+offsets[j],1+offsets[j]],color=C['gray'],lw=.5,alpha=.24,zorder=1)
    f.scatter(first,2+offsets,s=11,color=C['blue'],marker='o',zorder=3)
    f.scatter(second,1+offsets,s=11,color=C['orange'],marker='s',zorder=3)
    observed=float(oracle.loc[oracle.origin=='Observed development data','relative_gain'].iloc[0])*100
    f.scatter(observed,3,s=34,color=C['bio'],marker='D',zorder=4)
    f.axvline(0,color=C['ink'],lw=.7,ls='--')
    f.axhline(.5,color=C['light'],lw=.85)
    f.text(observed,3.25,f'{observed:.2f}%',color=C['bio'],ha='center',fontsize=7.4)
    f.text(first.mean(),2.28,f'{first.mean():.2f}%',color=C['blue'],ha='center',fontsize=7.4)
    f.text(second.mean(),1.29,f'{numeric_label(second.mean(),".2f")}%',color=C['orange'],ha='center',fontsize=7.4)
    h=pd.read_csv(DATA/'plot_inputs/fig6_honest_fit.csv').iloc[0]
    benefit=h.relative_benefit_percent
    hlo,hhi=h.relative_low_percent,h.relative_high_percent
    f.errorbar(benefit,0,xerr=[[benefit-hlo],[hhi-benefit]],fmt='^',ms=5,
               color=C['ink'],capsize=2,elinewidth=1.3,zorder=4)
    f.text(3.2,.0,f'{numeric_label(benefit,".3f")}%',ha='left',va='center',fontsize=7.1)
    f.set_xlim(-8,23); f.set_ylim(-.6,3.75); f.set_xticks([-5,0,10,20])
    f.set_yticks([3,2,1,0],['Observed\nhindsight','Simulated\nhindsight','Second\nsimulation','OOF fit\n(real data)'])
    f.set_xlabel('Relative Γ-CRPS benefit (%)',labelpad=5)
    f.tick_params(axis='y',length=0,pad=5,labelsize=7.2)
    f.spines['left'].set_visible(False); grid(f,'x')
    inset=f.inset_axes([.59,.105,.38,.20])
    inset.axvline(0,color=C['gray'],ls='--',lw=.5)
    inset.errorbar(benefit,0,xerr=[[benefit-hlo],[hhi-benefit]],fmt='^',ms=3.5,
                   color=C['ink'],elinewidth=1.1,capsize=2)
    inset.set_xlim(-.15,.06); inset.set_ylim(-.5,.5)
    inset.set_yticks([]); inset.set_xticks([-.1,0],['−0.1','0'])
    inset.tick_params(axis='x',labelsize=7,length=1.5,pad=1)
    inset.set_title('OOF zoom (%)',fontsize=7,pad=1)
    inset.spines['left'].set_visible(False)
    from release_qa import audit_layout
    issues=audit_layout(fig)
    print(json.dumps(dict(layout_issues=issues,object_counts=counts,pair_count=len(products),
                          target_overlap_pairs=int((target>0).sum()),simulation_first_mean=float(first.mean()),
                          simulation_second_mean=float(second.mean())),indent=2))
    if any(level=='FAIL' for level,_ in issues):
        raise RuntimeError('Figure layout audit failed')
    OUT.mkdir(exist_ok=True); name=OUT/'fig3_optional_information'
    fig.savefig(name.with_suffix('.pdf')); fig.savefig(name.with_suffix('.svg'))
    fig.savefig(name.with_suffix('.png'),dpi=600)
    fig.savefig(QA/'fig3_data_rich_preview.png',dpi=300)
    from PIL import Image
    with Image.open(QA/'fig3_data_rich_preview.png') as preview:
        preview.convert('L').save(QA/'fig3_data_rich_grayscale.png')
    plt.close(fig)

if __name__=='__main__':
    draw()
