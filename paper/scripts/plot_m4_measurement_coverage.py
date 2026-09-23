"""Figure 6: saved scalar measurement-observable coverage, not profile coverage."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image
EXPECTED = {'EU':904,'JUMP':639,'LINCS':1188,'RxRx3':10410}
RUNS = EXPECTED
def observable_names():
    return ('single_Z1','single_Z2','single_V','difference_Z1_Z2','difference_Z1_V','difference_Z2_V','average_Z1_Z2','average_Z1_V','average_Z2_V','average_Z1_Z2_V')

from release_paths import ROOT, DATA, OUT, QA, RESEARCH

CORE='#0072B2'; GAUSSIAN='#737D86'; INK='#182838'; ORANGE='#D55E00'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,
    'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'xtick.major.size':3,'ytick.major.size':3,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white'})


def main():
    summary=pd.read_csv(DATA/'m4coverage_summary.csv')
    # Full-width manuscript size in inches (183 x 165 mm).
    fig=plt.figure(figsize=(7.2047244,6.4960630))
    fig.text(.065,.982,'One joint forecast supplies ten observable distributions',
             fontsize=10,weight='bold',va='top')
    fig.legend(handles=[Line2D([],[],color=CORE,marker='o',lw=1.8,ms=3.4,label='CORE'),
        Line2D([],[],color=GAUSSIAN,marker='s',mfc='white',lw=1.3,ls='--',ms=3.2,
               label='Gaussian, same mean and scatter')],
        loc='upper center',bbox_to_anchor=(.53,.953),ncol=2,handlelength=2.7,columnspacing=2)
    pooled=summary[summary.kind=='overall']
    for j,(dataset,n) in enumerate(EXPECTED.items()):
        ax=fig.add_axes([.08+.232*j,.630,.19,.232])
        ax.text(-.22,1.14,'abcd'[j],transform=ax.transAxes,fontsize=10,weight='bold')
        ax.set_title(f'{dataset}\nn = {n:,}',loc='left',fontsize=8.4,pad=6,fontweight='bold')
        for arm,color,marker,ls,lw in [('CORE_ORIGINAL',CORE,'o','-',1.8),
            ('CORE_AMP_GAUSSIAN',GAUSSIAN,'s','--',1.3)]:
            d=pooled[(pooled.dataset==dataset)&(pooled.arm==arm)].sort_values('nominal_coverage')
            ax.plot(range(5),d.coverage_error_pp,color=color,marker=marker,ms=3.5,lw=lw,
                    ls=ls,mfc=color if marker=='o' else 'white',mew=.9)
        ax.axhline(0,color='#9FA8AF',lw=.7,zorder=0)
        ax.set(xlim=(-.15,4.15),ylim=(-10,4),xticks=range(5),
               xticklabels=['50','80','90','95','99'],yticks=[-10,-5,0],
               xlabel='Nominal coverage (%)')
        if j==0:ax.set_ylabel('Observed − nominal\ncoverage (pp)',labelpad=3)
        else:ax.set_yticklabels([])
        ax.grid(axis='y',color='#E1E6E9',lw=.5);ax.set_axisbelow(True)
    fig.text(.065,.547,'e',fontsize=10,weight='bold')
    fig.text(.12,.547,'Which measurements do 95% intervals cover?',fontsize=9,weight='bold')
    names=observable_names()
    cols=['Z1','Z2','V','Z1 − Z2','Z1 − V','Z2 − V',
          'mean(Z1, Z2)','mean(Z1, V)','mean(Z2, V)','mean(Z1, Z2, V)']
    matrix=[]; row_labels=[]; row_records=[]
    for dataset in RUNS:
        for arm,label in [('CORE_ORIGINAL','CORE'),('CORE_AMP_GAUSSIAN','Gaussian')]:
            d=summary[(summary.dataset==dataset)&(summary.arm==arm)&
                (summary.kind=='observable')&(summary.nominal_coverage==.95)].set_index('observable').loc[list(names)]
            matrix.append(100*d.observed_coverage.to_numpy())
            row_labels.append(dataset+'  '+label)
            for name,value in zip(names,matrix[-1]):
                row_records.append(dict(dataset=dataset,arm=arm,observable=name,observed_coverage_percent=value))
    matrix=np.asarray(matrix)
    assert matrix.shape == (8,10) and len(row_records) == 80
    for j,dataset in enumerate(RUNS):
        ax=fig.add_axes([.190+.197*j,.148,.174,.338])
        core,gaussian=matrix[2*j],matrix[2*j+1]
        ax.axvline(95,color='#8A9299',ls=(0,(3,2)),lw=.9,zorder=0)
        for split in [2.5,5.5,8.5]:ax.axhline(split,color='#E1E6E9',lw=.5,zorder=0)
        for row,(blue,grey) in enumerate(zip(core,gaussian)):
            ax.plot([grey,blue],[row+.08,row-.08],color='#AAB1B7',lw=.9,zorder=1)
        ax.plot(gaussian,np.arange(10)+.08,ls='none',marker='s',mfc='white',mec=GAUSSIAN,
                mew=.9,ms=3.6,zorder=2)
        ax.plot(core,np.arange(10)-.08,ls='none',marker='o',color=CORE,ms=3.8,zorder=3)
        ax.set(xlim=(84,99),ylim=(9.6,-.6),xticks=[85,90,95],yticks=range(10),
               yticklabels=cols if j==0 else [])
        ax.tick_params(axis='y',length=0,pad=4,labelsize=7.2)
        ax.tick_params(axis='x',labelsize=7.5,pad=4)
        ax.spines['left'].set_visible(False)
        ax.set_title(dataset,fontsize=8.4,fontweight='bold',pad=6)
    fig.text(.57,.086,'Observed coverage (%) · dashed line: 95% target',fontsize=7.5,ha='center')
    pd.DataFrame(row_records).to_csv(QA/'m4coverage_fig6_95_map.csv',index=False)
    for folder in [OUT,QA]:folder.mkdir(exist_ok=True)
    fig.savefig(OUT/'fig6_measurement_coverage.pdf',dpi=600)
    fig.savefig(OUT/'fig6_measurement_coverage.svg',dpi=600)
    fig.savefig(OUT/'fig6_measurement_coverage.png',dpi=600)
    fig.savefig(QA/'fig6_measurement_coverage_preview.png',dpi=300)
    Image.open(QA/'fig6_measurement_coverage_preview.png').convert('L').save(QA/'fig6_measurement_coverage_grayscale.png')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    tick_gaps=[]
    for a in fig.axes[:4]:
        boxes=[t.get_window_extent(renderer) for t in a.get_xticklabels()]
        tick_gaps.append(min(b.x0-a.x1 for a,b in zip(boxes[:-1],boxes[1:])))
    qa=dict(width_mm=183,height_mm=165,min_tick_gap_pixels_at100dpi=min(tick_gaps),
        objects_per_arm=EXPECTED,arms=['CORE_ORIGINAL','CORE_AMP_GAUSSIAN'],
        exclusions=0,role_coverage_values=int(matrix.size),intervals='none; descriptive empirical coverage',
        unit_definition='deployment cells, with repeated conditions grouped in their source analyses')
    (QA/'fig6_measurement_coverage_layout.json').write_text(json.dumps(qa,indent=2)+'\n')
    plt.close(fig)
    print(json.dumps(qa,indent=2))


if __name__=='__main__':main()
