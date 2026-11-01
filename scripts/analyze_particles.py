#!/usr/bin/env python3
"""Plot the 24 audited particles endpoints without filtering or optimization."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'A':'#777777','B':'#0072B2'}

def main(source,output):
    assert not output.exists()
    data=json.loads(source.read_text());assert data['passed'] and data['unique_endpoints']==24
    rows=data['rows'];groups={a:[r for r in rows if r['method']==a] for a in 'AB'}
    assert all([(r['start'],r['replicate']) for r in groups[a]]==[(s,k) for s in range(6) for k in range(2)] for a in 'AB')
    summary={k:data[k] for k in ['unique_endpoints','raw_PF_values_recomputed','fit_failures','precision_passed','low_precision_tasks','per_method','paired','actual_CPU_allocation_seconds','preflight_CPU_seconds','score_GPU_allocation_seconds']}
    summary.update(display_exclusions={},timing={},PF_calls=0,fits_run=0,all_endpoints_retained=True)
    for a,rr in groups.items():
        vals=np.array([r['logmeanlik'] for r in rr]);q1,q3=np.quantile(vals,[.25,.75],method='linear');lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        summary['display_exclusions'][a]=dict(lower=float(lo),upper=float(hi),tasks=[r['task'] for r in rr if not lo<=r['logmeanlik']<=hi])
        summary['timing'][a]=dict(median_charged_fit_seconds=statistics.median(r['charged_fit_seconds'] for r in rr),median_allocation_seconds=statistics.median(r['Slurm_CPU_seconds'] for r in rr))
    good=[r['B_minus_A'] for r in data['paired']['rows'] if r['both_scores_precision_passed']]
    summary['paired']['median_both_precision_passed']=statistics.median(good)
    output.mkdir(parents=True);(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),facecolor='white')
        fig.savefig(output/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for a,rr in groups.items():
        axs[0].plot(range(12),[r['logmeanlik'] for r in rr],marker='o',ms=4,lw=1,color=COLORS[a],label='A: Nelder–Mead' if a=='A' else 'B: IF2 + Nelder–Mead')
        for i,r in enumerate(rr):
            if not r['precision_passed']:axs[0].scatter(i,r['logmeanlik'],marker='s',s=105,facecolors='none',edgecolors='black',zorder=4)
    axs[0].set(title='All 24 retained endpoints',xlabel='Case (two replicates per start)',ylabel='Independent log mean likelihood',xticks=range(12));axs[0].legend(loc='lower right',fontsize=8)
    axs[0].text(.30,.25,'Open square: low scoring ESS at R=288',transform=axs[0].transAxes,fontsize=8)
    delta=[b['logmeanlik']-a['logmeanlik'] for a,b in zip(groups['A'],groups['B'])]
    se=[(a['MCSE']**2+b['MCSE']**2)**.5 for a,b in zip(groups['A'],groups['B'])]
    axs[1].errorbar(range(12),delta,yerr=se,fmt='o',ms=4,color=COLORS['B'],capsize=3)
    for i,(a,b) in enumerate(zip(groups['A'],groups['B'])):
        if not(a['precision_passed'] and b['precision_passed']):axs[1].scatter(i,delta[i],marker='s',s=105,facecolors='none',edgecolors='black')
    axs[1].axhline(0,color='grey',lw=1);axs[1].set(title='Paired B minus A',xlabel='Case',ylabel='Difference in log mean likelihood',xticks=range(12),ylim=(-60,max(delta)*1.19))
    axs[1].text(.02,.98,'Bars: ±1 combined Monte Carlo SE\nThree A estimates remain low precision',transform=axs[1].transAxes,va='top',fontsize=8)
    fig.suptitle('particles public IF2: Dacca, T=600, J=5000',fontsize=12);save(fig,'endpoints')
    fig,axs=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for i,(a,rr) in enumerate(groups.items()):
        x=np.arange(12)+(i-.5)*.10
        axs[0].scatter(x,[r['charged_fit_seconds'] for r in rr],color=COLORS[a],s=24,label=a+' fit')
        axs[0].scatter(x,[r['Slurm_CPU_seconds'] for r in rr],edgecolors=COLORS[a],facecolors='none',marker='s',s=27,label=a+' allocation')
    axs[0].axhline(1800,color='black',ls='--',lw=1);axs[0].set(title='Charged fitting time and complete allocation',xlabel='Case',ylabel='CPU seconds (one core per fit)',xticks=range(12));axs[0].legend(fontsize=8,ncol=2)
    axs[1].bar(range(12),[r['accepted_IF2_passes'] for r in groups['B']],color=COLORS['B'],width=.65)
    axs[1].set(title='B: completed IF2 passes within 900 seconds',xlabel='Case',ylabel='Accepted complete passes',xticks=range(12),ylim=(0,100))
    fig.suptitle('Both methods have a 1,800-second selection deadline\nLate calls are charged but their candidates are not accepted; GPU scoring is separate.',fontsize=11);save(fig,'timing')
    fig,axs=plt.subplots(1,2,figsize=(9,4.6),layout='constrained')
    for ax,(a,rr) in zip(axs,groups.items()):
        excluded=summary['display_exclusions'][a]['tasks'];shown=[r for r in rr if r['task'] not in excluded]
        ax.scatter([r['task']//2 for r in shown],[r['logmeanlik'] for r in shown],color=COLORS[a])
        for r in shown:
            if not r['precision_passed']:ax.scatter(r['task']//2,r['logmeanlik'],marker='s',s=105,facecolors='none',edgecolors='black')
        ax.axhline(data['per_method'][a]['median_all_scores_descriptive'],color=COLORS[a],ls='--',lw=1)
        ax.set(title=f'{a}: {len(excluded)} hidden / 12',xlabel='Case',xticks=range(12));ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axs[0].set_ylabel('Independent log mean likelihood');fig.suptitle('Display only: per-method 1.5×IQR rule; separate vertical scales\nDashed lines and all statistics include every endpoint.',fontsize=11);save(fig,'likelihood_iqr')
    print(json.dumps({k:summary[k] for k in ['unique_endpoints','precision_passed','display_exclusions','timing']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.input,a.output)
