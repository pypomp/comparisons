#!/usr/bin/env python3
"""Summarize audited saved PyDPF endpoints and plot them; no model execution."""
import argparse, json, math, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS=dict(A='#777777',B='#0072B2',C='#D55E00')

def main(source,audit_path,out):
    assert not out.exists()
    data=json.loads(source.read_text());audit=json.loads(audit_path.read_text())
    assert audit['passed'] and data['all_methods_audit_completed'] and audit['endpoints']==data['audited_endpoints']==36
    assert audit['raw_values']==data['original_PF_repetitions']==1548
    groups={m:sorted([e for e in data['endpoints'] if e['method']==m],key=lambda e:e['case']) for m in 'ABC'}
    summary=dict(methods={},paired={},display_exclusions={},all_endpoints_retained=True,
        PF_calls=0,optimizer_steps=0,timing_unit='entire twelve-member batch',audit=audit)
    for m,rows in groups.items():
        assert [e['case'] for e in rows]==list(range(12))
        vals=np.array([e['logmeanlik'] for e in rows]);q1,q3=np.quantile(vals,[.25,.75],method='linear')
        lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        summary['display_exclusions'][m]=dict(lower=float(lo),upper=float(hi),cases=[e['case'] for e in rows if not lo<=e['logmeanlik']<=hi])
        phase=data['phases'][m];timing=phase['timing']
        summary['methods'][m]=dict(endpoints=12,complete_fits=sum(not e['fit_failed'] for e in rows),
            median_logmeanlik_all=statistics.median(vals),
            nonfloor_precise_scores=sum(e['precision_passed'] and not e['floor_saturated'] for e in rows),
            low_precision_scores=sum(not e['precision_passed'] for e in rows),floor_scores=sum(e['floor_saturated'] for e in rows),
            gradient_kernel_seconds=timing['compute_seconds'],
            logical_fit_kernel_seconds=timing['compute_seconds']+timing['warm_IF2_call_seconds'],
            gradient_wall_seconds=timing['gradient_wall_seconds'],actual_GPU_seconds=phase['actual_GPU_seconds'])
    for left,right in [('C','B'),('B','A'),('C','A')]:
        values=[l['logmeanlik']-r['logmeanlik'] for l,r in zip(groups[left],groups[right])]
        good=[(l['case'],d) for l,r,d in zip(groups[left],groups[right],values)
            if all(e['precision_passed'] and not e['floor_saturated'] and not e['fit_failed'] for e in [l,r])]
        summary['paired'][left+'-'+right]=dict(n=12,positive=sum(v>0 for v in values),median=statistics.median(values),
            minimum=min(values),maximum=max(values),values=values,
            precise_nonfloor_subset=dict(n=len(good),cases=[i for i,d in good],positive=sum(d>0 for i,d in good),
                median=statistics.median(d for i,d in good),minimum=min(d for i,d in good),maximum=max(d for i,d in good)))
    out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),facecolor='white')
        fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.9),layout='constrained')
    for m,rows in groups.items():
        axes[0].plot(range(12),[e['logmeanlik'] for e in rows],marker='o',ms=4,lw=1,color=COLORS[m],label=m)
        for e in rows:
            if e['floor_saturated']:axes[0].scatter(e['case'],e['logmeanlik'],marker='v',s=150,facecolors='none',edgecolors='black',zorder=5)
            if not e['precision_passed']:axes[0].scatter(e['case'],e['logmeanlik'],marker='D',s=100,facecolors='none',edgecolors='black',zorder=5)
    axes[0].set(title='All 36 retained endpoints',xlabel='Case',ylabel='Independent log mean likelihood',xticks=range(0,12,2))
    axes[0].legend(loc='center right',title='Method')
    axes[0].text(.02,.50,'Triangle: B7 density floor\nDiamond: A9 low ESS\nAll fits completed 400 updates',transform=axes[0].transAxes,fontsize=9)
    values=summary['paired']['C-B']['values'];se=[math.hypot(b['MCSE'],c['MCSE']) for b,c in zip(groups['B'],groups['C'])]
    axes[1].errorbar(range(12),values,yerr=se,fmt='o',color=COLORS['C'],capsize=3)
    axes[1].set_yscale('symlog',linthresh=5)
    axes[1].axhline(0,color='grey',lw=1)
    axes[1].set(title='C minus B: all 12 pairs',xlabel='Case',ylabel='LL difference (symmetric log scale)',xticks=range(0,12,2))
    axes[1].annotate('Case 7: B at floor',xy=(7,values[7]),xytext=(1,5000),arrowprops=dict(arrowstyle='->'),fontsize=9)
    keep=summary['paired']['C-B']['precise_nonfloor_subset']['cases']
    axes[2].errorbar(keep,[values[i] for i in keep],yerr=[se[i] for i in keep],fmt='o',color=COLORS['C'],capsize=3)
    axes[2].axhline(0,color='grey',lw=1)
    axes[2].set(title='C minus B: 11 nonfloor pairs',xlabel='Case (7 omitted in this panel only)',ylabel='LL difference',xticks=range(0,12,2),ylim=(-.2,6.5))
    axes[2].text(.02,.98,'Bars: ±1 combined Monte Carlo SE\nNot between-start uncertainty',transform=axes[2].transAxes,va='top',fontsize=8)
    fig.suptitle('PyDPF public extensions: Dacca, T=600, J=5000');save(fig,'endpoints')
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),layout='constrained')
    for i,m in enumerate('ABC'):
        s=summary['methods'][m];g=s['gradient_kernel_seconds'];p=s['logical_fit_kernel_seconds']-g
        axes[0].bar(i,g/3600,color=COLORS[m]);axes[0].bar(i,p/3600,bottom=g/3600,color='#b7d8e8',hatch='//')
    axes[0].set(title='Logical method workload',xticks=range(3),xticklabels=['A','B','C'],ylabel='Fitting-kernel hours, whole B12 batch',ylim=(0,14))
    axes[0].text(.02,.98,'Hatched part: full shared IF2 prefix\nA/B have different total work',transform=axes[0].transAxes,va='top',fontsize=9)
    vals=[181,20216]+[summary['methods'][m]['actual_GPU_seconds'] for m in 'ABC']
    axes[1].bar(range(5),np.array(vals)/3600,color=['#b0b0b0','#b7d8e8']+[COLORS[m] for m in 'ABC'])
    for i,v in enumerate(vals):axes[1].text(i,v/3600+.12,f'{v:,} s',ha='center',fontsize=9)
    axes[1].set(title='Physical resource cost',xticks=range(5),xticklabels=['Old failed\nstartup','IF2','A','B','C'],ylabel='Allocation GPU hours',ylim=(0,9))
    axes[1].text(.02,.98,'Total 86,489 / 108,000 GPU seconds\nValidation 2,522 seconds is separate',transform=axes[1].transAxes,va='top',fontsize=9)
    fig.suptitle('Batch time is counted once; physical IF2 prefix is shared once');save(fig,'timing')
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
    for ax,m in zip(axes,'ABC'):
        hidden=summary['display_exclusions'][m]['cases'];shown=[e for e in groups[m] if e['case'] not in hidden]
        ax.scatter([e['case'] for e in shown],[e['logmeanlik'] for e in shown],color=COLORS[m])
        ax.axhline(summary['methods'][m]['median_logmeanlik_all'],ls='--',color=COLORS[m],lw=1)
        for e in shown:
            if not e['precision_passed']:ax.scatter(e['case'],e['logmeanlik'],marker='D',s=110,facecolors='none',edgecolors='black')
        ax.set(title=f'{m}: {len(hidden)} hidden / 12',xlabel='Case',xticks=range(0,12,2))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0].set_ylabel('Independent log mean likelihood')
    fig.suptitle('Display only: per-method 1.5×IQR, separate vertical scales\nAll numerical summaries and raw records retain every endpoint.',fontsize=11);save(fig,'likelihood_iqr')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.input,args.audit,args.output)
