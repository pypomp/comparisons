#!/usr/bin/env python3
"""Summarize all 36 audited endpoints and plot saved scores; no model calls."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = dict(A='#777777', B='#0072B2', C='#D55E00')
LABELS = dict(A='A: Adam, alpha=1', B='B: IF2 + Adam, alpha=1',
              C='C: IF2 + Adam, alpha=0.97')


def main(source, output):
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(source.read_text())
    assert data['passed'] and data['cases'] == 12 and data['endpoints'] == 36
    rows = data['records']
    groups = {a: sorted([r for r in rows if r['method'] == a], key=lambda r: r['case'])
              for a in 'ABC'}
    assert all([r['case'] for r in rr] == list(range(12)) for rr in groups.values())
    summary = {'methods': {}, 'paired': {}, 'display_exclusions': {},
               'all_endpoints_retained': True, 'PF_calls': 0, 'fits_run': 0}
    for arm, rr in groups.items():
        complete = [r for r in rr if not r['fit_failed']]
        values = np.array([r['logmeanlik'] for r in rr])
        q1, q3 = np.quantile(values, [.25, .75], method='linear')
        low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
        summary['display_exclusions'][arm] = dict(lower=float(low), upper=float(high),
            cases=[r['case'] for r in rr if not low <= r['logmeanlik'] <= high])
        summary['methods'][arm] = dict(endpoints=len(rr), complete_fits=len(complete),
            failed_fits=len(rr)-len(complete), precise_scores=sum(r['precision_passed'] for r in rr),
            median_logmeanlik_all=statistics.median(values),
            median_logmeanlik_complete=statistics.median(r['logmeanlik'] for r in complete),
            median_logical_seconds_complete=statistics.median(r['logical_fit_seconds'] for r in complete),
            failed_logical_seconds=[r['logical_fit_seconds'] for r in rr if r['fit_failed']])
    for left, right in [('C','B'), ('B','A'), ('C','A')]:
        delta = [x['logmeanlik']-y['logmeanlik'] for x,y in zip(groups[left],groups[right])]
        summary['paired'][left+'-'+right] = dict(n=12, positive=sum(x>0 for x in delta),
            median=statistics.median(delta), minimum=min(delta), maximum=max(delta), values=delta)
    summary['costs'] = {k:data[k] for k in ['formal_allocation_GPU_seconds',
        'formal_plus_v3_and_repeat_diagnosis_seconds','separate_original_validation_GPU_seconds',
        'all_Cuthbert_checks_and_formal_GPU_seconds']}
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'savefig.dpi':180, 'pdf.fonttype':42})

    def save(fig, name):
        fig.savefig(output/(name+'.png'), facecolor='white')
        fig.savefig(output/(name+'.pdf'), metadata={'CreationDate':None,'ModDate':None})
        plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for arm, rr in groups.items():
        axes[0].plot(range(12),[r['logmeanlik'] for r in rr], color=COLORS[arm],
                     marker='o',ms=4,lw=1,label=LABELS[arm])
        for r in rr:
            if not r['precision_passed']:
                axes[0].scatter(r['case'],r['logmeanlik'],s=110,marker='s',facecolors='none',edgecolors='black',zorder=4)
            if r['fit_failed']:
                axes[0].scatter(r['case'],r['logmeanlik'],s=70,marker='x',color='black',zorder=5)
    axes[0].set(title='All retained endpoints',xlabel='Case (two replicates per start)',
                ylabel='Independent log mean likelihood',xticks=range(12))
    axes[0].legend(fontsize=8,loc='lower left')
    axes[0].text(.02,.65,'Square: low score precision\nCross: fit stopped early',
                 transform=axes[0].transAxes,va='top',fontsize=8)
    delta=summary['paired']['C-B']['values']
    se=[(b['MCSE']**2+c['MCSE']**2)**.5 for b,c in zip(groups['B'],groups['C'])]
    axes[1].errorbar(range(12),delta,yerr=se,fmt='o',color=COLORS['C'],capsize=3)
    axes[1].axhline(0,color='grey',lw=1)
    axes[1].set(title='Paired C minus B',xlabel='Case',ylabel='Difference in log mean likelihood',xticks=range(12))
    axes[1].text(.02,.98,'Bars: ±1 combined Monte Carlo SE\nIndependent endpoint scoring; all 12 pairs',
                transform=axes[1].transAxes,va='top',fontsize=8)
    axes[1].set_ylim(-.5,max(np.array(delta)+se)+1.5)
    fig.suptitle('Cuthbert public extensions: Dacca, T=600, J=5000',fontsize=12)
    save(fig,'endpoints')

    fig, ax=plt.subplots(figsize=(7.5,4.8),layout='constrained')
    for i,arm in enumerate('ABC',1):
        rr=groups[arm]; complete=[r for r in rr if not r['fit_failed']]
        ys=[r['logical_fit_seconds'] for r in complete]
        ax.boxplot([ys],positions=[i],widths=.35,showfliers=False,
                   medianprops={'color':COLORS[arm]},boxprops={'color':COLORS[arm]})
        for r in rr:
            x=i+(r['case']-5.5)*.022
            ax.scatter(x,r['logical_fit_seconds'],marker='x' if r['fit_failed'] else 'o',
                       s=45 if r['fit_failed'] else 22,color='black' if r['fit_failed'] else COLORS[arm])
    ax.set(xticks=[1,2,3],xticklabels=['A\n10 complete, 2 failed','B\n12 complete','C\n12 complete'],
           ylabel='Synchronized fitting kernels (seconds)',title='Method computation; failures shown separately')
    ax.text(.02,.40,'Boxes summarize complete fits only; crosses retain failed-fit cost.\n'
            'B/C each include the full IF2 prefix. Excludes compilation, Python/IO and scoring.',
            transform=ax.transAxes,fontsize=8,va='bottom')
    ax.set_ylim(0,max(r['logical_fit_seconds'] for r in rows)*1.15)
    save(fig,'timing')

    fig, axes=plt.subplots(1,3,figsize=(11,4.6),layout='constrained')
    for ax,arm in zip(axes,'ABC'):
        rr=groups[arm]; excluded=summary['display_exclusions'][arm]['cases']
        shown=[r for r in rr if r['case'] not in excluded]
        ax.scatter([r['case'] for r in shown],[r['logmeanlik'] for r in shown],color=COLORS[arm])
        for r in shown:
            if not r['precision_passed']:
                ax.scatter(r['case'],r['logmeanlik'],s=110,marker='s',facecolors='none',edgecolors='black')
            if r['fit_failed']:
                ax.scatter(r['case'],r['logmeanlik'],s=70,marker='x',color='black')
        ax.set(title=f'{arm}: {len(excluded)} hidden / 12',xlabel='Case',xticks=[0,2,4,6,8,10])
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.axhline(summary['methods'][arm]['median_logmeanlik_all'],color=COLORS[arm],ls='--',lw=1)
    axes[0].set_ylabel('Independent log mean likelihood')
    fig.suptitle('Display only: per-method 1.5×IQR rule; separate vertical scales\n'
                 'Dashed lines and all statistics use all 12 endpoints per method.',fontsize=11)
    save(fig,'likelihood_iqr')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.input,args.output)
