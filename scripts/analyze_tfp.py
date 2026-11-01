#!/usr/bin/env python3
"""Figures and summaries from the completed saved-data TFP audit; no PF calls."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=dict(A='#777777',B='#0072B2',C='#D55E00')
def main(source,out):
 assert not out.exists();a=json.loads(source.read_text())
 assert a['passed'] and a['actual_GPU_all_methods_audit_completed'] and a['endpoint_count']==36
 groups={m:sorted([e for e in a['endpoints'] if e['method']==m],key=lambda e:e['case_id']) for m in 'ABC'}
 s={'methods':{},'paired':{},'display_exclusions':{},'all_endpoints_retained':True,'PF_calls':0,'optimizer_steps':0,'timing_unit':'entire twelve-member batch'}
 for m,rs in groups.items():
  assert [e['case_id'] for e in rs]==list(range(12));vals=np.array([e['logmeanlik'] for e in rs]);complete=[e for e in rs if not e['fit_failed']]
  q1,q3=np.quantile(vals,[.25,.75],method='linear');lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
  s['display_exclusions'][m]={'lower':float(lo),'upper':float(hi),'cases':[e['case_id'] for e in rs if not lo<=e['logmeanlik']<=hi]}
  s['methods'][m]={'endpoints':12,'complete_fits':len(complete),'failed_fits':12-len(complete),'nonfloor_precise_scores':sum(e['precision_passed'] and not e['floor_saturated'] for e in rs),'floor_scores':sum(e['floor_saturated'] for e in rs),'median_logmeanlik_all':statistics.median(vals),'median_logmeanlik_complete':statistics.median(e['logmeanlik'] for e in complete),**a['phases'][m]}
 for l,r in [('C','B'),('B','A'),('C','A')]:
  ds=[x['logmeanlik']-y['logmeanlik'] for x,y in zip(groups[l],groups[r])]
  good=[x['logmeanlik']-y['logmeanlik'] for x,y in zip(groups[l],groups[r]) if all(not e['fit_failed'] and e['precision_passed'] and not e['floor_saturated'] for e in (x,y))]
  s['paired'][l+'-'+r]={'n':12,'positive':sum(d>0 for d in ds),'median':statistics.median(ds),'minimum':min(ds),'maximum':max(ds),'values':ds,'complete_precise_nonfloor_subset':{'n':len(good),'positive':sum(d>0 for d in good),'median':statistics.median(good) if good else None}}
 s['costs']={k:a[k] for k in ['actual_formal_GPU_seconds','physical_prefix_GPU_seconds','formal_budget_seconds','prior_validation_GPU_seconds_separate']};s['costs']['formal_budget_remaining_seconds']=a['formal_budget_seconds']-a['actual_formal_GPU_seconds']
 out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
 def save(fig,name):
  fig.savefig(out/(name+'.png'),facecolor='white');fig.savefig(out/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for m,rs in groups.items():
  axes[0].plot(range(12),[e['logmeanlik'] for e in rs],marker='o',ms=4,lw=1,color=C[m],label=m)
  for e in rs:
   if e['fit_failed']:axes[0].scatter(e['case_id'],e['logmeanlik'],marker='x',s=95,color='black',zorder=6)
   if e['floor_saturated']:axes[0].scatter(e['case_id'],e['logmeanlik'],marker='v',s=180,facecolors='none',edgecolors='black',zorder=5)
 axes[0].set(title='All 36 retained endpoints',xlabel='Case (two repetitions per start)',ylabel='Independent log mean likelihood',xticks=range(12));axes[0].legend(loc='center right',title='Method')
 axes[0].text(.02,.42,'A case 6: 64 accepted updates, then failure\nTriangle: density floor; cross: failed fit',transform=axes[0].transAxes,fontsize=8)
 ds=s['paired']['C-B']['values'];se=[math.hypot(b['MCSE'],c['MCSE']) for b,c in zip(groups['B'],groups['C'])]
 axes[1].errorbar(range(12),ds,yerr=se,fmt='o',color=C['C'],capsize=3);axes[1].axhline(0,color='grey',lw=1)
 axes[1].set(title='Paired C minus B',xlabel='Case',ylabel='Difference in log mean likelihood',xticks=range(12),ylim=(-.5,max(d+e for d,e in zip(ds,se))+2))
 axes[1].text(.02,.98,'Bars: ±1 combined Monte Carlo SE\nNot between-start uncertainty',transform=axes[1].transAxes,va='top',fontsize=8)
 fig.suptitle('TFP public extensions: Dacca, T=600, J=5000');save(fig,'endpoints')
 fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');prefix=12561.816784949042
 for i,m in enumerate('ABC'):
  g=a['phases'][m]['gradient_kernel_seconds'];p=0 if m=='A' else prefix
  assert math.isclose(g+p,a['phases'][m]['logical_fit_kernel_seconds'],abs_tol=1e-8)
  axes[0].bar(i,g/3600,color=C[m]);axes[0].bar(i,p/3600,bottom=g/3600,color='#b7d8e8',hatch='//')
 axes[0].set(xticks=range(3),xticklabels=['A\n11 complete, 1 failed','B\n12 complete','C\n12 complete'],ylabel='Logical fitting-kernel hours, B12 batch',title='Method workload')
 axes[0].text(.02,.98,'Hatched part: full shared IF2 prefix\nA/B have different total work',transform=axes[0].transAxes,va='top',fontsize=8);axes[0].set_ylim(0,8)
 vals=[a['physical_prefix_GPU_seconds']]+[a['phases'][m]['actual_GPU_seconds'] for m in 'ABC']
 axes[1].bar(range(4),np.array(vals)/3600,color=['#b7d8e8']+[C[m] for m in 'ABC'])
 for i,v in enumerate(vals):axes[1].text(i,v/3600+.07,f'{v:,} s',ha='center',fontsize=8)
 axes[1].set(xticks=range(4),xticklabels=['IF2','A','B','C'],ylabel='Actual allocation GPU hours',title='Physical resource cost; prefix counted once',ylim=(0,4.5))
 axes[1].text(.02,.98,f"Total {a['actual_formal_GPU_seconds']:,} / 72,000 GPU seconds\nValidation 2,336 seconds is separate",transform=axes[1].transAxes,va='top',fontsize=8)
 fig.suptitle('Times belong to each whole batch; do not multiply by twelve');save(fig,'timing')
 fig,axes=plt.subplots(1,3,figsize=(11,4.6),layout='constrained')
 for ax,m in zip(axes,'ABC'):
  hidden=s['display_exclusions'][m]['cases'];shown=[e for e in groups[m] if e['case_id'] not in hidden]
  ax.scatter([e['case_id'] for e in shown],[e['logmeanlik'] for e in shown],color=C[m]);ax.axhline(s['methods'][m]['median_logmeanlik_all'],ls='--',color=C[m],lw=1)
  ax.set(title=f'{m}: {len(hidden)} hidden / 12',xlabel='Case',xticks=[0,2,4,6,8,10]);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 axes[0].set_ylabel('Independent log mean likelihood');fig.suptitle('Display only: per-method 1.5×IQR, separate vertical scales\nAll raw data, medians and paired statistics retain every endpoint.',fontsize=11);save(fig,'likelihood_iqr')
 print(json.dumps(s,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.input,a.output)
