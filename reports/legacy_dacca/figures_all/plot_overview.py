"""Plot audited original-Dacca results; no fitting, PF scoring or frozen edits.

Run from the repository root with the existing matplotlib dependency path.
All endpoints, including failures and low precision, remain in the source.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/competitor_estimation/final_comparison'
OUT = ROOT / 'outputs/competitor_estimation/overview_v1'
COL = {'IF2': '#3776a8', 'IFAD_0': '#329a89', 'IFAD_097': '#d67b24', 'IFAD_1': '#7664a8', 'NM': '#70827f'}
INK = '#1c2d3d'
MUTED = '#5d6d7c'
RED = '#b7423b'
LABEL = {'IF2': 'IF2', 'IFAD_0': 'IFAD\n$\\alpha=0$', 'IFAD_097': 'IFAD\n$\\alpha=0.97$', 'IFAD_1': 'IFAD\n$\\alpha=1$', 'NM': 'Nelder-Mead'}
METHODS = {'IF2': 'IF2', 'IFAD_0': 'IFAD_0', 'IFAD_097': 'IFAD_097', 'IFAD': 'IFAD_097',
           'IFAD_1': 'IFAD_1', 'ADPF': 'IFAD_1', 'ADPF_extension': 'IFAD_1',
           'SGD_alpha1': 'IFAD_1', 'PF + external Nelder-Mead': 'NM'}
# IFAD_1 is a shared plotting/color key only. Display names distinguish paper
# IFAD (60 IF2 + 425 Adam) from direct ADPF (425 Adam, no IF2 initialization).
ORDER = ['pypomp', 'cuthbert', 'PFJAX', 'PyDPF', 'TFP', 'Dynamax', 'particles', 'pypfilt']
DISPLAY = {'pypomp': 'PyPOMP', 'cuthbert': 'Cuthbert'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def method_label(package, method):
    if method == 'NM':
        return 'Package PF\n+ SciPy Nelder-Mead'
    if package in ('pypomp', 'cuthbert'):
        origin = 'package' if package == 'pypomp' else 'custom'
        return LABEL[method] + f'\n({origin})'
    if method == 'IFAD_1':
        origin = 'package PF' if package in ('PyDPF', 'TFP') else 'custom PF'
        return 'ADPF\n+ Adam\n(' + origin + ')'
    return LABEL[method] + '\n(custom)'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'axes.titlesize': 14,
                     'axes.labelsize': 11, 'xtick.labelsize': 9.5, 'ytick.labelsize': 10,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b1bbc3', 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'savefig.facecolor': 'white'})


def load():
    manifest = json.loads((SOURCE/'index_manifest.json').read_text())
    assert manifest['release_ready'] and manifest['independent_integration_review_passed']
    for name, digest in manifest['files'].items():
        assert sha(SOURCE/name) == digest, name
    review = json.loads((SOURCE/'independent_review.json').read_text())
    data = json.loads((SOURCE/'results.json').read_text())
    assert review['passed'] and sha(SOURCE/'results.json') == review['results_sha256']
    rows = data['rows']
    assert len(rows) == 968
    assert len({(r['experiment_scope'], r['package'], r['method'], r['start'], r['replicate']) for r in rows}) == 968
    assert sum(r['raw_repetitions'] for r in rows) == 37584
    assert sum(r['fit_failed'] for r in rows) == 9
    assert sum(r['score_floor_saturated'] for r in rows) == 4
    assert sum(r['logmeanlik'] is not None and not r['precision_passed'] for r in rows) == 17
    by = collections.defaultdict(list)
    for r in rows:
        by[(r['package'], METHODS[r['method']])].append(r)
    for s in data['summaries']:
        rr = by[(s['package'], METHODS[s['method']])]
        values = [r['logmeanlik'] for r in rr if r['logmeanlik'] is not None]
        assert len(rr) == s['n'] and len(values) == s['finite_n']
        assert math.isclose(statistics.median(values), s['median_all_finite'], abs_tol=1e-8)
    return data, by, manifest


def style(ax, ygrid=True):
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis='y', color='#e6ebef', lw=.8)
    ax.tick_params(length=0, pad=5)


def panel(ax, letter, title, subtitle, y=1.18):
    ax.text(0, y, letter, transform=ax.transAxes, fontsize=14, fontweight='bold', color=COL['IF2'])
    ax.text(.04, y, title, transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.text(0, y-.065, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED)


def group_plot(ax, groups, by, limits, zoom=False, main=False, median_labels=True):
    low, high = limits
    span = high-low
    ax.set_ylim(low, high)
    ticks, names, metadata = [], [], []
    for x, package, method, short in groups:
        rr = sorted(by[(package, method)], key=lambda r: (r['start'], r['replicate']))
        finite = [r for r in rr if r['logmeanlik'] is not None]
        vals = [r['logmeanlik'] for r in finite]
        color = COL[method]
        # Box statistics always include every finite endpoint, even outside zoom.
        box = ax.boxplot([vals], positions=[x], widths=.47, manage_ticks=False,
                         showfliers=False, patch_artist=True, whis=1.5,
                         boxprops={'facecolor': color+'22', 'edgecolor': color+'88', 'linewidth': 1},
                         medianprops={'color': color, 'linewidth': 2.2},
                         whiskerprops={'color': color+'99', 'linewidth': 1},
                         capprops={'color': color+'99', 'linewidth': 1})
        below = 0
        for i, r in enumerate(finite):
            jitter = (i*.61803398875 % 1 - .5)*.34
            val = r['logmeanlik']; plotted = val
            marker = 'o'
            isbelow = zoom and val < low
            if isbelow:
                below += 1
                plotted = low+span*.018
                marker = 'v'
            failed = r['fit_failed']
            floor = r['score_floor_saturated']
            ec = RED if (failed or floor) else color
            fc = ec if r['precision_passed'] else 'white'
            if not zoom and floor:
                marker = 'D'
            size = 13 if len(rr) == 100 else 27
            if floor or failed:
                size += 12
            if not main and not isbelow and r['MCSE'] is not None:
                ax.errorbar(x+jitter, plotted, yerr=1.96*r['MCSE'], color=ec, lw=.6, alpha=.5, zorder=2)
            ax.scatter(x+jitter, plotted, s=size, marker=marker, edgecolors=ec,
                       facecolors=fc, linewidths=.7, alpha=.78 if len(rr)==100 else .95, zorder=3)
            if failed:
                ax.scatter(x+jitter, plotted, marker='x', s=size+5, c=RED, lw=1.2, zorder=4)
        median = statistics.median(vals)
        if median_labels:
            ax.text(x, high-span*.018, f'{median:.2f}', ha='center', va='top', fontsize=8.5,
                    color=color, fontweight='bold', bbox={'facecolor': 'white', 'alpha': .92, 'edgecolor': 'none', 'pad': .6})
        if below:
            ax.text(x, low+span*.067, f'{below} below', ha='center', fontsize=8, color=RED,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.5})
        ticks.append(x)
        names.append(short + (f'\n{len(vals)}/{len(rr)}' if main else ''))
        metadata.append({'package': package, 'method': method, 'n': len(rr), 'finite': len(vals),
                         'fit_failures': sum(r['fit_failed'] for r in rr), 'below_zoom': below,
                         'low_precision': sum(not r['precision_passed'] for r in finite),
                         'floor': sum(r['score_floor_saturated'] for r in rr), 'median': median})
    ax.set_xticks(ticks, names)
    ax.set_xlim(min(ticks)-.6, max(ticks)+.6)
    style(ax)
    return metadata


def save(fig, stem):
    fig.savefig(OUT/f'{stem}.png', dpi=200)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    # Convenient in-chat preview; full PNG and vector files retain original size.
    fig.savefig(OUT/f'{stem}_preview.png', dpi=105)
    plt.close(fig)


def main_figure(data, by):
    fig = plt.figure(figsize=(24, 18), facecolor='white')
    fig.text(.055, .968, 'Dacca: parameter estimation', fontsize=25, fontweight='bold')
    fig.text(.055, .942, 'Comparison of eight software packages', fontsize=15, color=MUTED)
    fig.text(.945, .945, '600 months  |  20 Euler steps/month  |  Gaussian process noise  |  J = 5,000', fontsize=10.5, ha='right', color=MUTED)
    metrics = [('968', 'optimization runs'), ('37,584', 'PF likelihood estimates'), ('9', 'failed fits'), ('17', 'imprecise likelihood estimates'), ('4', 'estimates at -24,868')]
    for i, (value, caption) in enumerate(metrics):
        x=.055+i*.181
        fig.add_artist(FancyBboxPatch((x,.884),.169,.044,boxstyle='round,pad=.006,rounding_size=.004',transform=fig.transFigure,facecolor='#f0f4f7',edgecolor='none'))
        fig.text(x+.009,.904,value,fontsize=19,fontweight='bold',color=INK)
        fig.text(x+.065,.902,caption,fontsize=9.6,color=MUTED,va='center')

    axa=fig.add_axes([.065,.606,.417,.216]);axb=fig.add_axes([.546,.606,.424,.216])
    panel(axa,'A','PyPOMP and Cuthbert','100 initial values. Medians above; finite likelihood estimates / runs below.')
    panel(axb,'B','PFJAX, PyDPF, TFP and Dynamax','6 initial values x 2 replicates. Vertical range: -3,835 to -3,735, as in A.')
    paper=[]
    for pi,p in enumerate(['pypomp','cuthbert']):
        for j,m in enumerate(['IF2','IFAD_0','IFAD_097','IFAD_1']):
            paper.append((pi*4.7+j,p,m,method_label(p,m)))
        axa.text(pi*4.7+1.5,1.035,DISPLAY[p],transform=axa.get_xaxis_transform(),ha='center',fontsize=12,fontweight='bold')
    ma=group_plot(axa,paper,by,(-3835,-3735),zoom=True,main=True)
    axa.set_ylabel('Estimated log likelihood')
    axa.set_yticks([-3820,-3800,-3780,-3760,-3740])
    axa.axvline(3.85,color='#d9e1e7',lw=1)
    gpu=[]
    for pi,p in enumerate(['PFJAX','PyDPF','TFP','Dynamax']):
        for j,m in enumerate(['IF2','IFAD_097','IFAD_1']):
            gpu.append((pi*3.65+j,p,m,method_label(p,m)))
        axb.text(pi*3.65+1,1.035,p,transform=axb.get_xaxis_transform(),ha='center',fontsize=11,fontweight='bold')
        if pi<3:axb.axvline(pi*3.65+2.825,color='#d9e1e7',lw=1)
    mb=group_plot(axb,gpu,by,(-3835,-3735),zoom=True,main=True)
    axb.set_yticks([-3820,-3800,-3780,-3760,-3740])
    axb.tick_params(axis='x',labelsize=8)
    fig.text(.065,.550,'IFAD starts with 60 IF2 iterations (alpha 0 and 1) or 175 (alpha .97). Adam settings differ.',fontsize=9.6,color=MUTED)
    fig.text(.546,.550,'IFAD: 175 IF2 + 400 Adam iterations. ADPF: alpha 1, 425 Adam iterations, no IF2 initialization.',fontsize=9.1,color=MUTED)
    legend=[Line2D([],[],marker='o',ls='',mfc=COL['IF2'],mec=COL['IF2'],label='MCSE and ESS criteria met'),
            Line2D([],[],marker='o',ls='',mfc='white',mec=COL['IF2'],label='MCSE or ESS criterion not met'),
            Line2D([],[],marker='v',ls='',color=MUTED,label='Log likelihood < -3,835'),
            Line2D([],[],marker='x',ls='',color=RED,label='Optimization failed'),
            Line2D([],[],marker='D',ls='',color=RED,label='Log likelihood ≈ -24,868')]
    fig.legend(handles=legend,loc='center',bbox_to_anchor=(.52,.523),ncol=5,frameon=False,fontsize=10,handletextpad=.5,columnspacing=2)

    axc=fig.add_axes([.065,.316,.19,.153]);axd=fig.add_axes([.335,.316,.24,.153]);axe=fig.add_axes([.665,.316,.305,.153])
    panel(axc,'C','CPU: Nelder-Mead','12 runs per package; 1,800 s optimization per run.',y=1.20)
    mc=group_plot(axc,[(0,'particles','NM','particles'),(1,'pypfilt','NM','pypfilt')],by,(-5950,-3900),main=False)
    axc.set_ylabel('Estimated log likelihood')
    axc.set_yticks([-5800,-5400,-5000,-4600,-4200])
    axc.text(.5,-.20,'Package PF with SciPy Nelder-Mead',transform=axc.transAxes,ha='center',fontsize=9.5,color=MUTED)
    panel(axd,'D','PyPOMP and Cuthbert: fitting time','100 runs per batch; IFAD includes IF2 initialization.',y=1.20)
    methods=['IF2','IFAD_0','IFAD_097','IFAD_1'];yt=np.arange(4)
    for i,(p,c) in enumerate([('pypomp','#416e91'),('cuthbert','#9cabb8')]):
        times=[data['cost_by_experiment']['Paper two-package comparison']['by_method'][p+'/'+m]['fitting_seconds']/60 for m in methods]
        ys=yt+(i-.5)*.30
        origin = 'package IF2/IFAD' if p == 'pypomp' else 'custom IF2/IFAD'
        axd.barh(ys,times,height=.27,color=c,label=f'{DISPLAY[p]}: {origin}')
        for y,v in zip(ys,times):axd.text(v+3,y,f'{v:.1f}',va='center',fontsize=8.7)
    axd.set_yticks(yt,['IF2',r'IFAD ($\alpha=0$)',r'IFAD ($\alpha=0.97$)',r'IFAD ($\alpha=1$)']);axd.invert_yaxis();axd.set_xlim(0,260)
    axd.set_xlabel('Minutes per batch of 100');style(axd,False);axd.grid(axis='x',color='#e6ebef',lw=.8)
    hd,ld=axd.get_legend_handles_labels()
    fig.legend(hd,ld,frameon=False,ncol=2,loc='center',bbox_to_anchor=(.455,.274),fontsize=9.5)
    panel(axe,'E','GPU fitting time','One batch of 12 in PFJAX, PyDPF and TFP; 12 separate runs in Dynamax.',y=1.20)
    packages=['PFJAX','PyDPF','TFP','Dynamax'];y=np.arange(4)
    for mi,m in enumerate(['IF2','IFAD_097','IFAD_1']):
        vals=[]
        for p in packages:
            methods_cost=data['cost_by_experiment'][p]['by_method']
            actual=next(k for k in methods_cost if METHODS[k]==m)
            vals.append(methods_cost[actual]['fitting_seconds']/60)
        yy=y+(mi-1)*.23
        axe.barh(yy,vals,height=.21,color=COL[m],label=['IF2 (custom)',r'IFAD ($\alpha=0.97$, custom)','ADPF + Adam'][mi])
        for a,v in zip(yy,vals):axe.text(v*1.04,a,f'{v:.1f}',fontsize=8.5,va='center')
    axe.set_yticks(y,packages);axe.invert_yaxis();axe.set_xscale('log');axe.set_xlim(8,600)
    axe.set_xticks([10,30,60,180,360],['10','30','60','180','360']);axe.minorticks_off()
    axe.set_xlabel('Minutes for 12 runs (log scale)');style(axe,False);axe.grid(axis='x',color='#e6ebef',lw=.8)
    he,le=axe.get_legend_handles_labels()
    fig.legend(he,le,frameon=False,ncol=3,loc='center',bbox_to_anchor=(.818,.274),fontsize=9)
    fig.text(.065,.257,'IFAD times in D and E include IF2 initialization and failed iterations. Dynamax times include compilation within SSM.fit_sgd.',fontsize=10,color=MUTED)

    table_ax=fig.add_axes([.065,.108,.905,.123]);table_ax.axis('off')
    table_ax.text(0,1.09,'F',fontsize=14,fontweight='bold',color=COL['IF2'])
    table_ax.text(.023,1.09,'Implementation and computation time',fontsize=14,fontweight='bold')
    basis={
        'Paper two-package comparison':('PyPOMP / Cuthbert','Package IF2/IFAD in PyPOMP; custom IF2/IFAD on Cuthbert PF','GPU h'),
        'PFJAX':('PFJAX','Custom IF2, IFAD and ADPF; package OT differentiation failed','GPU h'),
        'PyDPF':('PyDPF','StopGradientDPF + torch.optim.Adam; custom IF2/IFAD','GPU h'),
        'TFP':('TFP','particle_filter (unbiased_gradients=True) + Adam; custom IF2/IFAD','GPU h'),
        'Dynamax':('Dynamax','Package SSM.fit_sgd optimizer; custom Dacca PF, IF2 and IFAD','GPU h'),
        'particles':('particles','Package PF + SciPy Nelder-Mead','CPU core h'),
        'pypfilt':('pypfilt','Package PF + SciPy Nelder-Mead','CPU core h')}
    table=[];quality=[]
    for key,(display,b,unit) in basis.items():
        pp=['pypomp','cuthbert'] if key=='Paper two-package comparison' else [key]
        rr=[r for r in data['rows'] if r['package'] in pp]
        finite=[r for r in rr if r['logmeanlik'] is not None]
        f=sum(r['fit_failed'] for r in rr);l=sum(not r['precision_passed'] for r in finite);fl=sum(r['score_floor_saturated'] for r in rr)
        cost=data['cost_by_experiment'][key];h=(cost['formal_allocation_seconds']+cost.get('repair_rescoring_allocation_seconds',0))/3600
        table.append([display,b,f'{f}/{len(rr)}',f'{l}/{len(finite)}',str(fl),f'{h:.2f} {unit}'])
        quality.append({'experiment':key,'fit_failure_count':f,'finite_low_precision_count':l,'floor_count':fl,'hours':h,'unit':unit})
    cols=['Package','Implementation','Failed fits','Low precision','At -24,868','Total computing time']
    tab=table_ax.table(cellText=table,colLabels=cols,cellLoc='left',colLoc='left',bbox=[0,0,1,1],colWidths=[.14,.43,.10,.10,.06,.17])
    tab.auto_set_font_size(False);tab.set_fontsize(9.5)
    for (r,c),cell in tab.get_celld().items():
        cell.set_linewidth(.4);cell.set_edgecolor('#dce4e9');cell.PAD=.065
        if r==0:cell.set_facecolor('#e8eff4');cell.set_text_props(weight='bold',color=INK)
        else:cell.set_facecolor('#f7f9fa' if r%2 else 'white')
    notes=[
        'Boxplots and medians use all finite log likelihood estimates. Failed fits use the last valid parameters where available; seven fits produced no finite likelihood estimate.',
        'Triangles in A and B indicate estimates below -3,835. The second figure shows the full range. Each point is one run; replicates in B and C share an initial value.',
        'Total time: 49.12 GPU h and 14.50 CPU core h, including PFJAX likelihood reevaluation. Preliminary tests are excluded. Timing depends on implementation, batching and compilation.',
        'Likelihood evaluation uses the original PyPOMP PF: J=5,000, R=36 for A, up to R=288 for B/C. Evaluation random numbers are independent of optimization; CPU methods share paired numbers.',
        r'Densities below $10^{-18}$ are replaced by $10^{-18}$. Log likelihood -24,868 corresponds to this bound at all 600 observations; MCSE=0 at this value results from the numerical bound.',
        'Package denotes code supplied by the library; custom denotes code added for this study. ADPF uses differentiable PF likelihoods with Adam. Package PF does not imply a package parameter-estimation method.'
    ]
    for i,t in enumerate(notes):fig.text(.065,.085-i*.014,t,fontsize=9.4,color=MUTED)
    save(fig,'dacca_overview')
    return {'paper_zoom':ma,'GPU_zoom':mb,'CPU_full_range':mc,'quality_and_cost_table':quality}


def full_range_figure(data,by):
    fig,axes=plt.subplots(2,4,figsize=(24,12))
    fig.subplots_adjust(left=.055,right=.98,bottom=.185,top=.84,hspace=.60,wspace=.29)
    fig.suptitle('Dacca: log likelihood estimates',fontsize=23,fontweight='bold',x=.055,ha='left',y=.975)
    fig.text(.055,.925,'Full range, with separate vertical scales for each package. Counts include failed optimizations.',fontsize=13,color=MUTED)
    fig.text(.055,.894,'PyPOMP and Cuthbert: 100 initial values per method. Other packages: 6 initial values x 2 replicates. PyPOMP PF likelihood evaluation, J=5,000.',fontsize=11,color=MUTED)
    metadata=[]
    for ax,p in zip(axes.flat,ORDER):
        mm=[m for m in ['IF2','IFAD_0','IFAD_097','IFAD_1','NM'] if (p,m) in by]
        vals=[r['logmeanlik'] for m in mm for r in by[(p,m)] if r['logmeanlik'] is not None]
        width=max(vals)-min(vals);pad=max(15,width*.085)
        bounds=(min(vals)-pad,max(vals)+pad*1.8)
        groups=[(i,p,m,method_label(p,m)) for i,m in enumerate(mm)]
        meta=group_plot(ax,groups,by,bounds,zoom=False,median_labels=False)
        rr=[r for m in mm for r in by[(p,m)]]
        finite=[r for r in rr if r['logmeanlik'] is not None]
        failures=sum(r['fit_failed'] for r in rr);low=sum(not r['precision_passed'] for r in finite);floor=sum(r['score_floor_saturated'] for r in rr)
        ax.set_title(DISPLAY.get(p,p),loc='left',fontweight='bold',pad=23)
        ax.text(0,1.025,f'Finite: {len(finite)}/{len(rr)}  |  failed: {failures}  |  imprecise: {low}  |  at -24,868: {floor}',transform=ax.transAxes,fontsize=8.3,color=MUTED)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.set_ylabel('Estimated log likelihood')
        for m in meta:metadata.append(m)
    legend=[Line2D([],[],marker='o',ls='',color=COL['IF2'],label='MCSE and ESS criteria met'),Line2D([],[],marker='o',ls='',mfc='white',mec=COL['IF2'],label='MCSE or ESS criterion not met'),Line2D([],[],marker='x',ls='',color=RED,label='Optimization failed'),Line2D([],[],marker='D',ls='',color=RED,label='Log likelihood ≈ -24,868')]
    fig.legend(handles=legend,ncol=4,loc='center',bbox_to_anchor=(.52,.115),frameon=False,fontsize=11)
    full_notes = [
        'Package denotes code supplied by the library; custom denotes code added for this study. Dynamax uses package SSM.fit_sgd with a custom PF. Package PF alone is not a package parameter-estimation method.',
        'ADPF + Adam: differentiable PF likelihood, alpha 1, 425 updates without IF2 initialization. IFAD starts with IF2; it is a different optimization procedure.',
        'Boxplots include all finite log likelihood estimates. Bars show +/- 1.96 MCSE. Failed optimizations are plotted at the last valid parameters where available.',
        r'Densities below $10^{-18}$ are replaced by $10^{-18}$. Log likelihood -24,868 corresponds to this bound at all 600 observations; MCSE=0 at this value results from the numerical bound.'
    ]
    for i,t in enumerate(full_notes):fig.text(.055,.078-i*.021,t,fontsize=10,color=MUTED)
    save(fig,'dacca_all_endpoints')
    assert sum(m['n'] for m in metadata)==968
    assert sum(m['finite'] for m in metadata)==961
    assert sum(m['fit_failures'] for m in metadata)==9
    assert sum(m['low_precision'] for m in metadata)==17
    assert sum(m['floor'] for m in metadata)==4
    return metadata


def main():
    data,by,manifest=load()
    OUT.mkdir(parents=True,exist_ok=True)
    details=main_figure(data,by)
    details['all_endpoints']=full_range_figure(data,by)
    details.update({'source_results_sha256':sha(SOURCE/'results.json'),'source_manifest_sha256':sha(SOURCE/'index_manifest.json'),
                    'source_independent_review_sha256':sha(SOURCE/'independent_review.json'),
                    'endpoint_records':968,'finite_scores':961,'fit_failures':9,'finite_low_precision':17,'floor_scores':4,
                    'actual_final_PF_repetitions':37584,'fit_calls':0,'PF_calls':0,'numeric_checks_passed':True,
                    'visual_review_passed':False,'visual_review_note':'Pending inspection of generated PNG and rendered PDF.'})
    for name,digest in manifest['files'].items():assert sha(SOURCE/name)==digest,name
    details['original_delivery_unchanged']=True
    (OUT/'numeric_audit.json').write_text(json.dumps(details,indent=2)+'\n')
    naming = {
        'definitions': {
            'package': 'Code supplied by the named library.',
            'custom': 'Code added for this study.',
            'ADPF + Adam': 'Differentiable particle-filter likelihood optimized with Adam; alpha=1, 425 updates from the original starting values, without IF2 initialization.',
            'package PF': 'The particle filter is supplied by the library; this label does not claim that the complete parameter-estimation procedure is supplied by the library.'
        },
        'display_names': [
            {'package':s['package'], 'recorded_method':s['method'],
             'figure_label':method_label(s['package'],METHODS[s['method']])}
            for s in data['summaries']
        ],
        'Dynamax': 'SSM.fit_sgd is supplied by Dynamax; the Dacca PF, IF2 and IFAD are custom implementations.',
        'PFJAX': 'The plotted IF2, IFAD and ADPF results are custom implementations. Package OT differentiation failed before formal fitting.',
        'unchanged_source_results_sha256':sha(SOURCE/'results.json')
    }
    (OUT/'method_names.json').write_text(json.dumps(naming,indent=2)+'\n')
    shutil.copy2(__file__,OUT/'plot_overview.py')
    (OUT/'README_CN.txt').write_text('原始 Dacca 全包总览\n\n主图 dacca_overview 将100起点论文实验、6起点乘2重复新增GPU实验、CPU1800秒实验分别展示，附拟合耗时、接口归属与实际资源占用。\n\n主图A/B为放大范围；每个范围外有限评分均以三角保留。全范围补图 dacca_all_endpoints 展示961个有限评分，各面板纵轴范围分别设置。7个非有限拟合无评分，仍计入968条记录和9次失败。低精度和底值饱和不作成功拟合解释。\n\n拟合批时间只计一次。Dynamax耗时是12次标量拟合之和且包含原生SGD内部编译。CPU/GPU分开；不能将配置、批处理、原生接口及编译口径不同的耗时当作统一库性能排名。\n\n数据源：../final_comparison/results.json；原交付不修改，不重新拟合或运行PF。数值审计与源SHA见numeric_audit.json。独立评分误差、失败和原生/扩展边界完整记录在原始报告及7份复现包。\n\n复现命令（项目根目录）：\nMPLCONFIGDIR=/private/tmp/dacca-matplotlib-cache PYTHONPATH=work/competitor-estimation/dynamax/plot-deps work/panel-cute-cpu/bin/python work/competitor-estimation/plot_overview.py\n')
    with (OUT/'README_CN.txt').open('a') as f:
        f.write('\n方法名称：package 表示软件包提供的代码，custom 表示本次添加的代码。IF2、IFAD分别标注实现来源。直接alpha=1方法称为ADPF + Adam，并注明package PF或custom PF；它从原始起点运行425步Adam，没有IF2初始化，不能称为论文IFAD1。\nPyDPF和TFP使用包提供的可微PF；PFJAX的正式三方法均为本地扩展；Dynamax仅优化接口SSM.fit_sgd来自原包，Dacca PF由本地实现。particles和pypfilt为Package PF + SciPy Nelder-Mead。具体记录名称到图示名称的对应关系见method_names.json。\n')
    print(json.dumps({'output_directory':str(OUT),'figures':2,'records':968,'numeric_checks_passed':True,'visual_review_pending':True}))


if __name__=='__main__':main()
