"""Plot audited original-Dacca results; no fitting, PF scoring or frozen edits.

Run from the repository root with the existing matplotlib dependency path.
All endpoints, including failures and low precision, remain in the source.
"""
from pathlib import Path
import argparse
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
HIDE_IQR_OUTLIERS = False
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


# These descriptions are checked against the frozen drivers, not inferred from
# package capabilities. In particular, PFJAX supplies the forward PF, whereas
# Dynamax supplies the optimization interface used with our Dacca PF.
PACKAGE_USE = {
    'pypomp': ('functional.mif; functional.train; Adam',
               'Paper settings and sequencing of IF2 / gradient phases'),
    'cuthbert': ('particle_filter.build_filter',
                 'IF2 / IFAD and gradient correction added here; pypomp.Adam'),
    'PFJAX': ('particle_filter; resample_multinomial',
              'IF2 / IFAD and gradient correction added here; optax.scale_by_adam'),
    'PyDPF': ('StopGradientDPF; ParticleFilter; SystematicResampler',
              'IF2 / alpha=.97 correction added here; torch.optim.Adam'),
    'TFP': ('particle_filter(unbiased_gradients=True); sequential_monte_carlo',
            'IF2 / alpha=.97 correction added here; tf.keras.optimizers.Adam'),
    'Dynamax': ('SSM.fit_sgd',
                'Dacca PF / IF2 / IFAD added here; optax.scale_by_adam'),
    'particles': ('SMC (via a subclass that records likelihoods)',
                  "Likelihood objective + scipy.optimize.minimize(method='Nelder-Mead')"),
    'pypfilt': ('fit (particle filtering)',
                "Likelihood objective + scipy.optimize.minimize(method='Nelder-Mead')")
}
PANEL_USE = {
    'pypomp': 'IF2: functional.mif\nIFAD: functional.mif then functional.train\nOptimizer: pypomp.Adam',
    'cuthbert': 'PF: particle_filter.build_filter\nAdded here: IF2 / IFAD and gradient correction\nOptimizer: pypomp.Adam',
    'PFJAX': 'PF: particle_filter; IF2: resample_multinomial\nAdded here: IF2 / IFAD and gradient correction\nOptimizer: optax.scale_by_adam',
    'PyDPF': 'Direct alpha=1: StopGradientDPF\nIFAD: ParticleFilter + added alpha=.97 correction\nIF2 added here; optimizer: torch.optim.Adam',
    'TFP': 'PF: particle_filter(unbiased_gradients=True)\nIF2: sequential_monte_carlo + added perturbations\nAdded alpha=.97 correction; tf.keras.optimizers.Adam',
    'Dynamax': 'Optimization interface: SSM.fit_sgd\nAdded here: Dacca PF / IF2 / IFAD\nOptimizer: optax.scale_by_adam',
    'particles': 'PF: SMC, via a subclass that records likelihoods\nAdded likelihood objective; scipy.optimize.minimize\n(method="Nelder-Mead")',
    'pypfilt': 'PF: fit, used to run particle filtering\nAdded likelihood objective; scipy.optimize.minimize\n(method="Nelder-Mead")'
}


def method_label(package, method, detailed=False):
    if method == 'NM':
        return 'SciPy\nNelder-Mead'
    if package in ('pypomp', 'cuthbert'):
        return LABEL[method]
    if method == 'IFAD_1':
        if detailed:
            return {'PFJAX': 'particle_filter\n+ Optax Adam',
                    'PyDPF': 'StopGradientDPF\n+ torch.optim.Adam',
                    'TFP': 'particle_filter\n+ Keras Adam',
                    'Dynamax': 'SSM.fit_sgd\n+ Optax Adam'}[package]
        return 'PF + Adam\n$\\alpha=1$'
    return LABEL[method]


def iqr_selection(rows):
    finite = [r for r in rows if r['logmeanlik'] is not None]
    values = np.asarray([r['logmeanlik'] for r in finite])
    q1, q3 = np.quantile(values, [.25, .75], method='linear')
    lower, upper = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
    shown = [r for r in finite if lower <= r['logmeanlik'] <= upper]
    hidden = [r for r in finite if not lower <= r['logmeanlik'] <= upper]
    return shown, hidden, {'q1': float(q1), 'q3': float(q3), 'lower': float(lower), 'upper': float(upper)}


def display_limits(groups, by, error_bars=False):
    values=[]
    for _,p,m,_ in groups:
        for r in iqr_selection(by[p,m])[0]:
            delta=1.96*r['MCSE'] if error_bars and r['MCSE'] is not None else 0
            values.extend([r['logmeanlik']-delta, r['logmeanlik']+delta])
    width=max(values)-min(values);pad=max(5,width*.05)
    return min(values)-pad,max(values)+pad*2

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
        displayed, hidden, fences = iqr_selection(rr) if HIDE_IQR_OUTLIERS else (finite, [], None)
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
            if HIDE_IQR_OUTLIERS and not fences['lower'] <= r['logmeanlik'] <= fences['upper']:
                continue
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
        count = f'\n{len(displayed)}/{len(finite)}' if HIDE_IQR_OUTLIERS else (f'\n{len(vals)}/{len(rr)}' if main else '')
        names.append(short + count)
        metadata.append({'package': package, 'method': method, 'n': len(rr), 'finite': len(vals),
                         'fit_failures': sum(r['fit_failed'] for r in rr), 'below_zoom': below,
                         'low_precision': sum(not r['precision_passed'] for r in finite),
                         'floor': sum(r['score_floor_saturated'] for r in rr), 'median': median})
        if HIDE_IQR_OUTLIERS:
            metadata[-1].update(displayed=len(displayed),iqr_hidden=len(hidden),iqr_fences=fences)
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
    fig = plt.figure(figsize=(24, 20), facecolor='white')
    fig.text(.055, .968, 'Dacca: parameter estimation', fontsize=25, fontweight='bold')
    subtitle = 'Outliers hidden using 1.5 IQR within each package-method group' if HIDE_IQR_OUTLIERS else 'Comparison of eight software packages'
    fig.text(.055, .942, subtitle, fontsize=15, color=MUTED)
    fig.text(.945, .945, '600 months  |  20 Euler steps/month  |  Gaussian process noise  |  J = 5,000', fontsize=10.5, ha='right', color=MUTED)
    metrics = [('968', 'optimization runs'), ('37,584', 'PF likelihood estimates'), ('9', 'failed fits'), ('17', 'imprecise likelihood estimates'), ('4', 'estimates at -24,868')]
    if HIDE_IQR_OUTLIERS:
        metrics[-1]=(str(sum(len(iqr_selection(rr)[1]) for rr in by.values())), 'outliers hidden')
    for i, (value, caption) in enumerate(metrics):
        x=.055+i*.181
        fig.add_artist(FancyBboxPatch((x,.884),.169,.044,boxstyle='round,pad=.006,rounding_size=.004',transform=fig.transFigure,facecolor='#f0f4f7',edgecolor='none'))
        fig.text(x+.009,.904,value,fontsize=19,fontweight='bold',color=INK)
        fig.text(x+.065,.902,caption,fontsize=9.6,color=MUTED,va='center')

    axa=fig.add_axes([.065,.606,.417,.216]);axb=fig.add_axes([.546,.606,.424,.216])
    panel(axa,'A','PyPOMP and Cuthbert','100 initial values. Medians use all finite estimates; shown/finite counts below.' if HIDE_IQR_OUTLIERS else '100 initial values. Medians above; finite likelihood estimates / runs below.')
    panel(axb,'B','PFJAX, PyDPF, TFP and Dynamax','6 initial values x 2 replicates; shown/finite counts below.' if HIDE_IQR_OUTLIERS else '6 initial values x 2 replicates. Vertical range: -3,835 to -3,735, as in A.')
    paper=[]
    for pi,p in enumerate(['pypomp','cuthbert']):
        for j,m in enumerate(['IF2','IFAD_0','IFAD_097','IFAD_1']):
            paper.append((pi*4.7+j,p,m,method_label(p,m)))
        axa.text(pi*4.7+1.5,1.035,DISPLAY[p],transform=axa.get_xaxis_transform(),ha='center',fontsize=12,fontweight='bold')
    ma=group_plot(axa,paper,by,display_limits(paper,by) if HIDE_IQR_OUTLIERS else (-3835,-3735),zoom=not HIDE_IQR_OUTLIERS,main=True)
    axa.set_ylabel('Estimated log likelihood')
    if not HIDE_IQR_OUTLIERS:axa.set_yticks([-3820,-3800,-3780,-3760,-3740])
    axa.axvline(3.85,color='#d9e1e7',lw=1)
    gpu=[]
    for pi,p in enumerate(['PFJAX','PyDPF','TFP','Dynamax']):
        for j,m in enumerate(['IF2','IFAD_097','IFAD_1']):
            gpu.append((pi*3.65+j,p,m,method_label(p,m)))
        axb.text(pi*3.65+1,1.035,p,transform=axb.get_xaxis_transform(),ha='center',fontsize=11,fontweight='bold')
        if pi<3:axb.axvline(pi*3.65+2.825,color='#d9e1e7',lw=1)
    mb=group_plot(axb,gpu,by,display_limits(gpu,by) if HIDE_IQR_OUTLIERS else (-3835,-3735),zoom=not HIDE_IQR_OUTLIERS,main=True)
    if not HIDE_IQR_OUTLIERS:axb.set_yticks([-3820,-3800,-3780,-3760,-3740])
    axb.tick_params(axis='x',labelsize=8)
    fig.text(.065,.550,'IFAD starts with 60 IF2 iterations (alpha 0 and 1) or 175 (alpha .97). Adam settings differ.',fontsize=9.6,color=MUTED)
    fig.text(.546,.550,'IFAD: 175 IF2 + 400 Adam. PF + Adam: alpha 1, 425 updates, no IF2 initialization. Package calls: F.',fontsize=9.1,color=MUTED)
    legend=[Line2D([],[],marker='o',ls='',mfc=COL['IF2'],mec=COL['IF2'],label='MCSE and ESS criteria met'),
            Line2D([],[],marker='o',ls='',mfc='white',mec=COL['IF2'],label='MCSE or ESS criterion not met'),
            Line2D([],[],marker='v',ls='',color=MUTED,label='Log likelihood < -3,835'),
            Line2D([],[],marker='x',ls='',color=RED,label='Optimization failed'),
            Line2D([],[],marker='D',ls='',color=RED,label='Log likelihood ≈ -24,868')]
    if HIDE_IQR_OUTLIERS:legend=legend[:2]
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
        axd.barh(ys,times,height=.27,color=c,label=DISPLAY[p])
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
        axe.barh(yy,vals,height=.21,color=COL[m],label=['IF2',r'IFAD ($\alpha=0.97$)',r'PF + Adam ($\alpha=1$)'][mi])
        for a,v in zip(yy,vals):axe.text(v*1.04,a,f'{v:.1f}',fontsize=8.5,va='center')
    axe.set_yticks(y,packages);axe.invert_yaxis();axe.set_xscale('log');axe.set_xlim(8,600)
    axe.set_xticks([10,30,60,180,360],['10','30','60','180','360']);axe.minorticks_off()
    axe.set_xlabel('Minutes for 12 runs (log scale)');style(axe,False);axe.grid(axis='x',color='#e6ebef',lw=.8)
    he,le=axe.get_legend_handles_labels()
    fig.legend(he,le,frameon=False,ncol=3,loc='center',bbox_to_anchor=(.818,.274),fontsize=9)
    fig.text(.065,.257,'IFAD times in D and E include IF2 initialization and failed iterations. Dynamax times include compilation within SSM.fit_sgd.',fontsize=10,color=MUTED)

    table_ax=fig.add_axes([.065,.091,.905,.140]);table_ax.axis('off')
    table_ax.text(0,1.09,'F',fontsize=14,fontweight='bold',color=COL['IF2'])
    table_ax.text(.023,1.09,'Package functions, added code and computation time',fontsize=14,fontweight='bold')
    basis={
        'Paper two-package comparison':('PyPOMP / Cuthbert','PyPOMP: functional.mif / train + Adam\nCuthbert: particle_filter.build_filter','Cuthbert: IF2 / IFAD and gradient correction\nadded here; optimizer: pypomp.Adam','GPU h'),
        'PFJAX':('PFJAX','particle_filter; resample_multinomial','IF2 / IFAD and gradient correction added here;\noptax.scale_by_adam','GPU h'),
        'PyDPF':('PyDPF','StopGradientDPF (direct alpha=1);\nParticleFilter + SystematicResampler','IF2 / alpha=.97 correction added here;\ntorch.optim.Adam','GPU h'),
        'TFP':('TFP','particle_filter(unbiased_gradients=True);\nsequential_monte_carlo (IF2)','IF2 / alpha=.97 correction added here;\ntf.keras.optimizers.Adam','GPU h'),
        'Dynamax':('Dynamax','SSM.fit_sgd','Dacca PF / IF2 / IFAD added here;\noptax.scale_by_adam','GPU h'),
        'particles':('particles','SMC (via a subclass recording likelihoods)','Likelihood objective + scipy.optimize.minimize\n(method="Nelder-Mead")','CPU core h'),
        'pypfilt':('pypfilt','fit (particle filtering)','Likelihood objective + scipy.optimize.minimize\n(method="Nelder-Mead")','CPU core h')}
    table=[];quality=[]
    for key,(display,called,added,unit) in basis.items():
        pp=['pypomp','cuthbert'] if key=='Paper two-package comparison' else [key]
        rr=[r for r in data['rows'] if r['package'] in pp]
        finite=[r for r in rr if r['logmeanlik'] is not None]
        f=sum(r['fit_failed'] for r in rr);l=sum(not r['precision_passed'] for r in finite);fl=sum(r['score_floor_saturated'] for r in rr)
        cost=data['cost_by_experiment'][key];h=(cost['formal_allocation_seconds']+cost.get('repair_rescoring_allocation_seconds',0))/3600
        table.append([display,called,added,f'{f}/{len(rr)}',f'{l}/{len(finite)}',str(fl),f'{h:.2f} {unit}'])
        quality.append({'experiment':key,'fit_failure_count':f,'finite_low_precision_count':l,'floor_count':fl,'hours':h,'unit':unit})
    cols=['Package','Code called from package','Added code / external optimizer','Failed\nfits','Low\nprecision','At\n-24,868','Total computing\ntime']
    tab=table_ax.table(cellText=table,colLabels=cols,cellLoc='left',colLoc='left',bbox=[0,0,1,1],colWidths=[.115,.255,.30,.06,.065,.06,.145])
    tab.auto_set_font_size(False);tab.set_fontsize(9.3)
    for (r,c),cell in tab.get_celld().items():
        cell.set_linewidth(.4);cell.set_edgecolor('#dce4e9');cell.PAD=.045
        if r==0:cell.set_facecolor('#e8eff4');cell.set_text_props(weight='bold',color=INK)
        else:cell.set_facecolor('#f7f9fa' if r%2 else 'white')
    notes=[
        'Boxplots and medians use all finite log likelihood estimates. Failed fits use the last valid parameters where available; seven fits produced no finite likelihood estimate.',
        'Triangles in A and B indicate estimates below -3,835. The second figure shows the full range. Each point is one run; replicates in B and C share an initial value.',
        'Total time: 49.12 GPU h and 14.50 CPU core h, including PFJAX likelihood reevaluation. Preliminary tests are excluded. Timing depends on implementation, batching and compilation.',
        'Likelihood evaluation uses the original PyPOMP PF: J=5,000, R=36 for A, up to R=288 for B/C. Evaluation random numbers are independent of optimization; CPU methods share paired numbers.',
        r'Densities below $10^{-18}$ are replaced by $10^{-18}$. Log likelihood -24,868 corresponds to this bound at all 600 observations; MCSE=0 at this value results from the numerical bound.',
        'Added here identifies code supplied for this comparison. PFJAX OT differentiation failed; plotted gradients use the added correction. A package PF alone is not a package parameter-estimation method.'
    ]
    if HIDE_IQR_OUTLIERS:
        notes[0]='Boxplots, medians and summary counts use all original results. Only individual points outside Q1 - 1.5 IQR or Q3 + 1.5 IQR are hidden; IQR = Q3 - Q1.'
        notes[1]='69 of 961 finite estimates are hidden; 892 are shown. Counts below methods give shown/finite estimates. Nine fits failed, including seven with no finite estimate; all failures remain in the counts.'
        notes[4]=r'All four estimates at the numerical bound (about -24,868) are hidden by the same IQR rule. Densities below $10^{-18}$ are replaced by $10^{-18}$; MCSE=0 there does not indicate a good fit.'
    for i,t in enumerate(notes):fig.text(.065,.076-i*.014,t,fontsize=9.4,color=MUTED)
    save(fig,'dacca_overview')
    return {'paper_zoom':ma,'GPU_zoom':mb,'CPU_full_range':mc,'quality_and_cost_table':quality}


def full_range_figure(data,by):
    fig=plt.figure(figsize=(24,15))
    axes=[fig.add_axes([.055+col*.2375,base,.205,.20])
          for base in [.60,.22] for col in range(4)]
    fig.suptitle('Dacca: log likelihood estimates',fontsize=23,fontweight='bold',x=.055,ha='left',y=.975)
    subtitle='Outliers hidden using 1.5 IQR within each package-method group. Vertical scales differ between packages.' if HIDE_IQR_OUTLIERS else 'Full range, with separate vertical scales for each package. Counts include failed optimizations.'
    fig.text(.055,.925,subtitle,fontsize=13,color=MUTED)
    fig.text(.055,.894,'PyPOMP and Cuthbert: 100 initial values per method. Other packages: 6 initial values x 2 replicates. PyPOMP PF likelihood evaluation, J=5,000.',fontsize=11,color=MUTED)
    metadata=[]
    for ax,p in zip(axes,ORDER):
        mm=[m for m in ['IF2','IFAD_0','IFAD_097','IFAD_1','NM'] if (p,m) in by]
        vals=[r['logmeanlik'] for m in mm for r in by[(p,m)] if r['logmeanlik'] is not None]
        width=max(vals)-min(vals);pad=max(15,width*.085)
        bounds=(min(vals)-pad,max(vals)+pad*1.8)
        groups=[(i,p,m,method_label(p,m,detailed=True)) for i,m in enumerate(mm)]
        if HIDE_IQR_OUTLIERS:bounds=display_limits(groups,by,error_bars=True)
        meta=group_plot(ax,groups,by,bounds,zoom=False,median_labels=False)
        rr=[r for m in mm for r in by[(p,m)]]
        finite=[r for r in rr if r['logmeanlik'] is not None]
        failures=sum(r['fit_failed'] for r in rr);low=sum(not r['precision_passed'] for r in finite);floor=sum(r['score_floor_saturated'] for r in rr)
        ax.set_title(DISPLAY.get(p,p),loc='left',fontweight='bold',pad=23)
        counts=f'Finite: {len(finite)}/{len(rr)}  |  failed: {failures}  |  imprecise: {low}  |  at -24,868: {floor}'
        if HIDE_IQR_OUTLIERS:counts=f'Finite: {len(finite)}/{len(rr)}  |  hidden: {sum(m["iqr_hidden"] for m in meta)}  |  failed: {failures}  |  imprecise: {low}'
        ax.text(0,1.025,counts,transform=ax.transAxes,fontsize=8.3,color=MUTED)
        ax.text(0,-.40,PANEL_USE[p],transform=ax.transAxes,fontsize=9.2,
                color=MUTED,va='top',linespacing=1.5)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.set_ylabel('Estimated log likelihood')
        for m in meta:metadata.append(m)
    legend=[Line2D([],[],marker='o',ls='',color=COL['IF2'],label='MCSE and ESS criteria met'),Line2D([],[],marker='o',ls='',mfc='white',mec=COL['IF2'],label='MCSE or ESS criterion not met'),Line2D([],[],marker='x',ls='',color=RED,label='Optimization failed'),Line2D([],[],marker='D',ls='',color=RED,label='Log likelihood ≈ -24,868')]
    if HIDE_IQR_OUTLIERS:legend=legend[:2]
    fig.legend(handles=legend,ncol=4,loc='center',bbox_to_anchor=(.52,.082),frameon=False,fontsize=11)
    full_notes = [
        'Function names refer to the package above each panel unless another library is named. Added here identifies our code. PFJAX OT differentiation failed; the plotted correction was added here.',
        'Purple methods in PFJAX, PyDPF, TFP and Dynamax: alpha 1, 425 Adam updates without IF2 initialization. PyPOMP / Cuthbert IFAD alpha 1 uses 60 IF2 + 425 Adam updates.',
        'Boxplots include all finite log likelihood estimates. Bars show +/- 1.96 MCSE. Failed optimizations are plotted at the last valid parameters where available.',
        r'Densities below $10^{-18}$ are replaced by $10^{-18}$. Log likelihood -24,868 corresponds to this bound at all 600 observations; MCSE=0 at this value results from the numerical bound.'
    ]
    if HIDE_IQR_OUTLIERS:
        full_notes[2]='Boxplots and medians use all finite estimates; only points outside Q1 - 1.5 IQR or Q3 + 1.5 IQR are hidden. Bars show +/- 1.96 MCSE. Counts beneath each method give shown/finite estimates.'
        full_notes[3]='69 of 961 finite estimates are hidden; 892 are shown. Nine fits failed, seven without a finite score; all four estimates near -24,868 are hidden. Full results remain in overview_v1.'
    for i,t in enumerate(full_notes):fig.text(.055,.060-i*.017,t,fontsize=10,color=MUTED)
    save(fig,'dacca_by_package' if HIDE_IQR_OUTLIERS else 'dacca_all_endpoints')
    assert sum(m['n'] for m in metadata)==968
    assert sum(m['finite'] for m in metadata)==961
    assert sum(m['fit_failures'] for m in metadata)==9
    assert sum(m['low_precision'] for m in metadata)==17
    assert sum(m['floor'] for m in metadata)==4
    return metadata


def main():
    global HIDE_IQR_OUTLIERS, OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--hide-iqr-outliers',action='store_true')
    args=parser.parse_args()
    HIDE_IQR_OUTLIERS=args.hide_iqr_outliers
    if HIDE_IQR_OUTLIERS:OUT=ROOT/'outputs/competitor_estimation/overview_iqr_v1'
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
    if HIDE_IQR_OUTLIERS:
        exclusions=[]
        for (package,method),rr in by.items():
            shown,hidden,fences=iqr_selection(rr)
            exclusions.append({'package':package,'plot_method_key':method,'recorded_method':rr[0]['method'],
                'finite_count':len(shown)+len(hidden),'displayed_count':len(shown),'hidden_count':len(hidden),'fences':fences,
                'hidden_records':[{k:r[k] for k in ['start','replicate','logmeanlik','fit_failed','precision_passed','score_floor_saturated']} for r in hidden]})
        hidden_count=sum(x['hidden_count'] for x in exclusions)
        assert hidden_count==69
        details.update(iqr_hidden_count=hidden_count,displayed_finite_count=961-hidden_count,statistics_use_all_original_results=True)
        (OUT/'outliers_hidden.json').write_text(json.dumps({'rule':'Per package x method; hide values outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR]; linear quantiles of all finite estimates. Display only; no statistical or raw-data exclusions.','groups':exclusions,'hidden_count':hidden_count},indent=2)+'\n')
    (OUT/'numeric_audit.json').write_text(json.dumps(details,indent=2)+'\n')
    naming = {
        'definitions': {
            'added here': 'Code supplied for this comparison, distinct from the listed original package functions.',
            'PF + Adam, alpha=1': 'Differentiable particle-filter likelihood optimized with Adam; 425 updates from the original starting values, without IF2 initialization. Actual PF and optimizer calls are listed by package.',
            'scope': 'Calling a package PF does not imply that the complete parameter-estimation procedure is supplied by that package.'
        },
        'display_names': [
            {'package':s['package'], 'recorded_method':s['method'],
             'overview_label':method_label(s['package'],METHODS[s['method']]),
             'by_package_label':method_label(s['package'],METHODS[s['method']],detailed=True)}
            for s in data['summaries']
        ],
        'package_usage': {p:{'called_from_package':v[0],'added_code_and_optimization':v[1],
                              'panel_annotation':PANEL_USE[p]} for p,v in PACKAGE_USE.items()},
        'Dynamax': 'SSM.fit_sgd is supplied by Dynamax; Dacca PF, IF2 and IFAD were added here. Adam uses optax.scale_by_adam.',
        'PFJAX': 'The gradient phases call pfjax.particle_filter with our corrected_resampler; IF2 uses pfjax.resample_multinomial in our loop. Adam uses optax.scale_by_adam. These results are not the native OT method, whose differentiation failed.',
        'unchanged_source_results_sha256':sha(SOURCE/'results.json')
    }
    (OUT/'method_names.json').write_text(json.dumps(naming,indent=2)+'\n')
    shutil.copy2(__file__,OUT/'plot_overview.py')
    (OUT/'README_CN.txt').write_text('原始 Dacca 全包总览\n\n主图 dacca_overview 将100起点论文实验、6起点乘2重复新增GPU实验、CPU1800秒实验分别展示，附拟合耗时、接口归属与实际资源占用。\n\n主图A/B为放大范围；每个范围外有限评分均以三角保留。全范围补图 dacca_all_endpoints 展示961个有限评分，各面板纵轴范围分别设置。7个非有限拟合无评分，仍计入968条记录和9次失败。低精度和底值饱和不作成功拟合解释。\n\n拟合批时间只计一次。Dynamax耗时是12次标量拟合之和且包含原生SGD内部编译。CPU/GPU分开；不能将配置、批处理、原生接口及编译口径不同的耗时当作统一库性能排名。\n\n数据源：../final_comparison/results.json；原交付不修改，不重新拟合或运行PF。数值审计与源SHA见numeric_audit.json。独立评分误差、失败和原生/扩展边界完整记录在原始报告及7份复现包。\n\n复现命令（项目根目录）：\nMPLCONFIGDIR=/private/tmp/dacca-matplotlib-cache PYTHONPATH=work/competitor-estimation/dynamax/plot-deps work/panel-cute-cpu/bin/python work/competitor-estimation/plot_overview.py\n')
    if HIDE_IQR_OUTLIERS:
        (OUT/'README_CN.txt').write_text('Dacca：隐藏箱线图离群点后的图\n\n每个软件包×方法分别用全部有限likelihood估计计算Q1、Q3（线性插值），隐藏小于Q1−1.5×IQR或大于Q3+1.5×IQR的点。共69个点隐藏，892个点显示；完整961个有限评分、968次优化、9次失败均保留。全部4个数值下限点被同一规则隐藏。\n\n箱体、须、中位数、失败/精度计数和计算成本仍使用全部原始结果，不根据绘图筛选重算。隐藏仅改变散点及误差条显示和坐标范围。按分组的阈值与隐藏记录见outliers_hidden.json。原来包含全部点的两张图保持在../overview_v1。\n\ndacca_overview为总览，dacca_by_package按包展示。两图均明确标注隐藏规则与数量，不称为全部结果图。总图F表直接列出包内函数、我们添加的部分及外部优化器；分包图在每个面板下标注实际调用方式。added here表示本次添加的代码。PFJAX调用包内particle_filter，梯度修正由我们添加；Dynamax调用SSM.fit_sgd，Dacca PF由我们实现。原生PF不代表整套参数估计方法来自原包。四个新增GPU包的直接alpha=1方法没有IF2初始化，与论文IFAD1不同。具体对应见method_names.json。\n\n数据：../final_comparison/results.json；未重新拟合或运行PF。\n复现：\nMPLCONFIGDIR=/private/tmp/dacca-matplotlib-cache PYTHONPATH=work/competitor-estimation/dynamax/plot-deps work/panel-cute-cpu/bin/python work/competitor-estimation/plot_overview.py --hide-iqr-outliers\n')
    print(json.dumps({'output_directory':str(OUT),'figures':2,'records':968,'numeric_checks_passed':True,'visual_review_pending':True}))


if __name__=='__main__':main()
