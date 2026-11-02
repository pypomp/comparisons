#!/usr/bin/env python3
"""Draw saved native A/B and IFAD results. No fitting or PF evaluation.

Native PFJAX input is an explicitly selected, audited table; existing C scores
must match the original table. Partial PFJAX samples receive points only.
Use --data to reproduce a dated figure from its saved input snapshot.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import plot_dacca_distributions as base

PACKAGES = ('cuthbert', 'pfjax', 'tfp', 'pydpf')
LABELS = {
    'cuthbert': ('A: Stop-gradient PF', 'B: IF2 + stop-gradient PF', 'C: IFAD (α = 0.97)'),
    'pfjax': ('A: Basic-PF autodiff', 'B: IF2 + basic-PF autodiff', 'C: IFAD (α = 0.97)'),
    'tfp': ('A: Differentiable PF', 'B: IF2 + differentiable PF', 'C: IFAD (α = 0.97)'),
    'pydpf': ('A: StopGradientDPF', 'B: IF2 + StopGradientDPF', 'C: IFAD (α = 0.97)'),
}


def read_inputs(root, native):
    root, native = root.resolve(), native.resolve()
    old, sources = base.read_data(root)
    sources[str(native.relative_to(root))] = base.digest(native)
    with native.open(newline='') as f:
        rows = list(csv.DictReader(f))
    cases = sorted({int(r['case']) for r in rows})
    assert cases and len(rows) == 3 * len(cases)
    assert len({(r['method'], int(r['case'])) for r in rows}) == len(rows)
    new = []
    for r in rows:
        case = int(r['case'])
        assert r['method'] in 'ABC' and 0 <= case < 12
        assert int(r['start_id']) * 2 + int(r['replicate']) == case
        d = dict(package='pfjax', method=r['method'], case=case,
                 start=int(r['start_id']), replicate=int(r['replicate']))
        for k in ('logmeanlik', 'MCSE', 'likelihood_ESS'):
            d[k] = float(r[k])
        d['repetitions'] = int(r['repetitions'])
        for k in ('fit_failed', 'precision_passed', 'floor_saturated'):
            d[k] = base.boolean(r[k])
        if r['method'] == 'C':
            prior = next(x for x in old if x['package'] == 'pfjax' and
                         x['method'] == 'C' and x['case'] == case)
            assert d == prior, ('Existing C changed', case)
        new.append(d)
    for arm in 'ABC':
        assert sorted(x['case'] for x in new if x['method'] == arm) == cases
    selected = [r for r in old if r['package'] in PACKAGES and r['package'] != 'pfjax'] + new
    return dict(records=selected, original_records=old, source_sha256=sources,
                native_PFJAX_cases=cases, native_PFJAX_complete=len(cases) == 12,
                display=dict(zoom=[-3800, -3740], KDE_bandwidth=2,
                             all_values_in_statistics=True, partial_PFJAX_points_only=True),
                PF_calls=0, optimizer_updates=0)


def rows(data, pkg, arm):
    return sorted((r for r in data['records'] if r['package'] == pkg and r['method'] == arm),
                  key=lambda r: r['case'])


def setup(ax, limits):
    base.decorate(ax, limits)
    ax.tick_params(labelsize=10)
    ax.set_ylim(.45, 3.50)


def legend(fig, y):
    handles = [Line2D([], [], marker='o', color='#555555', ls='', ms=4, label='Final estimate'),
               Line2D([], [], marker='D', color='#222222', mfc='white', ls='', ms=5, label='Low scoring precision'),
               Line2D([], [], marker='v', color='#222222', mfc='white', ls='', ms=5, label='Density floor'),
               Line2D([], [], marker='x', color='#222222', ls='', ms=5, label='Fit stopped early')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, y),
               frameon=False, ncol=4, fontsize=9, handletextpad=.5, columnspacing=1.6)


def subtitle(data, pkg):
    if pkg == 'pfjax':
        n = len(data['native_PFJAX_cases'])
        return f'Biased native gradient · {n}/12 cases' + (' · preliminary' if n < 12 else '')
    return '12 runs per method · six starts, two repetitions'


def title(ax, data, pkg, letter):
    ax.text(0, 1.14, f'{letter}   {base.NAMES[pkg]}', transform=ax.transAxes,
            fontsize=15, weight='bold', ha='left')
    ax.text(0, 1.045, subtitle(data, pkg), transform=ax.transAxes, fontsize=9,
            color='#555555', ha='left')


def raincloud(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.1))
    fig.subplots_adjust(left=.185, right=.98, bottom=.18, top=.85, wspace=.78, hspace=.62)
    fig.text(.025, .962, 'Dacca: final log likelihood', fontsize=21, weight='bold')
    for ax, pkg, letter in zip(axes.flat, PACKAGES, 'abcd'):
        setup(ax, (-3800, -3740))
        ax.set_xticks([-3800, -3780, -3760, -3740])
        ax.set_yticks([3, 2, 1], LABELS[pkg], fontsize=10)
        for tick, arm in zip(ax.get_yticklabels(), 'ABC'):
            tick.set_color(base.COLORS[arm])
        title(ax, data, pkg, letter)
        counts = []
        for arm, y in zip('ABC', [3, 2, 1]):
            rr = rows(data, pkg, arm)
            x = np.array([r['logmeanlik'] for r in rr])
            if len(rr) == 12:
                grid = np.linspace(-3800, -3740, 800)
                density = np.exp(-.5*((grid[:, None]-x)/2)**2).mean(axis=1)/(2*np.sqrt(2*np.pi))
                ax.fill_between(grid, y+.04, y+.04+2.1*density,
                                color=base.COLORS[arm], alpha=.36, linewidth=0)
                base.draw_box(ax, rr, y-.07, arm, width=.12)
            base.draw_points(ax, rr, y-.25, arm)
            counts.append(f'{arm}: {sum(x < -3800)}/{len(rr)}')
        ax.text(0, -.18, 'Below −3800:  ' + '   '.join(counts), transform=ax.transAxes,
                fontsize=8.7, color='#555555')
    fig.text(.57, .097, 'Independent log likelihood (higher is better)', ha='center', fontsize=12)
    legend(fig, .034)
    fig.text(.54, .012, 'External Adam in all arms. Boxes and densities use all runs; the view is zoomed.',
             fontsize=9, ha='center', color='#555555')
    base.save(fig, out, 'likelihood_distributions')


def full_range(data, out):
    fig = plt.figure(figsize=(13.2, 8.1))
    outer = fig.add_gridspec(2, 2, left=.185, right=.98, bottom=.18, top=.85, wspace=.78, hspace=.62)
    fig.text(.025, .962, 'Dacca: all final estimates', fontsize=21, weight='bold')
    for cell, pkg, letter in zip(outer, PACKAGES, 'abcd'):
        gs = cell.subgridspec(1, 2, width_ratios=[1, 5], wspace=.1)
        a, b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        for ax, lim in ((a, (-24920, -24815)), (b, (-4900, -3730))):
            setup(ax, lim)
            for arm, y in zip('ABC', [3, 2, 1]):
                rr = rows(data, pkg, arm)
                assert all(-24920 <= r['logmeanlik'] <= -24815 or -4900 <= r['logmeanlik'] <= -3730 for r in rr)
                if len(rr) == 12:
                    base.draw_box(ax, rr, y, arm)
                base.draw_points(ax, rr, y-.22, arm)
            for yy in (0,):
                x = 1 if ax is a else 0
                ax.plot([x-.025, x+.025], [yy-.02, yy+.02], transform=ax.transAxes,
                        color='#444444', clip_on=False, lw=.9)
        a.set_xticks([-24868], ['−24,868']); b.set_xticks([-4800, -4400, -4000, -3740])
        a.set_yticks([3, 2, 1], LABELS[pkg], fontsize=10); b.set_yticks([])
        for tick, arm in zip(a.get_yticklabels(), 'ABC'):
            tick.set_color(base.COLORS[arm])
        a.text(0, 1.14, f'{letter}   {base.NAMES[pkg]}', transform=a.transAxes, fontsize=15, weight='bold')
        a.text(0, 1.045, subtitle(data, pkg), transform=a.transAxes, fontsize=9, color='#555555')
    fig.text(.57, .097, 'Independent log likelihood (broken axis; all estimates shown)', ha='center', fontsize=12)
    legend(fig, .034)
    foot = 'All 12 runs per method are retained, including density-floor scores and stopped fits.'
    if not data['native_PFJAX_complete']:
        foot = 'Density-floor scores remain in every summary. Partial PFJAX results are points only.'
    fig.text(.54, .012, foot,
             fontsize=9, ha='center', color='#555555')
    base.save(fig, out, 'all_estimates')


def paired_rows(data):
    result = []
    for pkg in PACKAGES:
        b, c = rows(data, pkg, 'B'), rows(data, pkg, 'C')
        assert [r['case'] for r in b] == [r['case'] for r in c]
        for br, cr in zip(b, c):
            result.append(dict(package=pkg, case=br['case'], start=br['start'], replicate=br['replicate'],
                               difference=cr['logmeanlik']-br['logmeanlik'],
                               MCSE=(cr['MCSE']**2+br['MCSE']**2)**.5,
                               floor_pair=br['floor_saturated'] or cr['floor_saturated'],
                               low_precision=not(br['precision_passed'] and cr['precision_passed']),
                               failed_fit=br['fit_failed'] or cr['fit_failed']))
    return result


def paired(data, out):
    pairs = paired_rows(data)
    near = [r for r in pairs if abs(r['difference']) < 1000]
    far = [r for r in pairs if abs(r['difference']) >= 1000]
    assert near and far, 'Choose an unbroken layout if there are no distant differences'
    central = (min(-1, np.floor(min(r['difference']-r['MCSE'] for r in near))),
               max(10, np.ceil(max(r['difference']+r['MCSE'] for r in near)/5)*5))
    distant = (min(r['difference'] for r in far)-60, max(r['difference'] for r in far)+60)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.7, 5.1), gridspec_kw=dict(width_ratios=[5, 1.4]))
    fig.subplots_adjust(left=.15, right=.98, bottom=.24, top=.76, wspace=.08)
    fig.text(.025, .94, 'Paired log likelihood differences: C − B', fontsize=19, weight='bold')
    fig.text(.025, .872, 'Same IF2 start and Adam settings; each point is one paired case', fontsize=11, color='#555555')
    for ax, lim in ((a, central), (b, distant)):
        ax.set_xlim(*lim); ax.set_ylim(-.6, 3.6)
        ax.grid(axis='x', color='#e6e6e6'); ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#aaaaaa'); ax.tick_params(axis='y', length=0)
        for pkg, y in zip(PACKAGES, [3, 2, 1, 0]):
            rr = [p for p in pairs if p['package'] == pkg]
            jitter = np.linspace(-.15, .15, len(rr)) if len(rr) > 1 else [0]
            for r, off in zip(rr, jitter):
                assert central[0] <= r['difference'] <= central[1] or distant[0] <= r['difference'] <= distant[1]
                marker = 'v' if r['floor_pair'] else 'D' if r['low_precision'] else 'o'
                ax.errorbar(r['difference'], y+off, xerr=None if r['floor_pair'] else r['MCSE'],
                            fmt=marker, color=base.COLORS['C'], mfc='white' if marker != 'o' else base.COLORS['C'],
                            ms=5, elinewidth=.8, alpha=.9)
                if r['failed_fit']:
                    ax.scatter(r['difference'], y+off, marker='x', color='black')
            if len(rr) == 12:
                med = np.median([r['difference'] for r in rr])
                ax.plot([med, med], [y-.25, y+.25], color='#222222', lw=1.5)
        x = 1 if ax is a else 0
        ax.plot([x-.014, x+.014], [-.02, .02], transform=ax.transAxes, color='#444444', clip_on=False)
    a.axvline(0, color='#888888', ls=':', lw=1)
    a.set_xticks(np.arange(0, central[1]+.01, 5)); b.set_xticks([round(distant[0]+20, -1), round(distant[1]-20, -1)]); b.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    a.set_yticks([3, 2, 1, 0], ['Cuthbert (12 pairs)', f"PFJAX ({len(data['native_PFJAX_cases'])} of 12)", 'TFP (12 pairs)', 'PyDPF (12 pairs)'], fontsize=10)
    b.set_yticks([])
    floor = next(p for p in far if p['floor_pair'])
    b.text(.5, .99, f"B at density floor\nΔ = {floor['difference']:,.2f}", transform=b.transAxes, ha='center', va='top', fontsize=9)
    fig.text(.55, .13, 'Difference in independent log likelihood (C − B)', ha='center', fontsize=12)
    fig.text(.55, .067, 'Bars: ±1 estimated Monte Carlo SE. Black ticks: medians over all 12 pairs.', ha='center', fontsize=9, color='#555555')
    foot = 'PFJAX changes gradient construction as well as discounting.'
    if not data['native_PFJAX_complete']:
        foot += ' Its partial sample has no median tick.'
    fig.text(.55, .024, foot, ha='center', fontsize=9, color='#555555')
    base.save(fig, out, 'paired_C_minus_B')
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--native-table', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert bool(a.native_table) != bool(a.data), 'Provide --native-table or --data'
    data = json.loads(a.data.read_text()) if a.data else read_inputs(a.root, a.native_table)
    a.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'axes.unicode_minus': True})
    raincloud(data, a.output); full_range(data, a.output)
    pairs = paired(data, a.output)
    data['group_statistics'] = []
    for pkg in PACKAGES:
        for arm in 'ABC':
            rr = rows(data, pkg, arm)
            data['group_statistics'].append(dict(package=pkg, method=arm, n=len(rr),
                **base.box_stats(rr), floor=sum(r['floor_saturated'] for r in rr),
                failed=sum(r['fit_failed'] for r in rr), low_precision=sum(not r['precision_passed'] for r in rr),
                precise_nonfloor=sum(r['precision_passed'] and not r['floor_saturated'] for r in rr)))
    for name, table in (('endpoints.csv', data['records']), ('paired_differences.csv', pairs)):
        with (a.output/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    data['pairs'] = pairs
    (a.output/'figure_data.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(dict(endpoints=len(data['records']), native_PFJAX_cases=data['native_PFJAX_cases'],
                          PF_calls=0, optimizer_updates=0, output=str(a.output))))


if __name__ == '__main__':
    main()
