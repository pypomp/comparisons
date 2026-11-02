#!/usr/bin/env python3
"""Summarize the selected Dacca results from saved records; never fit or score.

The current selection uses native PFJAX A/B and the existing C. Earlier PFJAX
alpha=1 A/B remain a separate five-method comparison. All finite scores enter
medians, including floors and low-precision scores. Timing units stay separate.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from repository import saved_bytes

ROOT = Path(__file__).resolve().parents[1]
FIGURES = 'reports/dacca_native_figures_20261102T0718'


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def saved_summary(root=ROOT):
    inputs = {}

    def read(name, kind='json'):
        data = (root / name).read_bytes() if (root / name).is_file() else saved_bytes(name)
        inputs[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data) if kind == 'json' else list(csv.DictReader(io.StringIO(data.decode())))

    figure = read(FIGURES + '/figure_data.json')
    audit = read('results/pfjax_native/final_audit_20261102.json')
    assert audit['passed'] and audit['complete_twelve_case_review']
    assert figure['native_PFJAX_complete'] and figure['native_PFJAX_cases'] == list(range(12))
    for name, digest in figure['source_sha256'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == digest, name
        inputs[name] = actual
    records = figure['records'] + [r for r in figure['original_records'] if r['package'] == 'particles']
    assert len(records) == 168
    assert {r['package'] for r in records} == {'cuthbert', 'pfjax', 'tfp', 'pydpf', 'particles'}
    assert len({(r['package'], r['method'], r['start'], r['replicate']) for r in records}) == 168
    summaries, pairs = [], []
    for package in ('cuthbert', 'pfjax', 'tfp', 'pydpf', 'particles'):
        methods = 'AB' if package == 'particles' else 'ABC'
        groups = {}
        for method in methods:
            rows = [r for r in records if r['package'] == package and r['method'] == method]
            assert len(rows) == 12
            assert {(r['start'], r['replicate']) for r in rows} == {(s, r) for s in range(6) for r in range(2)}
            groups[method] = {(r['start'], r['replicate']): r for r in rows}
            finite = [r['logmeanlik'] for r in rows if math.isfinite(r['logmeanlik'])]
            summaries.append(dict(package=package, method=method, endpoints=len(rows),
                median_logmeanlik=statistics.median(finite), finite_scores=len(finite),
                fit_failed=sum(r['fit_failed'] for r in rows),
                floor_scores=sum(r['floor_saturated'] for r in rows),
                low_precision_scores=sum(not r['precision_passed'] for r in rows),
                precise_nonfloor_scores=sum(r['precision_passed'] and not r['floor_saturated'] for r in rows),
                raw_repetitions=sum(r['repetitions'] for r in rows)))
        left, right = ('B', 'A') if package == 'particles' else ('C', 'B')
        all_pairs = [(groups[left][k], groups[right][k]) for k in sorted(groups[left])]
        for selection in ('all', 'precise_nonfloor'):
            rr = all_pairs if selection == 'all' else [p for p in all_pairs if all(r['precision_passed'] and not r['floor_saturated'] for r in p)]
            differences = [a['logmeanlik'] - b['logmeanlik'] for a, b in rr]
            pairs.append(dict(package=package, contrast=left+'-'+right, selection=selection,
                pairs=len(rr), positive=sum(d > 0 for d in differences),
                median_difference=statistics.median(differences),
                minimum_difference=min(differences), maximum_difference=max(differences)))

    five = read('results/pfjax_native/native_and_extension_20261102.csv', 'csv')
    reference = read(FIGURES + '/summary.json')
    five_summary = []
    for method in ('Native A', 'Added alpha=1 A', 'Native B', 'Added alpha=1 B', 'C: IFAD'):
        rows = [r for r in five if r['method'] == method]
        assert len(rows) == 12 and {int(r['case']) for r in rows} == set(range(12))
        complete = [r for r in rows if int(r['accepted_updates']) == 400 and r['fit_failed'] == 'False']
        item = dict(method=method, endpoints=12,
            median_logmeanlik=statistics.median(float(r['logmeanlik']) for r in rows),
            complete_fits=len(complete),
            failed_fits=sum(r['fit_failed'] == 'True' for r in rows),
            floor_scores=sum(r['floor_saturated'] == 'True' for r in rows),
            low_precision_scores=sum(r['precision_passed'] != 'True' for r in rows),
            median_complete_gradient_seconds=statistics.median(float(r['gradient_seconds']) for r in complete),
            median_complete_logical_fit_seconds=statistics.median(float(r['logical_fit_kernel_seconds']) for r in complete),
            timing_unit='one complete fit; median', timing_denominator=len(complete))
        ref = reference['methods'][method]
        assert item['median_logmeanlik'] == ref['median_logmeanlik']
        assert item['median_complete_gradient_seconds'] == ref['median_400_update_gradient_seconds']
        assert item['median_complete_logical_fit_seconds'] == ref['median_complete_logical_fit_kernel_seconds']
        five_summary.append(item)

    timing = []
    cuthbert = read('results/cuthbert/endpoints.csv', 'csv')
    for method in 'ABC':
        complete = [r for r in cuthbert if r['method'] == method and r['iterations_completed'] == '400' and r['fit_failed'] == 'False']
        timing.append(dict(package='cuthbert', method=method, unit='one complete fit; median',
            complete_fits=len(complete),
            gradient_kernel_seconds=statistics.median(float(r['gradient_seconds']) for r in complete),
            logical_fit_kernel_seconds=statistics.median(float(r['logical_fit_seconds']) for r in complete)))
    for method, label in zip('ABC', ('Native A', 'Native B', 'C: IFAD')):
        r = next(r for r in five_summary if r['method'] == label)
        timing.append(dict(package='pfjax', method=method, unit=r['timing_unit'],
            complete_fits=r['complete_fits'], gradient_kernel_seconds=r['median_complete_gradient_seconds'],
            logical_fit_kernel_seconds=r['median_complete_logical_fit_seconds']))
    for package, date in (('tfp', '20261101'), ('pydpf', '20261102')):
        report = read('reports/'+package+'_'+date+'/summary.json')
        for method, r in report['methods'].items():
            timing.append(dict(package=package, method=method, unit='whole batch of 12; one measurement',
                complete_fits=r['complete_fits'], gradient_kernel_seconds=r['gradient_kernel_seconds'],
                logical_fit_kernel_seconds=r['logical_fit_kernel_seconds']))

    overview = dict(selection='Native gradient baselines for four GPU packages; CPU particles shown separately',
        distinct_starts=6, repetitions_per_start=2, endpoints=len(records),
        raw_repetitions=sum(r['repetitions'] for r in records),
        failed_fits=sum(r['fit_failed'] for r in records),
        floor_scores=sum(r['floor_saturated'] for r in records),
        low_precision_scores=sum(not r['precision_passed'] for r in records),
        precise_nonfloor_scores=sum(r['precision_passed'] and not r['floor_saturated'] for r in records),
        source_sha256=dict(sorted(inputs.items())), PF_calls=0, optimizer_updates=0,
        limitations=['These are saved-data summaries, not new scientific audits.',
                    'Fit failure and score categories can overlap.',
                    'A/B workloads differ; batch times are not per-endpoint times.',
                    'Native PFJAX B versus C changes gradient construction and alpha.',
                    'Earlier PFJAX alpha=1 A/B are separate and retained.'])
    return {'summary.json': (json.dumps(overview, indent=2)+'\n').encode(),
            'likelihood.csv': csv_bytes(summaries), 'paired_differences.csv': csv_bytes(pairs),
            'fit_times.csv': csv_bytes(timing), 'pfjax_five_methods.csv': csv_bytes(five_summary)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    outputs = saved_summary()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        (args.output / name).write_bytes(data)
    print('Wrote five saved-data summaries to', args.output)


if __name__ == '__main__':
    main()
