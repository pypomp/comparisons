#!/usr/bin/env python3
"""Recalculate descriptive tables from saved endpoint scores, without fitting."""
import argparse
import csv
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from repository import ROOT, history_bytes


def summarize(rows, keys, repeats):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for group, rr in sorted(groups.items()):
        finite = [float(r['logmeanlik']) for r in rr if r['logmeanlik'] and math.isfinite(float(r['logmeanlik']))]
        out.append(dict(zip(keys, group), n=len(rr), finite_scores=len(finite),
                        nonfinite_scores=len(rr) - len(finite),
                        median_logmeanlik=statistics.median(finite) if finite else '',
                        fit_failed=sum(r['fit_failed'].lower() == 'true' for r in rr),
                        evaluation_failed=sum(any(r.get(k, '').lower() == 'true' for k in ('evaluation_failed', 'score_failure')) for r in rr),
                        floor_saturated=sum(r.get('score_floor_saturated', r.get('floor_saturated', '')).lower() == 'true' for r in rr),
                        precision_passed=sum(r['precision_passed'].lower() == 'true' for r in rr),
                        raw_repetitions=sum(int(r[repeats]) for r in rr)))
    return out


def csv_bytes(rows):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return f.getvalue().encode()


def tables():
    new = list(csv.DictReader((ROOT / 'results/particles/endpoints.csv').open()))
    old = list(csv.DictReader(io.StringIO(history_bytes('reports/legacy_dacca/endpoints.csv').decode())))
    # User excluded Dynamax from the main comparison; its 36 rows stay archived.
    main = [r for r in old if r['package'].lower() != 'dynamax']
    assert len(old) == 968 and len(main) == 932
    assert len(new) == 24 and len({r['task'] for r in new}) == 24
    assert sum(int(r['raw_repetitions']) for r in old) == 37584
    assert sum(int(r['repetitions']) for r in new) == 1692
    assert sum(r['precision_passed'] == 'True' for r in new) == 21
    return {
        'particles_summary.csv': csv_bytes(summarize(new, ['method'], 'repetitions')),
        'previous_summary.csv': csv_bytes(summarize(main, ['experiment_scope', 'package', 'method'], 'raw_repetitions')),
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'build/tables')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Choose a new output directory; existing results are not overwritten.')
    a.output.mkdir(parents=True)
    for name, data in tables().items():
        (a.output / name).write_bytes(data)
    print(f'Rebuilt two tables in {a.output.resolve()}; no PF calls or optimization.')
