#!/usr/bin/env python3
"""One-time local import. Copies evidence; never runs experiments or contacts a server."""
import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('workspace', type=Path)
    ap.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    ws, dst = args.workspace.resolve(), args.destination.resolve()
    manifest = dst / 'manifests/import.json'
    if manifest.exists():
        raise SystemExit('Import already recorded; create a new dated snapshot instead of overwriting evidence.')
    records, archive_records = {}, []
    skip = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.DS_Store', '.venv', 'venv', 'node_modules'}

    def copy(src, target, expected=None):
        src, target = Path(src), dst / target
        if src.is_symlink() or not src.is_file():
            raise ValueError(f'Expected regular source file: {src}')
        before = sha(src)
        if expected is not None and before != expected:
            raise ValueError(f'Original checksum mismatch: {src}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != before:
                raise ValueError(f'Conflicting destination: {target}')
        else:
            shutil.copy2(src, target)
        if sha(src) != before or sha(target) != before:
            raise ValueError(f'File changed during copying: {src}')
        rel = target.relative_to(dst).as_posix()
        records[rel] = {'source': src.relative_to(ws).as_posix(), 'sha256': before, 'bytes': target.stat().st_size}
        if rel.startswith('artifacts/'):
            archive_records.append({'path': rel, **records[rel]})
        return rel

    def is_archive(p):
        return p.name.endswith(('.tar.gz', '.tgz', '.zip', '.tar'))

    def tree(src, target, excluded=(), recursive=True):
        src = Path(src)
        if not src.exists():
            return
        it = src.rglob('*') if recursive else src.iterdir()
        for p in sorted(it):
            rel = p.relative_to(src)
            if any(x in skip or x in excluded for x in rel.parts) or p.is_symlink() or not p.is_file():
                continue
            if p.suffix in {'.pyc', '.pyo'}:
                continue
            to = Path(target) / rel
            if is_archive(p):
                to = Path('artifacts') / to
            copy(p, to)

    def write(rel, obj):
        p = dst / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

    outs = ws / 'outputs/competitor_estimation'
    camp = ws / 'work/implementation-comparison-20261030'
    packages = {}
    versions = {'cuthbert': 3, 'pfjax': 2, 'pydpf': 1, 'tfp': 1, 'particles': 1}
    for package, version in versions.items():
        review_name = f'{package}_implementation_v{version}' + ('_draft' if package == 'cuthbert' else '')
        review = outs / review_name
        evidence = json.loads((review / 'review_manifest.json').read_text())
        source = ws / f'work/{package}-implementation-v{version}'
        clean = camp / 'cuthbert_gate_v3/cuthbert' if package == 'cuthbert' else source
        selected = evidence.get('candidate_files', evidence.get('source_files'))
        for name, digest in selected.items():
            copy(clean / 'candidate' / name, f'packages/{package}/candidate/{name}', digest)
        # Upstream files are selected from the reviewed paths, never copied from a live environment.
        upstream = evidence.get('upstream_files')
        if upstream is not None:
            for name, digest in upstream.items():
                copy(clean / 'upstream' / name, f'packages/{package}/upstream/{name}', digest)
        else:
            for name in selected:
                p = clean / 'upstream' / name
                if p.is_file():
                    copy(p, f'packages/{package}/upstream/{name}')
        tree(review, f'packages/{package}/review')
        tree(source / 'validation', f'packages/{package}/validation')
        tree(source / 'validation_results', f'packages/{package}/validation_results')
        packages[package] = {
            'revision': version,
            'base': evidence.get('base_commit', evidence.get('commit', evidence.get('baseline'))),
            'baseline_scope': evidence.get('baseline_scope', 'See original review manifest and preserved upstream files.'),
            'review_manifest': f'packages/{package}/review/review_manifest.json',
            'patch': f'packages/{package}/review/full.patch',
            'candidate_files': len(selected),
            'upstream_accepted': False,
        }
    copy(camp / 'upstream_repositories/tfp/LICENSE', 'packages/tfp/LICENSE.upstream')

    # Completed earlier comparisons, with their own frozen implementation and final scoring.
    index = json.loads((outs / 'final_comparison/index_manifest.json').read_text())
    finals = {}
    for label, item in index['source_delivery_archives'].items():
        src = Path(item['path'])
        key = 'paper_cuthbert_pypomp' if 'cuthbert_dacca_paper' in str(src) else label.lower()
        folder = f'experiments/legacy_dacca/{key}'
        tree(src.parent, folder, recursive=False)
        copied = copy(src, f'artifacts/{folder}/{src.name}', item['sha256'])
        finals[key] = {'archive': copied, 'sha256': item['sha256'], 'report_directory': folder}
    tree(outs / 'final_comparison', 'reports/legacy_dacca')
    tree(outs / 'overview_v1', 'reports/legacy_dacca/figures_all')
    tree(outs / 'overview_iqr_v1', 'reports/legacy_dacca/figures_iqr')
    tree(outs / 'package_call_boundaries_v2', 'reports/package_calls_legacy')
    for name in ['implementation_requirements_20261029.txt', 'next_steps_20261029.txt', 'cuthbert_coding_audit_20261029.txt', 'cuthbert_coding_audit_20261029.json']:
        copy(outs / name, f'docs/meeting_notes/{name}')
    copy(ws / 'work/competitor-estimation/plot_overview.py', 'reports/legacy_dacca/plot_overview_original.py')
    tree(outs / 'pfjax/formal_v1/history/score_v1_before_independence_fix', 'experiments/legacy_dacca/pfjax/superseded_scores')
    for file in (outs / 'pfjax/formal_v1').glob('*review*'):
        if file.is_file():
            copy(file, f'experiments/legacy_dacca/pfjax/{file.name}')

    # Readable old adapters supplement the complete archived sources and raw records.
    older = ws / 'work/competitor-estimation'
    for package, rel in [('dynamax', 'dynamax/dacca/formal_fixed_v1'), ('pypfilt', 'pypfilt/formal_v1')]:
        root = older / rel
        for p in sorted(root.iterdir()):
            if p.is_file() and not is_archive(p):
                copy(p, f'packages/{package}/experiment_code/{p.name}')
        for name in ['adapter', 'tools']:
            tree(root / name, f'packages/{package}/experiment_code/{name}')
        packages[package] = {'new_public_interface': False, 'source_and_raw_archive': finals[package]['archive']}
    tree(ws / 'work/cuthbert-dacca-paper/source/pypomp', 'packages/pypomp/reference_source/pypomp')
    for name in ['LICENSE', 'pyproject.toml', 'CITATION.bib', 'README.md']:
        copy(ws / 'work/cuthbert-dacca-paper/source' / name, f'packages/pypomp/reference_source/{name}')
    packages['pypomp'] = {'base': '17f8798e7d944977335c81b8525353fa90e14be9', 'role': 'Paper comparator and fixed independent endpoint scorer', 'source_and_raw_archive': finals['paper_cuthbert_pypomp']['archive']}

    # Earlier Cuthbert probes are history, never merged with J5000 comparisons.
    for name in ['cuthbert_case_bundle_manifest.json', 'cuthbert_common_warm_bundle_manifest.json']:
        p = ws / 'outputs' / name
        d = json.loads(p.read_text())
        copy(p, f'experiments/history/{name}')
        copy(Path(d['path']), f'artifacts/experiments/history/{Path(d["path"]).name}', d['sha256'])

    # Current comparison: readable drivers and protocols plus existing frozen archives.
    # Installed dependencies, mutable raw directories, duplicate source trees and live
    # upstream test runs are omitted here; source snapshots and raw archives are above/below.
    excluded = {'test_deps', 'test-deps', 'test-science-deps', 'cuthbert_example_deps', 'candidate', 'upstream', 'source',
                'reference_source', 'gl', 'raw', 'results', 'original', 'upstream_tests',
                'upstream_tests_with_deps', 'upstream_repositories', 'local-venv'}
    tree(camp, 'experiments/public_interfaces/snapshot_20261031', excluded=excluded)
    tree(camp / 'cuthbert_formal_v3/reference_source', 'experiments/public_interfaces/reference_source')
    for name in ['tfp.tar.gz', 'pydpf.tar.gz', 'sources.json']:
        p = camp / 'upstream_repositories' / name
        prefix = 'artifacts/' if is_archive(p) else ''
        copy(p, prefix + 'experiments/public_interfaces/upstream_test_sources/' + name)

    # Only complete paired files are copied from the still-running local TFP tests.
    tests = camp / 'tfp/upstream_tests_with_deps'
    out = 'experiments/public_interfaces/tfp_upstream_tests_snapshot'
    pairs = sorted((tests / 'pairs').glob('*.json'))
    counts = {'completed_pairs_copied': len(pairs), 'expected_pairs': 515, 'finished': False,
              'candidate_only_failing_pairs': 0, 'any_failing_or_incomplete_original_pair': 0}
    for p in pairs:
        d = json.loads(p.read_text())
        if len(d['results']) != 2 or any(x['state'] != 'finished' for x in d['results']):
            # Timeouts are terminal evidence too; the pair JSON is the runner's completion marker.
            pass
        copy(p, f'{out}/pairs/{p.name}')
        counts['candidate_only_failing_pairs'] += bool(d.get('candidate_only_failure'))
        counts['any_failing_or_incomplete_original_pair'] += any(x.get('exit_code') != 0 for x in d['results'])
        for result in d['results']:
            xml = next((x.split('=', 1)[1] for x in result['command'] if x.startswith('--junitxml=')), None)
            if xml:
                rec = Path(xml).parent
                tree(rec, f'{out}/records/{rec.name}')
    for name in ['environment.json', 'inventory.json']:
        copy(tests / name, f'{out}/{name}')
    write(f'{out}/snapshot.json', counts)

    # Older public-interface revisions remain available for tracing failed validation.
    for name in ['cuthbert_implementation_v2', 'pfjax_implementation_v1']:
        tree(outs / name, f'experiments/history/{name}')
    tree(outs / 'implementation_comparison_20261030', 'reports/public_interfaces', excluded={'raw', 'gl'})

    write('manifests/packages.json', packages)
    write('manifests/completed_legacy_deliveries.json', finals)
    unique_archives = {x['path']: x for x in archive_records}
    write('manifests/archives.json', list(unique_archives.values()))
    write('manifests/import.json', {
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_workspace': str(ws),
        'purpose': 'Local repository assembly; no fit, score, submission, or remote publication.',
        'files': records,
        'tfp_test_snapshot': counts,
        'originals_checked_before_and_after_copy': True,
    })
    print(json.dumps({'imported_files': len(records), 'bytes': sum(x['bytes'] for x in records.values()),
                      'archives': len(unique_archives), 'tfp_tests': counts}, indent=2))


if __name__ == '__main__':
    main()
