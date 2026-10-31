#!/usr/bin/env python3
"""Verify copied evidence and patches offline. Does not run a PF or optimizer."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe archive path: {name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archives', action='store_true', help='Also read all archive members to verify decompression and ZIP CRCs.')
    ap.add_argument('--originals', type=Path, help='Optional original workspace to check source files still match this snapshot.')
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    imp = json.loads((ROOT / 'manifests/import.json').read_text())
    checked = 0
    for rel, rec in imp['files'].items():
        p = ROOT / rel
        assert p.is_file() and not p.is_symlink(), rel
        assert sha(p) == rec['sha256'] and p.stat().st_size == rec['bytes'], rel
        if a.originals:
            assert sha(a.originals / rec['source']) == rec['sha256'], rec['source']
        checked += 1
    repo_manifest = ROOT / 'manifests/repository_files.json'
    if repo_manifest.exists():
        for rel, digest in json.loads(repo_manifest.read_text())['files'].items():
            assert sha(ROOT / rel) == digest, rel
    patches = {}
    for package in ['cuthbert', 'pfjax', 'pydpf', 'tfp', 'particles']:
        pkg = ROOT / 'packages' / package
        with tempfile.TemporaryDirectory(prefix='comparisons-patch-') as tmp:
            clean = Path(tmp) / 'source'
            shutil.copytree(pkg / 'upstream', clean)
            patch = pkg / 'review/full.patch'
            subprocess.run(['git', 'apply', '--check', str(patch)], cwd=clean, check=True, capture_output=True)
            subprocess.run(['git', 'apply', str(patch)], cwd=clean, check=True, capture_output=True)
            expected = {p.relative_to(pkg / 'candidate').as_posix(): sha(p) for p in (pkg / 'candidate').rglob('*') if p.is_file()}
            actual = {p.relative_to(clean).as_posix(): sha(p) for p in clean.rglob('*') if p.is_file()}
            assert actual == expected, f'Patched {package} differs from candidate'
            patches[package] = {'patch_applied': True, 'all_files_match': True, 'files': len(actual)}
    def rows(rel):
        with (ROOT / rel).open() as f:
            return list(csv.DictReader(f))
    old = rows('reports/legacy_dacca/endpoints.csv')
    new = rows('experiments/public_interfaces/snapshot_20261031/recovery_20261031/particles/endpoints.csv')
    assert len(old) == 968
    assert sum(int(r['raw_repetitions']) for r in old) == 37584
    assert len(new) == 24 and len({r['task'] for r in new}) == 24
    assert sum(int(r['repetitions']) for r in new) == 1692
    assert sum(r['precision_passed'] == 'True' for r in new) == 21
    score = ROOT / 'experiments/public_interfaces/snapshot_20261031/shared/score.py'
    assert sha(score) == '8aec1ba02982a1ad9bff227b300806eaafa972a1e9fd00b71bef2c5f0ebdc508'
    archives = []
    if a.archives:
        for rec in json.loads((ROOT / 'manifests/archives.json').read_text()):
            path = ROOT / rec['path']
            count = 0
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as z:
                    for name in z.namelist():
                        relative(name)
                    assert z.testzip() is None, path
                    count = len(z.infolist())
            elif tarfile.is_tarfile(path):
                with tarfile.open(path) as tf:
                    for member in tf:
                        relative(member.name)
                        if member.isfile():
                            f = tf.extractfile(member)
                            while f.read(1024 * 1024):
                                pass
                        count += 1
            else:
                raise ValueError(f'Unrecognized archive: {path}')
            archives.append({'path': rec['path'], 'members_read': count})
    result = {'repository_integrity_passed': True, 'imported_files_checked': checked,
              'patches': patches, 'legacy_endpoints': 968, 'legacy_raw_PF_repetitions': 37584,
              'new_particles_endpoints': 24, 'new_particles_raw_PF_repetitions': 1692,
              'new_particles_precision_passed': 21, 'archives_read': archives,
              'scientific_experiments_rerun': False,
              'scope': 'Repository integrity and existing evidence; not a new GPU correctness or optimization result.'}
    text = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    print(json.dumps({k: v for k, v in result.items() if k != 'archives_read'}, indent=2))
    print('Archives fully read:', len(archives))


if __name__ == '__main__':
    main()
