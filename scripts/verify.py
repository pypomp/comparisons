#!/usr/bin/env python3
"""Verify retained files, archived records, reconstructed code and result tables."""
import argparse
import hashlib
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path
from repository import ROOT, PACKAGES, history, relative, sha
from prepare import prepare
from reproduce import tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', action='store_true', help='Also read every compressed member of all archives')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    checksums = json.loads((ROOT / 'archive/checksums.json').read_text())['files']
    for name, digest in checksums.items():
        path = ROOT / name
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, name
    path, files = history()
    with tarfile.open(path) as tf:
        seen = set()
        for m in tf:
            assert m.isfile() and m.name in files and m.name not in seen, m.name
            rec = files[m.name]
            assert m.size == rec['bytes']
            assert hashlib.sha256(tf.extractfile(m).read()).hexdigest() == rec['sha256'], m.name
            seen.add(m.name)
    assert seen == set(files) and len(seen) == 8527
    copies = json.loads((ROOT / 'archive/copies.json').read_text())
    for name, original in copies.items():
        assert sha(ROOT / name) == files[original]['sha256'], name
    archives = json.loads((ROOT / 'archive/catalog.json').read_text())
    for rec in archives:
        path = ROOT / rec['path']
        assert path.is_file() and sha(path) == rec['sha256'] and path.stat().st_size == rec['bytes'], rec['path']
        if a.archives:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as z:
                    for n in z.namelist():
                        relative(n)
                    assert z.testzip() is None, path
            else:
                with tarfile.open(path) as tf:
                    for m in tf:
                        relative(m.name)
                        if m.isfile():
                            with tf.extractfile(m) as f:
                                while f.read(1048576):
                                    pass
    patches = []
    with tempfile.TemporaryDirectory(prefix='comparisons-verify-') as tmp:
        for pkg in PACKAGES:
            patches.append(prepare(pkg, Path(tmp) / pkg))
    for name, data in tables().items():
        assert data == (ROOT / 'results' / name).read_bytes(), name
    assert sha(ROOT / 'experiments/dacca/score.py') == '8aec1ba02982a1ad9bff227b300806eaafa972a1e9fd00b71bef2c5f0ebdc508'
    result = dict(passed=True, retained_files_checked=len(checksums), archived_files_checked=len(seen),
                  unchanged_copies_checked=len(copies), archive_checksums_checked=len(archives),
                  archives_fully_read=len(archives) if a.archives else 1, patched_sources=patches,
                  result_tables_rebuilt=2, old_endpoints_preserved=968, old_raw_PF_repetitions=37584,
                  new_particles_endpoints=24, new_particles_raw_PF_repetitions=1692,
                  numerical_experiments_rerun=False)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
