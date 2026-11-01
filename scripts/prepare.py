#!/usr/bin/env python3
"""Restore pinned original source, apply the full diff, verify every file."""
import argparse
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from repository import ROOT, PACKAGES, history, restore, sha


def prepare(package, output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Choose a new destination: {output}')
    if output.is_relative_to(ROOT) and output.relative_to(ROOT).parts[0] not in ('build', 'extracted', 'scratch'):
        raise ValueError('Inside this repository use build/, extracted/ or scratch/.')
    _, files = history()
    # Apply outside this Git checkout so Git cannot reinterpret relative paths.
    with tempfile.TemporaryDirectory(prefix='comparisons-source-') as tmp:
        source = Path(tmp) / 'source'
        restore(source, f'packages/{package}/upstream', strip_prefix=True)
        patch = ROOT / f'packages/{package}/review/full.patch'
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=source, check=True, capture_output=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=source, check=True, capture_output=True)
        prefix = f'packages/{package}/candidate/'
        expected = {n[len(prefix):]: r['sha256'] for n, r in files.items() if n.startswith(prefix)}
        actual = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
        if actual != expected:
            raise ValueError(f'Patched {package} differs from saved modified source')
        shutil.copytree(source, output)
    return {'package': package, 'files_checked': len(actual), 'all_files_match': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package', choices=PACKAGES)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.package, a.output), indent=2))
