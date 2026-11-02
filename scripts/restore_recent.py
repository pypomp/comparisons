#!/usr/bin/env python3
"""Restore selected files from the latest cleanup snapshot into a new directory."""
import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path
from repository import ROOT, recent_history, relative, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', default='')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.is_symlink():
        raise FileExistsError('Choose a new output directory.')
    if a.prefix:
        relative(a.prefix)
    rec, files = recent_history()
    assert sha(ROOT / rec['path']) == rec['sha256']
    prefix = a.prefix.rstrip('/')
    selected = {n: r for n, r in files.items() if not prefix or n == prefix or n.startswith(prefix + '/')}
    if not selected:
        raise FileNotFoundError(prefix)
    a.output.mkdir(parents=True)
    with tarfile.open(ROOT / rec['path']) as tf:
        for name, item in selected.items():
            relative(name)
            m = tf.getmember(name)
            assert m.isfile()
            dest = a.output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, dest.open('xb') as out:
                shutil.copyfileobj(src, out)
            assert sha(dest) == item['sha256'] and dest.stat().st_size == item['bytes']
    print(f'Restored {len(selected)} files to {a.output.resolve()}.')


if __name__ == '__main__':
    main()
