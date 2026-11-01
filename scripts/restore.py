#!/usr/bin/env python3
"""Restore the previous repository layout, or one path in it. No experiments."""
import argparse
import json
import shutil
from pathlib import Path
from repository import ROOT, history_bytes, relative, restore, sha

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', default='', help='Optional path in the archived repository')
    p.add_argument('--archives', action='store_true', help='Also copy the 50 original bulk archives for the former verification program')
    a = p.parse_args()
    if a.archives and a.prefix:
        p.error('--archives requires a full restoration without --prefix')
    n = restore(a.output, a.prefix)
    if a.archives:
        archives = json.loads(history_bytes('manifests/archives.json'))
        for rec in archives:
            relative(rec['path'])
            src, dst = ROOT / rec['path'], a.output / rec['path']
            if dst.exists() or dst.is_symlink():
                raise FileExistsError(dst)
            if sha(src) != rec['sha256'] or src.stat().st_size != rec['bytes']:
                raise ValueError(f"Archive checksum mismatch: {rec['path']}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if sha(dst) != rec['sha256']:
                raise ValueError(f"Copied archive checksum mismatch: {rec['path']}")
        print(f'Copied and verified {len(archives)} bulk archives')
    print(f'Restored and verified {n} files in {a.output.resolve()}')
