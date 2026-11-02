#!/usr/bin/env python3
"""Extract one catalogued archive into a NEW directory, with checksum/path checks."""
import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from repository import ROOT, sha, relative, archive_record

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('archive', help='Path under artifacts/, as listed in archive/catalog.json')
p.add_argument('destination', type=Path)
a = p.parse_args()
rec = archive_record(a.archive)
src = ROOT / rec['path']
assert sha(src) == rec['sha256'], 'Archive hash mismatch'
if a.destination.exists():
    raise SystemExit('Destination already exists. Choose a new directory to preserve existing evidence.')
if zipfile.is_zipfile(src):
    with zipfile.ZipFile(src) as z:
        for m in z.infolist():
            relative(m.filename)
            if (m.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symbolic links are not extracted by this helper')
        z.extractall(a.destination)
else:
    with tarfile.open(src) as tf:
        members = tf.getmembers()
        for m in members:
            relative(m.name)
            if not (m.isfile() or m.isdir()):
                raise ValueError(f'Non-regular member requires manual review: {m.name}')
        a.destination.mkdir(parents=True)
        for m in members:
            target = a.destination / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as f, target.open('wb') as out:
                    shutil.copyfileobj(f, out)
print(a.destination.resolve())
