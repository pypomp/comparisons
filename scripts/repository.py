"""Checksums and recovery helpers. Uses only Python's standard library."""
import gzip
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('cuthbert', 'pfjax', 'pydpf', 'tfp', 'particles')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    return p


def history():
    rec = json.loads((ROOT / 'archive/history.json').read_text())
    path = ROOT / rec['path']
    if not path.is_file():
        raise FileNotFoundError(f'Copy the archive listed in archive/catalog.json: {path}')
    if sha(path) != rec['sha256'] or path.stat().st_size != rec['bytes']:
        raise ValueError('History archive checksum mismatch')
    with gzip.open(ROOT / 'archive/files.json.gz', 'rt') as f:
        return path, json.load(f)


def history_bytes(name):
    relative(name)
    path, files = history()
    with tarfile.open(path) as tf:
        data = tf.extractfile(name).read()
    if hashlib.sha256(data).hexdigest() != files[name]['sha256']:
        raise ValueError(f'Archived file checksum mismatch: {name}')
    return data


def recent_history():
    """The small-file snapshot made before the latest report cleanup."""
    rec = json.loads((ROOT / 'archive/trim.json').read_text())
    with gzip.open(ROOT / rec['manifest'], 'rt') as f:
        return rec, json.load(f)


def saved_bytes(name):
    """Read a current file, or its verified archived copy if it was retired."""
    relative(name)
    path = ROOT / name
    if path.is_file() and not path.is_symlink():
        return path.read_bytes()
    rec, files = recent_history()
    if name not in rec['removed_paths'] or name not in files:
        raise FileNotFoundError(name)
    with tarfile.open(ROOT / rec['path']) as tf:
        data = tf.extractfile(name).read()
    if len(data) != files[name]['bytes'] or hashlib.sha256(data).hexdigest() != files[name]['sha256']:
        raise ValueError(f'Archived file checksum mismatch: {name}')
    return data


def archive_record(name):
    """Resolve a current archive path or its retained former path."""
    relative(name)
    records = json.loads((ROOT / 'archive/catalog.json').read_text())
    found = [r for r in records if name in (r['path'], r.get('previous_path'))]
    if len(found) != 1:
        raise FileNotFoundError(name)
    return found[0]


def restore(destination, prefix='', strip_prefix=False):
    """Restore selected regular files into a new directory; never overwrite."""
    destination = Path(os.path.abspath(destination))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Choose a new destination: {destination}')
    if destination.is_relative_to(ROOT) and destination.relative_to(ROOT).parts[0] not in ('build', 'extracted', 'scratch'):
        raise ValueError('Inside this repository use build/, extracted/ or scratch/.')
    if prefix:
        relative(prefix)
    path, files = history()
    prefix = prefix.rstrip('/')
    selected = {n: r for n, r in files.items() if not prefix or n == prefix or n.startswith(prefix + '/')}
    if not selected:
        raise ValueError(f'No archived files at {prefix}')
    destination.mkdir(parents=True)
    seen = set()
    with tarfile.open(path) as tf:
        for m in tf:
            if m.name not in selected:
                continue
            relative(m.name)
            if not m.isfile() or m.name in seen:
                raise ValueError(f'Unexpected archive member: {m.name}')
            name = m.name[len(prefix) + 1:] if strip_prefix else m.name
            relative(name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, target.open('xb') as out:
                shutil.copyfileobj(src, out)
            rec = selected[m.name]
            if sha(target) != rec['sha256'] or target.stat().st_size != rec['bytes']:
                raise ValueError(f'Restored file checksum mismatch: {m.name}')
            target.chmod(rec['mode'])
            seen.add(m.name)
    if seen != set(selected):
        raise ValueError('Missing archived files')
    return len(seen)
