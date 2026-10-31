"""Verify archive contents and reconstruct the candidate; no model runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def files(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
def reconstruct(folder, manifest):
    upstream = folder / 'upstream'
    candidate = folder / 'candidate'
    assert not upstream.exists() and not candidate.exists()
    upstream.mkdir()
    with tarfile.open(ROOT / 'upstream.tar.gz') as tf:
        for member in tf.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute()
            assert '..' not in Path(member.name).parts
        tf.extractall(upstream)
    assert files(upstream) == manifest['upstream_files']
    shutil.copytree(upstream,candidate)
    subprocess.run(['git','apply','--check',str(ROOT/'full.patch')],cwd=candidate,check=True)
    subprocess.run(['git','apply',str(ROOT/'full.patch')],cwd=candidate,check=True)
    assert files(candidate) == manifest['candidate_files']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    hashes=json.loads((ROOT/'SHA256SUMS.json').read_text())
    for name,digest in hashes.items():
        assert sha(ROOT/name)==digest,name
    manifest=json.loads((ROOT/'review_manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='particles-review-') as temp:
        reconstruct(Path(temp),manifest)
    if args.prepare:
        reconstruct(ROOT,manifest)
    print(json.dumps({'archive_files_verified':len(hashes),'patch_applies':True,
                      'reconstructed_files_match':True,'numerical_checks_run':False}))

if __name__=='__main__':
    main()
