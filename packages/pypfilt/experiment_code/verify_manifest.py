from pathlib import Path
import argparse,hashlib,json,os

def verify(name='manifest.json'):
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/name).read_text())
    for file,expected in manifest['files'].items():
        assert hashlib.sha256((root/file).read_bytes()).hexdigest()==expected,file
    external=manifest.get('external_reference_source_files',{})
    reference=Path(os.environ.get('PFB_REFERENCE_SOURCE',str(root.parents[3]/'work/cuthbert-dacca-paper/source')))
    for file,expected in external.items():
        assert hashlib.sha256((reference/file).read_bytes()).hexdigest()==expected,file
    return dict(verified_files=len(manifest['files']),verified_reference_files=len(external),manifest_sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='manifest.json')
    print(json.dumps(verify(p.parse_args().manifest)))
