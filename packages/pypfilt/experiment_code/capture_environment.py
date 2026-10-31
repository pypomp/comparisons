"""Read-only environment provenance; does not import GPU libraries."""
import argparse,hashlib,importlib.metadata,json,pathlib,platform,sys

p=argparse.ArgumentParser();p.add_argument('--pypomp-source',required=True);a=p.parse_args()
packages=[]
for d in importlib.metadata.distributions():
    record=d.read_text('RECORD')
    packages.append(dict(name=d.metadata['Name'],version=d.version,
        record_sha256=None if record is None else hashlib.sha256(record.encode()).hexdigest()))
root=pathlib.Path(a.pypomp_source)
sources={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*.py'))}
print(json.dumps(dict(python=platform.python_version(),executable=sys.executable,
    executable_sha256=hashlib.sha256(pathlib.Path(sys.executable).resolve().read_bytes()).hexdigest(),
    machine=platform.machine(),packages=sorted(packages,key=lambda d:(d['name'].lower(),d['version'])),
    pypomp_source_files=sources),indent=2))
