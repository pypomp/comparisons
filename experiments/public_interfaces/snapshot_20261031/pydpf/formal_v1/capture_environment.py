"""Read-only metadata snapshot; does not initialize CUDA or import Torch/JAX."""
import hashlib,importlib.metadata,json,pathlib,platform,sys
packages=[]
for d in importlib.metadata.distributions():
    record=d.read_text('RECORD')
    packages.append(dict(name=d.metadata['Name'],version=d.version,
        record_sha256=None if record is None else hashlib.sha256(record.encode()).hexdigest()))
print(json.dumps(dict(python=platform.python_version(),executable=sys.executable,
    executable_sha256=hashlib.sha256(pathlib.Path(sys.executable).resolve().read_bytes()).hexdigest(),
    machine=platform.machine(),packages=sorted(packages,key=lambda d:(d['name'].lower(),d['version']))),indent=2))
