from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
m=json.loads((root/'manifest.json').read_text())
for name,sha in m['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
print('Verified',len(m['sha256']),'source/input hashes',flush=True)
