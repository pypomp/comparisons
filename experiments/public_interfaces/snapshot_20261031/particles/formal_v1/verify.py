from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
for n,h in json.loads((root/'manifest.json').read_text())['sha256'].items():
    assert hashlib.sha256((root.parent/n).read_bytes()).hexdigest()==h,n
print('Formal source manifest verified',flush=True)
