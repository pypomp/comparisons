from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
m=json.loads((root/'source_manifest.json').read_text())
for name,value in m['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,name
print(json.dumps({'source_files_verified':len(m['files'])}),flush=True)
