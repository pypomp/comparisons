from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source_manifest.json').read_text())
for name,expected in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({'verified_files':len(manifest['files']),'manifest_sha256':hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest()}))
