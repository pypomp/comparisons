"""Verify every archived review file against SHA256SUMS.json."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SHA256SUMS.json").read_text())
for name, expected in manifest.items():
    path = (root / name).resolve()
    assert root == path or root in path.parents, name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
print(f"Verified {len(manifest)} files")
