"""Verify extracted review files, then apply the full patch to the baseline."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    for name, expected in sums.items():
        path = root / name
        assert path.resolve().is_relative_to(root), name
        assert digest(path) == expected, f"SHA256 mismatch: {name}"
    initial = json.loads((root / "initial_source_check.json").read_text())
    review = json.loads((root / "review_manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="pydpf-verify-") as tmp:
        clean = Path(tmp).resolve()
        with tarfile.open(root / "upstream.tar.gz") as tf:
            for member in tf.getmembers():
                assert member.isfile() and (clean / member.name).resolve().is_relative_to(clean)
                destination = clean / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as stream:
                    destination.write_bytes(stream.read())
        for row in initial["baseline_files"]:
            assert digest(clean / row["path"]) == row["sha256"]
        for command in (["git", "apply", "--check", str(root / "full.patch")], ["git", "apply", str(root / "full.patch")]):
            subprocess.run(command, cwd=clean, check=True, capture_output=True, text=True)
        expected = review["candidate_files"]
        found = {str(p.relative_to(clean)): digest(p) for p in clean.rglob("*") if p.is_file()}
        assert found == expected, "patched files differ from reviewed candidate"
    print(json.dumps({"passed": True, "hashed_files": len(sums), "baseline_files": len(initial["baseline_files"]), "patched_files": len(expected), "numerical_tests_rerun": False}))


if __name__ == "__main__":
    main()
