TFP original Dacca, three methods,36 endpoints.
Read report_CN.txt and summary.json for results and limits.
Extract reproduction.tar.gz and run python verify_archive.py (Python+NumPy).
The verification script reads saved records; it never reruns fitting or PF.
Frozen sources and inputs are under source/. Original complete method archives and audit scripts are under evidence/.
Original package/runtime versions are pinned in source/source_versions.json and source/runtime_versions.json.
No original terminal manifests exist; after-completion retrieval hashes are retained.
All99 original PyPOMP reference source/data files are under reference_source/pypomp, checked against source/reference_source_hashes.json.
The3 executed TFP native files are under reference_source/tensorflow_probability/python/experimental/mcmc, checked against source/source_versions.json.
Root and visual review records, when present, are included unchanged.
Final delivery_manifest.json is outside the tar to avoid circular SHA hashing.
