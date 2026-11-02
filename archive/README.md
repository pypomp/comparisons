# Archive

The main tree contains current reports, plots, data and code. Earlier progress reports, superseded figures and temporary outputs are stored in `artifacts/history/comparisons_before_trim_20261102.tar.gz`. [Snapshot record](trim.json) and `trim_files.json.gz` give file hashes. Existing links in the local browser open the archived copy.

```bash
python3 scripts/restore_recent.py --prefix reports/pydpf_20261102 --output build/old_report
python3 scripts/extract_archive.py artifacts/experiments/dacca/pydpf_review_20261102.tar.gz build/full_run
```

The [catalog](catalog.json) lists archives of complete source code and raw results, with SHA256 hashes and file sizes. Older archives are under `artifacts/history/previous`. The extraction command also accepts their former paths, recorded in `previous_path`. File contents and failure records are unchanged.

The earlier 8,527-file snapshot remains in [history.json](history.json), with its manifest in `files.json.gz`. Use `scripts/restore.py` to recover it. It includes the old 968 final estimates, meeting transcript, original PF review and Dynamax records. Those results do not validate the package functions added later.

Archives are local and ignored by Git. Include catalogued archives when transferring the repository. New summaries do not overwrite archived source, raw data or failed experiments.
