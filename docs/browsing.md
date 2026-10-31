# Browse comparisons locally

Run the following in a terminal:

```bash
cd /Users/mac/Documents/Codex/2026-10-22/cloud-gl/comparisons
python3 scripts/browse.py
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/) in a browser. The page provides directory navigation, file search, Markdown previews, source line numbers, colored diffs, image previews and original-file downloads. Each modified package has a **Diff** link in the sidebar.

Start with **Dacca report**, **Package functions** and **How each package is used**, then open the package's `review/full.patch`. The `upstream/` directory contains the corresponding original code; `candidate/` contains the code with our additions; `validation/` contains validation programs and records. Experiments and reports are under `experiments/` and `reports/`, with earlier experiments separate from the new interfaces.

English is the default for the browser and review documents. Chinese notes remain available as supplementary documents. Original sources, logs and historical reports retain their recorded language.

The browser is read only and listens on `127.0.0.1`. It has no remote repository or publication function. Press Ctrl+C in its terminal to stop it; run the command again to restart. When copying the Git repository to another computer, also copy the original archives under `artifacts/`, listed in `manifests/archives.json`.

The browser does not run experiments or refresh scheduler state. Its file index is built at startup; restart it after adding files.
