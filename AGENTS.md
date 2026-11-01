# Working in comparisons

This is a local comparison repository. Preserve imported evidence and its SHA256 records.

- Write plain technical English. Name the package function, code change, test or measured quantity. Avoid vague labels such as "evidence bundle", "implementation boundary" or "release gate" in review prose. Preserve exact function names and filenames.
- When attributing a requirement to a meeting, cite the speaker and transcript time. Distinguish meeting proposals from later experiment settings and user decisions.
- English is the primary language for the browser, README files and documents intended for package authors. Keep Chinese notes as explicitly identified supplementary documents. Preserve the language and bytes of frozen sources and historical evidence.

- Start with README.md, reports/status.md and the dated experiment status. Old job records are history; use the active workspace jobs.json and a real scheduler query before taking cluster actions.
- Keep correctness, native PF runtime and optimization outcomes separate. Old driver results do not validate later public interfaces.
- State the original package function, added code and external optimizer explicitly. Added public APIs are not upstream-native methods.
- Dynamax is excluded from the main comparison by the user's 31 October 2026 decision. Do not add further Dynamax implementation or experiments, or include its self-written PF in future main tables, figures or conclusions. Keep its earlier evidence as history.
- Do not edit a frozen source, overwrite original raw records, delete failures, choose the best of old/new scores, rerun completed fits or silently relax tolerances.
- Append a new dated snapshot for new evidence or a new code revision. Update the catalog and checksums. Never start a collector merely to assemble or inspect this repository.
- CPU/GPU, particle counts, starts, iteration and time budgets must remain distinguishable. Batch execution time is counted once.
- Final scoring uses the stated independent fixed PF and retains raw values, MCSE, likelihood ESS, failures and density-floor saturation.
- If figures hide points using 1.5×IQR, retain all data and summary statistics and state the display rule.
- Bulk archives in artifacts/ exist locally but are not in Git. Use archive/catalog.json when transferring them.
- Run scripts/verify.py after repository changes; it is an integrity check, not a numerical validation claim.
- Do not create a remote, send messages, push or open a PR without explicit user instructions.

- Keep the main tree small: changed code, full diffs, protocols, principal programs, result tables and a short report. Archive logs, complete source trees and prior reports with hashes and recovery instructions. Never discard failures to simplify presentation.
