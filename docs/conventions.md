# Repository conventions

The repository separates package changes, experiment programs and reports. A reader should be able to find the original function, inspect its diff, and follow a reported number back to a saved result.

## Examples consulted

These are specific examples from local checkouts, not claims about every file written by either author.

| Source | Observed practice | Use here |
|---|---|---|
| Kevin Tan, `pypomp/dmop`, commit `fa3b6167a613b5d4ec44d177282f7126dfb32588`, changes to `ms.tex` and `si.tex` | Removes stock phrasing and unsupported adjectives; states operations and measured quantities directly | Short descriptions of the code, test conditions and measured results |
| `pypomp/quant`, local revision `526402cd49a84ffd21b4bfea385b33eefcf19f03`, committed by `hetankevin`; `README.md`, `makefile`, `tests/dacca/timing/run.py` | Reports are indexed by scientific question; timing programs identify starts and timed work; common commands have Make targets | Separate correctness, PF runtime and optimization reports; a small Makefile for local recovery and checks |
| `pypomp/quant`, local revision `a8a519aae4332e49a8b773cf7772b8d9ac4afff8`, committed by Aaron Abkemeier; `tests/spx/scaling/run.py` and `report.qmd` | Model code, experiment execution, saved results and report generation are separate; reports identify failed runs and distinguish compilation from warm timings | Keep package changes separate from experiment drivers; derive tables from saved records and retain failures |
| `pyMeasles`, local revision `747ce9a43cb56ebfc9801e3ce4f0f6fafa5a0077`, committed by Aaron Abkemeier; `lab/model_001b/search_002/compare_arms.py` | Analysis programs state the input records and the precise quantities they compute; runs retain script snapshots | Keep analysis separate from fitting and retain the source used by each run |

Commit authors identify the revisions inspected. These repositories contain shared work; a commit author alone does not establish authorship of every line. The examples were read locally. The latest remote branches were not checked.

## Code and files

`packages/<name>/changes/` follows that package's own module structure and coding conventions. `review/full.patch` records the complete change against the version in `version.json`. Keep library functions general; Dacca settings belong in the experiment or model callbacks.

`experiments/dacca/<name>/` contains the recorded protocol and fitting programs. These are preserved copies of the experimental code. Some programs still use their original workspace paths; the [reproduction instructions](reproduction.md) explain how to restore the full layout. Reformatting or moving a frozen program would change the record of what ran.

`scripts/` contains local recovery and analysis commands. Each program should state its inputs and outputs, use explicit command-line arguments where needed, and keep work under a `main` function or a main guard. Shared archive handling belongs in `repository.py`. Make targets call these programs; they do not submit experiments.

`results/` holds saved estimates and derived tables. `reports/` explains the study and results. Put complete source snapshots, checkpoints, logs and failed runs in dated archives, with checksums and extraction instructions. New numerical evidence gets a new dated record. Keep superseded records available.

## Writing reports

Use the package's function names. Say which function is original, what we changed, and which optimizer updates the parameters. A newly added public function is still our code until accepted upstream.

State the model, particle count, starts, repetitions, hardware and measured quantity. Give the denominator when reporting successes or medians. Distinguish observations from explanations that remain untested. Keep compilation, ordinary PF calls, fitting and final scoring costs separate. A completed scheduler job does not imply that a numerical comparison passed.

Correctness, ordinary PF runtime and likelihood optimization have separate reports. The main study description gives the model and design; the status page records dated progress. Avoid repeating the same numbers across these pages.

## Meeting decisions

The 29 October 2026 transcript supports the following organization. Times are those printed in the transcript.

| Speaker and time | Request | Location here |
|---|---|---|
| Kevin, 15:23:10–15:23:41 | Separate runtime, correctness and likelihood performance; compare runtime using the common PF | The three reports linked from the README |
| Ed, 15:27:20–15:28:52 | Use the package's framework and style; add general parameter perturbations and extend its ADPF | Per-package changes and function descriptions |
| Giles, 15:29:17–15:29:59 | Make the change from original code intelligible as a diff | `packages/<name>/review/full.patch` |
| Ed, 15:54:22–15:55:17 | Develop the comparison and report in quant or a comparisons repository, then share the report and code | This local repository and its report; publication remains a separate step |

Exact A/B/C iteration counts and random-stream settings were chosen after that discussion. The archived meeting notes can be restored with the command in [Reproduction](reproduction.md).
