# Meeting notes: 29 October 2026

Source: `dmop etc transcript_2026-10-29_16.02.38.txt`, supplied by Kunyang. Times below follow the transcript. The automatic transcript misspells package names and sometimes labels Kunyang's replies as Edward Ionides. The account below uses the clear questions and continuous exchanges; mixed-speaker replies are not treated as Ed's statements. The private transcript is not copied into this repository. [Source details and checked excerpts](meeting_review_20261031.json) identify the passages used.

## What was requested

| Discussion | Meaning for this work | Where to inspect it |
|---|---|---|
| Ed, 15:20:13 and 15:20:40 to 15:21:40: could this be a meaningful contribution to the package, and which Cuthbert functions does it use? | Show the actual PF calls. A package name on a plot is insufficient when its only contribution is an optimizer or the underlying array library. | [Function table](package_calls.md) and [source review](../reports/native_pf_audit_20261031/report.md) |
| Ed, 15:27:20 and 15:27:43 to 15:28:52: use the package's framework; add parameter perturbations without rewriting every model; extend its existing ADPF with α. | Put IF2's parameter population, perturbations and cooling in reusable functions. For Cuthbert, explain the change to the original ADPF derivative calculation. | [Cuthbert code and diff](../packages/cuthbert/README.md) |
| Giles, 15:29:17 to 15:29:59: start from their code, make the necessary changes and provide an intelligible diff. | Keep a specified original version beside the modified code. Explain new loops as well as edits to existing functions. A successful patch application alone does not establish that the changes are small or appropriate. | Each package's `upstream/`, `candidate/` and `review/full.patch` |
| Kevin, 15:22:47 to 15:23:41: use the method's actual name and separate runtime, correctness and likelihood performance. | Name the derivative calculation and optimizer. Time ordinary PF calls separately from fitting. Use numerical tests for correctness and independent PF scores for optimization. | [Results and tests](../reports/status.md) |
| Ed and Giles, 15:24:45 to 15:25:17 and 15:53:37 to 15:53:55: compare methods within Cuthbert; show the existing method, that method after IF2, and the proposed method. | Compare initialization and discounting within the same package. Explain the extra work used by IF2. | [A/B/C experiment](../experiments/public_interfaces/README.md) |
| Ed, 15:32:06 to 15:34:31 and 15:47:51 to 15:48:15: where does PFJAX's gradient come from, and does PF plus Nelder-Mead represent the package's own method? | Distinguish original PF code, a derivative correction we added, and an external optimizer. Describe unsupported or untested package methods explicitly. | [Method descriptions](package_calls.md) |
| Giles, 15:52:23 to 15:52:30 and 15:56:28 to 15:56:46: why was IF2 absent from the CPU packages, and how far should the extensions go? | Check each package's interfaces before deciding what to add. Do not implement every method merely to fill a comparison table. | [particles](../packages/particles/README.md) and [pypfilt](../packages/pypfilt/README.md) |
| Ed, 15:54:22 to 15:55:17 and 15:57:49 to 15:58:16; Giles, 15:56:49 to 15:57:33: keep code and a short report together, describe the model and results, and offer a PR if authors are interested. | Provide a report linked to the exact code and results. Treat preliminary findings as an invitation to discuss the implementation. | [Dacca report](../reports/dacca_comparison.md), code directories and reproduction instructions |

The discussion of Dynamax at 15:17:33 to 15:19:52 raised the same question about package use: the experiment supplied its own PF. The later decision to exclude Dynamax from the main comparison came from Kunyang on 31 October. The earlier records remain available as history.

## Decisions made after the meeting

The meeting proposed a separate repository called `comparisons`, but did not settle forks, submodules or hosting. Kunyang subsequently requested this local repository and English review documents. We currently keep copies of original and modified code with complete diffs. A future upstream contribution will still need the package's development checkout, contribution process and maintainer review. No remote repository or pull request has been created.

The A/B/C protocol follows the within-package comparison discussed in the meeting. The specific choices of six starts, two repetitions, J=5000, 175 IF2 passes, 400 Adam updates, paired random streams and independent scoring were fixed afterward. They are experiment settings, not quotations or instructions from Ed or Giles.

Likewise, the meeting did not prescribe a shared optimizer implementation, a single function name, or a requirement that the experiment call one combined `fit` function. Our Cuthbert driver calls `iterated_filtering.fit` and `dmop.log_likelihood` with an external Optax loop. It does not call the combined `ifad.fit`. The code description must state this directly.

Fixed-seed Nelder-Mead was a suggestion during the discussion at 15:50:42 to 15:51:50, not a settled requirement. PF plus SciPy Nelder-Mead is an external optimization procedure in our experiments, not evidence of a package-provided MLE method. The current work does not exhaust every package's estimation methods.

## Work still needed

Cuthbert is the current priority. Its new code reuses the original PF update functions, but also adds `scan_final`, a loop over observation times that returns only the final filter state. That addition must remain visible in the diff and explanation; the code does not simply call the complete original `cuthbert.filter` function.

The included Cuthbert GPU checks are unfinished, and there are no new A/B/C results. PFJAX, PyDPF and TFP also have unresolved GPU checks. Their earlier optimization results used different code. The new particles fits and scores are available, with three low-precision scores retained. See [the dated status](../reports/status.md) for the tests, failures and remaining experiments.

The repository contains the code and report structure discussed in the meeting. Conclusions about correctness and optimization still depend on the corresponding tests and experiments. Earlier Chinese planning notes are preserved in [meeting_notes/](meeting_notes/); later user decisions and dated experiment records take precedence.
