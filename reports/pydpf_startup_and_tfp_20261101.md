# PyDPF startup failure and TFP full-Dacca check

PyDPF job 65170442 failed after 181 GPU seconds, before formal IF2 started. The failure was in a small Dacca optimizer-resume test, with T=12, J=32 and two parameter vectors. Resumed and uninterrupted runs had exactly equal final parameters and RNG state. Their finite Adam moments differed by at most 3.725290298461914e-9, so the test's exact-equality assertion failed.

These two saved runs do not establish whether the difference comes from restoration or GPU arithmetic. The allocation's failed status, original source and complete saved test states remain unchanged. No formal endpoint was produced. The assertion occurred before creation of the terminal manifest; the collector instead verified all stopped files before and after transfer. It checked 158 source files and 39 transferred files.

## PyDPF revision

The new `pydpf_formal_v3` revision enables deterministic Torch algorithms only inside the small regression-test process. Exact comparisons are retained. The production driver starts in a separate process with its original default setting; its numerical code is unchanged. [The diff](../experiments/dacca/pydpf/startup_test_changes_v3.diff) shows the test, budget and launch-path changes.

The local regression passed exact optimizer restoration, B/C pairing and preservation of three deliberately rejected proposals. This is a CPU test, not a completed GPU test or formal fit. The GPU regression still runs before fitting and can stop the new allocation.

The earlier 181 GPU seconds are charged to the original 30 GPU-hour formal budget. The new warm allocation is 24,119 seconds; the three planned gradient allocations remain 27,900 seconds each. Together with the failed startup, the maximum is 108,000 seconds. Only the warm phase, job 65170880, has been submitted. The 10:42 UTC query found it pending Priority. Its actual full swarms, 175-pass histories, failures and cost must be reviewed before A/B/C gradient phases are submitted.

The [earlier full-Dacca comparison](pydpf_native_DPF_20261101.md) remains separate: native `StopGradientDPF` and the public interface match in all 48 forward/RNG comparisons, but ordinary native PF and native DPF differ. That discrepancy is unresolved.

## TFP check

TFP job 65170859 was running on gl1503 at 10:42 UTC. It checks T=600, J=5000 and the twelve planned start/replicate slots. It compares the original differentiable PF with the public alpha=1 and alpha=0.97 calls using the same execution mode. The original default nondeterministic setting and checkpointing remain unchanged.

The check retains monthly likelihood factors, parameters, gradients, seeds, graph definitions, full IF2 clouds and isolated Adam proposals. It compares repeated calls, checks finite values, and measures memory and the rates needed for the planned 175 IF2 passes and 400 gradient updates. Gradient differences are recorded separately from exact forward agreement. These calls are validation and cost measurements, not formal optimization endpoints or a new ordinary-PF timing benchmark.

The T=12/J=8 CPU preflight produced nine finite filter/gradient calls and 72 exactly matching forward endpoint pairs, three finite IF2 passes and four finite Adam proposals. Saved-array recalculation passed. CPU timings do not establish GPU feasibility.

The first deployment's remote source verification timed out before `sbatch`. A subsequent read-only reconciliation found no matching job or start record and verified all eight source files and 3,090 references. After the import check completed, exactly one GPU job was submitted. The 30-minute limit plus 1,635 previously used validation seconds totals 3,435 seconds, within the original 3,600-second validation budget. There is no new TFP formal job.

## Records

The [current scheduler snapshot](../results/current_jobs.json) records actual job states. The [PyDPF startup audit](../results/pydpf/startup_failure_20261101.json) and [new revision and TFP submission records](../results/pydpf/startup_revision_and_tfp_20261101.json) contain the measured values and source identifiers. Complete source, failed logs, test states, CPU arrays and audit programs are retained in the dated archive listed in [the catalog](../archive/catalog.json). The archived audits read saved data; they do not rerun fitting or PF scoring.

The archive contains 3,610 files. Its SHA256 is `04d2a973bf846f32fea202bd6f775cca67ad749e9dc494f9e6117738fa86e59f`. All members were verified, and the extracted startup, rejection and TFP CPU audits reproduced their saved JSON byte for byte. No new PFJAX case was complete in the recorded scheduler query.
