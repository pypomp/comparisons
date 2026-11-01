# Comparison with existing PyPOMP tests

The earlier Dacca test treated an uncalibrated componentwise gradient threshold as a reason to stop all optimization. That interpretation was too strong. We inspected actual test and analysis programs, then applied the existing CuTe gradient comparison arithmetic to every saved Cuthbert pair. This changes how we interpret the evidence; it does not change the old assertions, job states or raw values.

## Programs inspected

| Revision and file | What it checks | What we can use here |
|---|---|---|
| PyPOMP `41430ec524dabec64a847d019ec6b6ba17017f79`, committed by `hetankevin`; `tests/test_cute/test_cute_backend.py:125–177` | CuTe/XLA simulation derivatives with common random draws, likelihoods from 200 PF repetitions, and three optimization steps | Separate forward calculations, derivative discrepancies and short optimization behavior; measure differences relative to their scale |
| quant `526402cd49a84ffd21b4bfea385b33eefcf19f03`, committed by `hetankevin`; `tests/checks.py` | Log-likelihood sample mean differences with standard errors and distribution checks; small smoke runs receive no scientific verdict | Account for Monte Carlo error when comparing likelihood samples; distinguish a smoke test from an accuracy result |
| pyMeasles `747ce9a43cb56ebfc9801e3ce4f0f6fafa5a0077`, committed by Aaron Abkemeier; `lab/model_001b/search_002/compare_arms.py` | PF evaluations after fitting, repeated-run summaries, per-unit contrasts and fitting time | Judge optimization by subsequent PF evaluation and report repetition counts and actual time |

The files were read from these Git revisions and checked against the local copies. These are shared repositories: the author of a revision is not necessarily the author of every line. In particular, Aaron's inspected program is an optimization analysis, not a derivative test. It does not establish an Aaron-specific gradient tolerance. Kevin's meeting request at 15:23:10–15:23:41 on 29 October separates correctness, runtime and likelihood performance; it does not specify our old numerical thresholds.

The CuTe test compares derivatives of a weighted simulation output over one observation interval, for states and parameters. Its likelihood test compares means with four estimated standard errors. Its three-step training comparison uses `rtol=2e-3, atol=2e-3`. These are different checks with different purposes. None is a universal Dacca threshold. We did not pool likelihood values from different Dacca starting parameters into the quant mean-difference test, and we did not replace the fixed independent `log(mean likelihood)` scoring protocol with a different summary.

## Recalculation from saved Cuthbert GPU records

For original gradient `a` and public-interface gradient `b`, the CuTe test calculates

```python
rel = abs(a - b) / (abs(a) + 1e-3 * max(abs(a)) + 1e-12)
assert median(rel) < 1e-4
assert mean(rel > 1e-2) < 0.01
```

We reconstructed the saved float32 arrays and executed the original test's NumPy statements and assertions without changing them. The result agrees with an explicit recalculation for all 24 pairs. No PF, gradient or optimization was rerun.

| Saved T=600 comparison | J=32 | J=5000 |
|---|---:|---:|
| Parameter/key pairs | 12 | 12 |
| Exactly equal forward log likelihoods | 12/12 | 12/12 |
| Original Dacca elementwise rule satisfied | 11/12 | 4/12 |
| CuTe formula and thresholds satisfied retrospectively | 12/12 | 11/12 |
| Largest median normalized gradient difference | 0.0000655% | 0.0175491% |
| Largest individual normalized gradient difference | 0.0020691% | 0.1148170% |

J=5000 case 8 still exceeds the CuTe median threshold of 0.01%. No coordinate exceeds its 1% normalized-difference cutoff. We retain this exception. The formula includes a scale term in its denominator, so these percentages differ from ordinary componentwise relative error. The [calculation record](../results/cuthbert/reference_tests.json) contains every case, original outcome, input hash and reference revision.

This is a retrospective comparison of rules, not a successful reproduction of the full CuTe test on Dacca. It shows why “eight failed pairs” was an incomplete description of the discrepancy. It does not prove that 400 Adam updates would be identical or determine the source of the GPU variation. The existing repeated-original/public Cuthbert diagnostic addresses repeatability. The [PyDPF control](pydpf_gpu_20261101.md) already showed a difference within unmodified original code that disappeared in deterministic mode on its small test model; that finding cannot be transferred automatically to Cuthbert.

JAX documents that compilation may change floating-point arithmetic in its [numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs). Its [`stop_gradient` documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html) also distinguishes the returned value from automatic derivatives. These support separate forward and derivative checks; neither establishes the cause of our discrepancy.

## Decision for the comparison

Following the user's explicit acceptance of forward-likelihood agreement, the empirical A/B/C comparison can proceed while numerical derivative differences remain disclosed. The acceptance record requires the passed public-interface and full-length PF checks, exact forward agreement at all recorded starts, finite gradients, source checks and the existing failure guards. It does not require that the retrospective CuTe calculation pass, or claim that the full GPU upstream suite passed.

The new submission wrapper uses the unchanged Cuthbert v3 package, fitting program and independent scorer. It starts with case 0 in the original order, retaining all three complete arms. This is the first of the planned twelve cases, not an extra pilot fit. Subsequent cases follow after actual cost review, without selecting cases by their scores. All 36 endpoints remain in scope.

The per-case allocation is 57 minutes. The unchanged conservative estimate is 3419.469 seconds, including 50% additional kernel time, three times measured compilation and 300 seconds overhead. It is only 0.531 seconds below the allocation; the former additional 240-second reserve is explicitly removed. Staging the first complete case allows actual time to inform the remaining allocations. Reserving all twelve cases plus the 879 seconds already used and the existing seven-minute diagnosis totals 42339 seconds, below the original 42480-second budget. An interrupted fit must be audited, not replayed or silently shortened.

Full reference files, raw comparison inputs, the recalculation program and the new submission record are retained in the dated archive listed in [the catalog](../archive/catalog.json). The [status page](status.md) records submission and completion separately from this numerical review.

The first complete case was submitted as job 65158659 at 03:21:48 UTC on 1 November 2026. See the [job record](../results/cuthbert/submission_v4.json). Two login-node import deadlines occurred before submission; the subsequent fit and scorer imports both completed. Diagnostic stacks show waiting during JAX CUDA-library loading and SciPy import. These preflight records remain archived; no fitting or scoring was repeated.
