# PFJAX case4 and preparation for PyDPF/TFP review

PFJAX case4 completed and passed the saved-record audit. This brings the partial comparison to 15 of 36 endpoints: two starts with both replicates and the first replicate of start2. All five IF2 runs completed 175 passes and all fifteen Adam fits completed 400 updates. All 540 original independent PF values have been checked; every score meets MCSE ≤ 1 and likelihood ESS ≥ 10 at R=36. No fit failure, low-precision score or density-floor result has occurred in these five cases.

| Method | Independent log mean likelihood | MCSE | Likelihood ESS | Logical fitting kernel seconds |
|---|---:|---:|---:|---:|
| B | -3748.398029 | 0.144905 | 20.750353 | 570.237 |
| C | -3745.080133 | 0.141133 | 21.212097 | 570.129 |
| A | -3840.773042 | 0.074264 | 30.175245 | 529.666 |

A uses 400 alpha=1 Adam updates from the original start. B uses 175 public IF2 passes followed by 400 alpha=1 updates; C uses the same IF2 endpoint followed by 400 alpha=.97 updates. B/C initial parameters, zero Adam states, rates and training keys match. Their method costs each include the full shared IF2 prefix. The prefix ran physically once. IF2 retains the final 5000×23 swarm and the 175-pass acceptance/likelihood trace, not every intermediate particle swarm.

The A40 allocation 65170500_4 on gl1504 used 1,739 seconds. The physical fitting kernels used 1,629.956 seconds and compilation used 23.174 seconds; these are parts of the allocation, not additional costs. The audit checks 113 frozen sources, 394 references, 41 transferred files, 30 terminal hashes, the B,C,A execution order, all optimizer states and all 108 independent raw PF scores.

C has the highest estimated likelihood in the five observed cases; B was worse than A in case1. This partial sample does not establish a general ranking. [Earlier cases](pfjax_cases0_3_20261101.md) and [all current endpoints](../results/pfjax/endpoints.csv) remain available. Completed allocations total 8,910 GPU seconds. Including 841 earlier-check seconds and the seven remaining 3,360-second limits gives 33,271 seconds, within the original 42,480-second budget. Summed resource seconds are not concurrent elapsed wall time.

At 11:27 UTC, PFJAX cases5–7 were running and 8–11 were pending the array concurrency limit. PyDPF shared IF2 job 65170880 was running on gl1503; TFP shared-prefix job 65171217 was pending Priority. All these jobs already existed; this update submitted no jobs and changed no frozen source.

## Preparation while the prefixes are unfinished

The PyDPF warm-state audit now reads saved full terminal swarms, parameter histories, pass counts, seeds, generator bytes, hashes, failures and times. It passed against the existing two-row CPU records and rejected an altered hash. The command refuses to create a formal audit when the real collection is absent. The mean check compares the saved float32 mean with double accumulation using a conservative float32 summation-error bound; it does not change any experimental likelihood or gradient tolerance. Generator bytes are checked and retained without replaying a filter. The actual GPU warm phase has not yet been audited.

The [separate TFP recording diff](../experiments/dacca/tfp/gradient_recording_preparation_20261101.diff) adds pre-rollback parameters and optimizer variables to the Adam return value. The existing four outputs and full optimizer state matched the unchanged code exactly in five CPU controls: a finite update, NaN loss, NaN gradient, optimizer overflow and natural-parameter overflow. A NumPy audit independently recomputed the saved finite flags. These tests use synthetic gradients with no PF calls. The change is not deployed and is not yet integrated into a new gradient driver. The active TFP prefix code is unchanged. A prefix-specific audit and a separate, tested gradient revision are still required before submitting A/B/C; the shared IF2 prefix will be reused.

## Records

All 4,264 protected frozen files were rechecked unchanged. The archive contains the original case4 records, source, audit dependencies, existing CPU fixtures and preparation tests. Every member hash was verified after extraction; the PFJAX audit and TFP saved-array audit reproduce byte for byte, and the PyDPF CPU saved-state audit reproduces the same structured result. No filter or fit was rerun during archiving. These integrity checks do not certify mathematical gradients or a full GPU test suite.

Archive: `artifacts/validation/pfjax_case4_and_audit_preparation_20261101T1126.tar.gz` (3,898 files), SHA256 `caaa913271794fb1f0d6f3631d6b60e2fbe6e085e4f35b1400015a30263aa32b`. [Structured records](../results/pfjax/case4_and_preparation_20261101.json) · [Archive catalog](../archive/catalog.json).
