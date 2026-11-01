# PFJAX cases5–8 and TFP review preparation

Four more PFJAX allocations completed and were collected once. The saved-data audit now covers cases0–8: 27 of 36 endpoints, 1,224 original independent PF values and nine complete 175-pass IF2 prefixes. Twenty-six Adam fits completed all 400 updates. Case8 A failed after 116 accepted updates; its last valid state and rejected proposal remain in the results. B/C initial parameters, zero Adam states, learning rates and training keys match in all nine cases.

| Case | A log mean likelihood | B | C | Allocation GPU seconds |
|---|---:|---:|---:|---:|
| 5 | -3847.215975 | -3748.546967 | -3744.640453 | 1696 |
| 6 | -4247.885292 | -3748.030237 | -3745.228729 | 1700 |
| 7 | -3814.511080 | -3780.893273 | -3774.132663 | 1708 |
| 8 | -24867.998047 (failed; floor) | -3746.878622 | -3744.171275 | 1313 |

Case6 A reached R=288 under the original adaptive scoring rule and passed with MCSE 0.275928 and likelihood ESS 12.603357. The other new scores stopped at R=36. All 27 scores satisfy the numerical MCSE/ESS thresholds, but case8 A is separately flagged as density-floor saturated. That result is not a satisfactory fit or informative Monte Carlo precision result. Every raw value and both flags are retained. No fit or score was repeated.

## Case8 A failure

The zero-based update116 proposal has finite loss 41.4467163, but all 23 gradient components are NaN. Its 23 estimation coordinates and 46 Adam moment components are also NaN. The proposed natural parameters retain five fixed finite entries and 23 NaNs. The audit verifies the unchanged last valid parameters and optimizer state, the rejected proposal, iteration, learning rate and training key. These records establish the nonfinite values at the rejected update; they do not identify the underlying operation that caused them.

The independent score is −24867.998047, with 36 equal raw PF values, MCSE 0 and likelihood ESS 36. The density-floor flag remains true. The failed A fitting kernel used 154.550 seconds; this early stop is not a successful speed improvement. Its allocation still used 1,313 GPU seconds including B, C, preparation and scoring. There is no automatic retry or replacement endpoint.

## Scope and cost

The nine audited cases cover four starts with both replicates and the first replicate of start4. C has the largest estimated likelihood in these nine observed cases, while B remains below A in case1. These partial results do not establish a general optimizer ranking. [All retained endpoints and times](../results/pfjax/endpoints.csv) include failures and floor scores. IF2 retains its final 5000×23 swarm and 175-pass acceptance/likelihood trace; intermediate particle swarms were not all saved.

Cases5–7 each verify 113 frozen sources, 394 references, 41 transferred files and 30 terminal hashes. Case8 has one additional transferred and terminal file for its rejected proposal. The new cases contribute 684 raw PF values and 6,417 GPU seconds. Completed formal allocations total 15,327 GPU seconds. Adding 841 prior-check seconds and three remaining 3,360-second limits gives 26,248 of the original 42,480 seconds. Shared IF2 and compilation count once physically; B/C method costs each include the full prefix. Resource seconds are not concurrent wall time.

At 11:57 UTC, PFJAX cases9–11 were running. PyDPF shared IF2 job65170880 was running on gl1503, and TFP prefix65171217 was pending AssocGrpGRES. No new job was submitted and none of the 4,264 protected frozen source files changed.

## TFP preparation, not completed GPU results

The new prefix auditor checks original starts, each saved final cloud, accepted passes, failure records, parameter means, canonical zero Adam state, source/transfer/terminal hashes and actual allocation cost. Its local controls cover a complete prefix, a rejected IF2 proposal and a warmup exception; altered means, optimizer moments and rejected keys are refused. Its zero-state layout also matches 95 arrays from an existing saved Keras state. These are explicitly synthetic prefix fixtures. Successful-call keys are reconstructed from frozen code and iteration/member records; they were not separately saved at runtime. The actual TFP prefix has not finished or been audited. The first local syntax failure is preserved with the corrected preparation.

The [separate gradient-driver diff](../experiments/dacca/tfp/gradient_driver_preparation_20261101.diff) integrates the [earlier proposal-recording return value](../experiments/dacca/tfp/gradient_recording_preparation_20261101.diff). CPU synthetic-gradient runs of the original and prepared B/C drivers produce identical endpoints and initial/final optimizer states. Four rows exercise NaN loss, NaN gradient, optimizer overflow and natural-parameter overflow; eight other rows complete 400 synthetic updates. A whole-call execution exception retains the last valid state and exception without inventing an unavailable proposal. A NumPy audit recomputes the saved acceptance flags and compares both drivers. No PF or IF2 was called.

The first exception-control attempt raised a Python error during tracing, before the intended execution guard. Its script, log and records are preserved. The corrected control injects a graph-execution failure; completed mixed controls were reused without rerunning them. The synthetic prefix and learning-rate override are test fixtures only. The prepared directory has no job, frozen manifest or approval, and is not deployed. A separate revision must verify and reuse the actual completed physical prefix, retain the original scientific settings and pass its submission guard before A/B/C can start.

## Archive

All new raw results, failed proposals, CPU preparation records, failed local attempts and source dependencies are archived. Every member hash was checked after extraction. Four PFJAX case audits and the TFP saved-driver audit reproduce byte for byte; the synthetic prefix audit reproduces the same structured result. No filter or fit was rerun during archive verification. These checks establish saved-record integrity, not mathematical derivative correctness or full GPU-suite success.

Archive `artifacts/validation/pfjax_cases5_8_and_tfp_preparation_20261101T1156.tar.gz` contains 7,028 files; SHA256 `b629ab3fe317f47ec9bfed1f53a6c7326c5a0bffa6e13edb81495d8f305e515e`. [Structured results](../results/pfjax/cases5_8_and_tfp_preparation_20261101.json) · [Archive catalog](../archive/catalog.json).
