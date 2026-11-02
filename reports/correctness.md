# Correctness

The tests below compare the original and modified calculations. Agreement is limited to the cases tested, and failed checks remain in the record.

| Package | Observed agreement | Remaining limitation |
|---|---|---|
| Cuthbert | 32 interface tests; identical forward values under corresponding settings in repeated checks of the full model | The earlier gradient criterion also failed when the unchanged original code was evaluated repeatedly. The PF regression tests reached 61/64 passes before the time limit. |
| PFJAX | 324/324 exact original/public likelihood pairs under matched execution | Forward-only and differentiated calculations differ in cases 0 and 11; cause unresolved. Native basic-PF autodiff is biased. |
| PyDPF | 48/48 native-DPF/public monthly likelihood and RNG comparisons agree exactly | The ordinary-PF/native-DPF difference of 558.885 remains unexplained. |
| TFP | 72 exact forward pairs; all 515 original/candidate CPU outcomes and failure identities match | The test suite itself did not pass: each side retained 127 failures, nine timeouts and one abort. |
| particles | Nine CPU PF pairs match likelihood, state, weights and RNG; IF2 checks pass | The pinned source has no upstream pytest suite. |

The optimization comparison proceeded with the user's agreement after the forward calculations were compared under the same execution settings. We checked source hashes, initial and final states, failures, individual PF evaluations and resource use separately. Fits can fail even when the overall job exits successfully.

[Detailed checks and failures](../archive/README.md) remain archived. The later PyDPF callback and maze network tests evaluated individual calls and did not produce new fitted results. We used Kevin's CuTe formula for a retrospective sensitivity check and Aaron's `compare_arms.py` to organize the optimization analysis. These uses do not imply that Kevin or Aaron approved the implementations.

[Code changes](../docs/package_calls.md) · [Runtime](timing.md) · [Optimization](optimization.md).
