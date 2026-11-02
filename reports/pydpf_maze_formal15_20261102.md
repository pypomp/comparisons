# Maze study: 6 of fifteen fits audited

This dated snapshot contains all three methods for paired initialization seeds 0, 10. Every included fit completed 100 epochs and 1500 AdamW updates, without a recorded failure or rejected update. The remaining 9 outcomes are outstanding. This is not the final five-initialization comparison.

| Seed | Method | Test RMSE | Selected epoch (1–100) | Actual A40 minutes |
|---:|---|---:|---:|---:|
| 0 | Native DPF | 309.509517 | 40 | 21.75 |
| 0 | Native StopGradientDPF | 373.337060 | 74 | 22.15 |
| 0 | Added discounted gradient, alpha=0.97 | 401.092878 | 71 | 26.20 |
| 10 | Native DPF | 279.630268 | 54 | 26.23 |
| 10 | Native StopGradientDPF | 323.229363 | 70 | 22.05 |
| 10 | Added discounted gradient, alpha=0.97 | 288.907544 | 41 | 26.08 |

Lower test RMSE is better. Checkpoints were chosen by the earliest minimum validation MSE, then evaluated on all 700 test trajectories using three independent original ordinary-PF streams. The repeated PF streams share the model and trajectories and are not independent training repetitions. No checkpoint, hyperparameter, extra fit or retry was selected using test error. All methods and seeds remain in the report, regardless of their ordering.

The saved pair audit verifies common initial model, empty AdamW, scheduler and RNG; all 100 training orders and 1500 training-call seeds; and identical validation/test IDs, truths and seed streams. Different methods' gradients and evolved parameters need not match. Each full task audit also checks exact source/transfer/terminal/claim/guard records, all epoch checkpoint model/AdamW/scheduler states and the raw prediction/error/RMSE calculations. Successful full per-step gradients and proposals were not all saved. Hashes cannot reconstruct missing tensors.

The completed training allocations sum to 8668 GPU seconds (2.407778 hours). The completed precheck costs 266 seconds once. Kernel, step, evaluation and I/O timings are components of allocation cost and are not added again. The original requested limits remain 1980/2040/2040 seconds per method; no budget or configuration changed. Equal epochs are not equal computation, and different-node timings do not constitute an isolated code-speed comparison.

The original author model, supervised loss and AdamW settings remain unchanged, with T100/J100/B64 and B4 training tails. Image normalization uses only the training images of the fixed 900/400/700 split, unlike the global normalization in the historical pilot. Native DPF and StopGradientDPF retain their original filters; alpha=.97 is an added discounted-gradient method. This is supervised localization, not MLE, IF2 or IFAD.

All checkpoints, raw validation/test predictions, RNG records and failures remain recoverable. Final figures and the complete fifteen-outcome audit await every authorized outcome. No smooth population distribution or general ordering is inferred from these few initializations.

The full-size A40 precheck passed before the fifteen tasks were submitted once. For each method, original and accelerated B64/B64/B4 updates matched forward/loss/model/RNG, all 403,958 finite gradient components and post-AdamW parameters/moments bitwise. Restored-checkpoint and original self-repeat controls and ordinary-PF evaluations at B4/16/60/64 also passed. These are tested execution agreements, not a mathematical certificate. The original training forecast was 6.235 GPU hours; requested limits total 8.417 training GPU hours plus the separately counted 266-second precheck.

The last scheduler snapshot was 2026-11-02T20:16:00.670203+00:00. It is retained with the saved results; it does not establish any later task state. Previously dated reports and packages remain unchanged in the archive.

[Saved results and archive verification](../results/pydpf_maze/formal15_pairs_20261102T2015.json) · [Active design](../experiments/maze/pydpf_formal15_20261102/README.md) · [Authorization](../results/pydpf_maze/formal15_authorization_20261102.json) · [Archive catalog](../archive/catalog.json).
