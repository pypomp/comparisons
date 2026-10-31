# Dacca experiments with the added functions

The dated [snapshot](snapshot_20261031/) contains experiment programs, protocols, scheduler records, local checks, recovered GPU failures and the new particles results. [Results and tests](../../reports/status.md) describes the included records. Historical status files are retained verbatim; read the dated summary before interpreting an old job as current.

The numerical settings below were fixed after the [29 October meeting](../../docs/meeting_decisions.md). They implement its proposed comparison of a method, the same method after IF2, and the discounted method. They are not settings prescribed in the meeting.

Four GPU packages: Cuthbert, PFJAX, PyDPF and TFP. For each package there are six fixed starts and two repetitions, with three planned arms:

| Arm | Initialization | Gradient stage |
|---|---|---|
| A | Original start | α=1, Adam, 400 updates |
| B | Added IF2, 175 passes | α=1, Adam, 400 updates |
| C | Exactly the same IF2 endpoint as B | α=0.97, Adam, 400 updates |

B/C use the same parameter vector, zero initial Adam state, learning rates and training random stream. A/B have different total work. IF2 perturbation and cooling details differ between packages, so the protocol does not claim identical IF2 implementations across packages. The IF2 run is computed once. Its time is counted once in the total job cost and included for each method when reporting B and C separately.

Cuthbert v3 is the current priority. Its actual formal driver uses public `iterated_filtering.fit` and `dmop.log_likelihood` with external Optax Adam. It does not call the complete `ifad.fit` function. All three arms explicitly disable checkpointing after the recorded differentiation-path diagnosis. GPU tests and measured memory and runtime checks must pass before fitting. No formal Cuthbert v3 job has been submitted in this snapshot.

For particles, A is 1800 seconds of native PF + external SciPy Nelder-Mead; B spends the first 900 seconds on added IF2 and uses the remaining time for the same optimizer. All 24 endpoints and 1692 independent PF repeats have been audited. Three A scores reached the maximum repeats without satisfying the precision threshold. They are retained.

All formal endpoints use the frozen original PyPOMP PF, J=5000, R=36→72→144→288, log(mean likelihood), MCSE≤1 and likelihood ESS≥10. Failures, low precision and density-floor saturation are retained. Training and scoring seeds are separate. [shared/score.py](snapshot_20261031/shared/score.py) has SHA256 `8aec1ba02982a1ad9bff227b300806eaafa972a1e9fd00b71bef2c5f0ebdc508`.

Existing raw and deployment archives are under [`artifacts/experiments/public_interfaces/`](../../artifacts/experiments/public_interfaces/). The first six particles tasks and the later eighteen tasks plus scores are in separate archives because they were collected at different times. Together they preserve all 24 tasks; no task was rerun for this repository.

The partial TFP upstream-test snapshot copies only completed original/candidate pairs and their corresponding logs. It does not claim that the entire suite passed. Its exact copied count and failures are in [snapshot.json](tfp_upstream_tests_snapshot/snapshot.json). The original test runner was not restarted or modified.
