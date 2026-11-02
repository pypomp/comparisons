# Fifteen paired supervised maze fits

The user explicitly authorized fifteen fits on 2 November 2026: three methods,
five paired initialization seeds (0,10,20,30,40), one training repetition per seed.
Start automatically after the existing GPU training-equivalence check passes;
estimate the budget from its actual timings. No additional confirmation is
required for these fifteen fits within this design. The earlier 36-task automatic
submission is permanently superseded and its local launcher remains blocked.

Reuse the unmodified parent numerical source and the same physical train-only
normalized cache. The original source indexes six seeds with two repetitions;
select only its even cases 0,2,4,6,8. The resulting original Slurm task indices are
0,1,2,6,7,8,12,13,14,18,19,20,24,25,26. This is a deliberate selection of fifteen
unrun tasks, not fifteen refits. No model, native filter, alpha correction,
optimizer, training loop, seed arithmetic, data, 100 epochs, T100/J100/B64+B4,
validation-only checkpoint selection or three-stream ordinary-PF test changes.

The active scientific design is this directory's protocol.json plus the parent
frozen numerical source manifest. The parent protocol remains an immutable
historical file; its36-fit total is not the active sample size. Each fit records
its original case and seed. Reports must display the actual five seeds and all
fifteen outcomes, including failures. Prefer individual and paired RMSE points
over smooth population density estimates from five observations.

Budget.json is created only after the actual saved GPU audit passes. Estimate
1,400 full training batches,100 tail batches,600 full+100 B16 validation batches
and30 full+3 B60 test batches per fit. Use measured full-step medians for the
central training forecast and maxima for request sizing. Retain the three timing
samples and shape-specific evaluation samples. Add a stated checkpoint-write
allowance using measured larger raw-record writes. Separately include observed
cold-call excess, graph setup and unattributed bridge time. The latter is charged
conservatively to each fit; it is not an isolated startup measurement. Apply35%
headroom to recurring upper work and round requests to full minutes. Actual
allocation cost, forecast and scheduler maximum remain separate. No30-hour
forecast is assumed. If the resulting per-fit request exceeds the existing
45-minute safety ceiling or the bridge audit fails, preserve and report the
limitation; do not silently change model, length, tolerance or rerun the bridge.

After root reviews the actual audit and budget, prepare_release_once.py freezes
this overlay and creates the exclusive internal release. submit_once.py adds
only this new overlay and audit/release to the stopped verified remote root,
verifies the original cache/source, then submits three sparse five-task arrays,
one per method. Each array permits one active task, giving at most three total.
The per-method time limit is the measured-budget result, not a constant45minutes.
guard.py checks exact allowed indices, method, actual Slurm time limit, review
and all source hashes before the unchanged launcher claims a task. No automatic
retry or additional task is permitted. Receipts/intents are kept per array.

The current check_selection.py controls use no model or GPU. They validate all
fifteen selections, reject the other21old indices and reject metadata tampering.
They do not replace actual GPU numerical validation or actual Slurm guard checks.
Future collection must read only stable stopped tasks, once each, and saved-data
audits must verify pairing, orders, update count or true failure, all validation
and test predictions, source/terminal provenance and actual allocation sums.
