# Formal supervised maze comparison: active new scope

The user explicitly chose native DPF / native StopGradientDPF / added alpha=.97,
with independent held-out localization error. The screenshot is a presentation
example, not a promised ranking. No IF2/MLE/IFAD or new PyPOMP implementation.
All old Dacca, pilot, profiling and network optimization scopes stay closed.
Never rerun old jobs or restore old monitors. List visible agents first; no new
agents were spawned. Use actual jobs/status before actions.

New root: `work/pydpf-maze-formal-20261102`. The complete frozen protocol and
README specify 6 initializations x 2 training repetitions x 3 methods, 100 epochs,
T100/J100/B64 with B4 tail, original model/loss/AdamW, 1,500 updates per fit. Fixed
900/400/700 split. New data preparation uses train-only image normalization;
the earlier global-normalized pilot cache must NOT substitute for this input.
Validation and test both use original ordinary PF/SystematicResampler. Validation
chooses an epoch; only then test 700 trajectories on three PF streams. Keep
failure statuses and prior validation checkpoint testing as explicitly incomplete
fits. Equal updates, not matched compute. Never tune against test results.

New scope cap 108,000 GPU seconds. One bridge <=1,200 seconds plus 36 formal tasks
each <=2,700 seconds reserves <=98,400 seconds. At most three formal tasks run
concurrently. No retries, extra candidates, hidden old-budget reuse or relaxed
numerical thresholds. A failed bridge/cost check is a limitation to preserve and
report; do not automatically submit formal work or alter the protocol.

CPU checks are complete. Three sequential T100/J8 optimizer pairs for each arm
pass exact outputs/state/RNG and all 403,958 gradient values plus exact post-AdamW
parameters/moments, tail and checkpoint restore. Real training driver also passed
2-epoch/4-update small CPU fixtures for all arms; saved-data audits verify
validation selection, split separation and three-repeat RMSE. These are not GPU
results or formal trained models. First CPU harness failed because CPU
optimizer.load_state_dict aliased the checkpoint tensors; common_attempt1.py,
bridge_attempt1.py, all three first raw runs/logs and cpu_attempt1_failure.json
remain. Deep-copying optimizer/scheduler state fixes the harness; no tolerance
change or native model change. Frozen GPU bridge checks input checkpoint hashes
remain unchanged. The previously validated network_graph.py and 20 native/model/
worker files are byte-identical. The new training_graph.py handles B64/B4 and
inference fallback.

70 frozen files, manifest SHA
57960b2249ca35c49b6e4efc523b3a38f10572c4a17d66bb050268bdb29db23c.
Deployment SHA f128d3bd1bce7a65849f9072f010e865afa86eac5e06b61283090eb90a2ab472.
Bridge job **65238998**, only once submitted at18:11:53UTC 2November, A40/4CPU48G,
20-minute limit. Actual18:12:56UTC PENDING Priority/0seconds. No running result
directory was read. Formal array is NOT submitted. Never redeploy bridge.

Remote root:
`/scratch/ionides_root/ionides1/hkunyang/model024_20261006/pydpf_maze_formal_20261102`.
Only existing SSH:
`ssh -S /tmp/greatlakes-codex-login2.sock -o BatchMode=yes -o ConnectTimeout=15 greatlakes`.
Sandbox refusal can use require_escalated for this same authorized operation.
Do not ask for passwords or classify network failure as experiment failure.
Each check run `python3 work/pydpf-maze-formal-20261102/query_once.py` once. Use
Slurm ElapsedRaw, not DST local End-Start. Do not query old jobs.

When bridge is stable terminal and squeue empty, first check
`bridge_collection/collection.json` and `collection_attempt.json`. If neither
exists, run `collect_bridge_once.py` exactly once. An incomplete attempt requires
reading its saved status, not blind recollection. The 1.4GB regenerable data cache
stays remote; its exact hash and training-only statistics/split remain in metadata.
Do not run collector after formal jobs start; it deliberately collects only this
pre-formal stopped root.

After COMPLETED0:0 and collection:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=work/competitor-estimation/pydpf/local-deps work/panel-cute-cpu/bin/python work/pydpf-maze-formal-20261102/audit_saved.py --root work/pydpf-maze-formal-20261102/bridge_collection/raw --receipt work/pydpf-maze-formal-20261102/bridge_collection/collection.json --source work/pydpf-maze-formal-20261102 --output work/pydpf-maze-formal-20261102/bridge_audit.json
```

Only saved tensors/statistics are read, no model/optimizer/PF. Requires all three
arms GPU chains, exact outputs/loss/model/RNG, finite gradient relativeL2 and
maxabs<=2e-6, exact updated optimizer/parameters, checkpoint restore and same-GPU
original self repeat. Evaluation checks include B4/16/60/64, with original
inference fallback. Cost uses actual warm complete steps including driver copies,
maximum warm tail/evaluation, 35% margin and600seconds overhead. Preserve every
raw/cold cost. If schema/check fails, save original error before inspecting data;
no alteration of frozen benchmark, tolerance or failed evidence.

Only after actual source/data/terminal/gradient/optimizer and cost audit passes,
root reviews `bridge_audit.json` within the user's existing formal authorization
and creates new exclusive `release_approval.json` with fields:
`passed:true`, `manifest_sha256`, `audit_sha256` (SHA of actual bridge_audit.json),
`data_sha256`, `planned_total_cap`, plus UTC and rationale. Then run
`python3 work/pydpf-maze-formal-20261102/deploy_once.py --stage formal` once. It
refuses prior formal submission intent/jobs, uploads only audit/release, verifies
source/data and prior job names, then sbatch train.sbatch array0-35%3 once.
Each task maps case=index//3 and arm=[native_dpf,native_stop,discount097][index%3].
Each launch has an exclusive claim. No duplicate/resume/retry.

While pending/running, quiet unless real failure or user question. Formal
collection/audit aggregation still must be implemented outside frozen source:
collect each stopped array task only once, never tar other running tasks or the
shared data cache; verify original source/transfer/complete terminal lists,
unique claims, 1,500 updates or true failure, initial model and optimizer pairing,
per-call RNG bytes/seeds and epoch orders, all validation/test predictions and
selection, rederived per-trajectory errors/pooled RMSE, all failures and actual
allocation sum. `audit_saved.py --training` already audits successful saved
training records; failed/partial fits need a separate explicit saved partial
audit, not relaxed success rules. Extend external audits for cross-case pairing,
fixed schedules/seeds, all checkpoints and budget before final delivery.

Final English report/Chinese supplement, distribution and paired RMSE plots,
all samples/failures retained, scientific visual checks, full raw/source/failure/
logs/checkpoint archive with hashes and extracted saved audit equality. Do not
claim 1.47x training speed or alpha advantage from prior fixed-parameter timing.
No rerun is needed for report/audit recovery. No email/push/PR/remote, broad Git
commit/reset or browser restart. Preserve all other chats' dirty/untracked work.
comparisons edits append only this scope and update catalog/checksums, verify.py.
The monitor should end only after formal outcome or verified limitation and
recoverable report/archive delivery.
