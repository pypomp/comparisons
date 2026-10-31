pypfilt formal_v1 — frozen 12-task CPU protocol; submission requires parent review

This directory contains a new, separately versioned driver. The original
preflight source and its frozen manifest remain unchanged.

Protocol: original Dacca, T=600, 20 Euler steps/month, Gaussian process noise,
23 free coordinates, J=5000. Six common paper starts, two independent repetitions
per start. Each fit uses 1800 seconds, fresh independent PF random streams,
SciPy Nelder-Mead default simplex, xatol=.01, fatol=.0001, adaptive=False.
This is pypfilt PF plus our external optimizer, not native pypfilt MLE/IFAD.

Budget and endpoint:
- Every actual objective call, optimizer computation and in-run checkpoint is
  charged to fitting. A PF call already started may overrun. It is fully charged
  but its candidate is excluded if it finishes after the deadline.
- Endpoint is the best finite training evaluation completed before the deadline.
  This historical best is part of the optimizer wrapper. Independent scoring
  never changes, selects or rejects that endpoint.
- Initialization and tiny T2/J32 PF warmup are preparation. Independent scoring
  and JAX compilation are separately timed. Artifact finalization and recovery
  replay are preparation/reporting overhead. Slurm allocation captures all costs.

Independent scoring uses the original pinned PyPOMP PF, J=5000. Start at R=36,
then append 36 independent repetitions while MCSE>1 or likelihood ESS<10,
up to R=288. Save all raw values and every batch's seed, compilation/execution
and wall time. Low precision at the cap is retained as inconclusive. Endpoint
hash is checked before and after scoring. Training scores are not final results.

Recovery:
- Each objective and score batch is saved atomically; completed fits are skipped.
- SIGTERM/SIGINT stores accumulated fit cost. Resume replays saved objective
  values through the same pinned SciPy version, without rerunning those PFs.
  The first new objective resumes the charged clock. Replay is separately timed.
- An unexpected hard kill leaves state=running. The driver refuses to silently
  restart it. Reconcile terminal Slurm accounting with reconcile_interruption.py
  and include lost work in charged-fit-seconds. Do not count a killed PF as free.
- Resume scoring from completed batches; there is no repeated endpoint selection.

Allocation and independent tasks:
- SLURM array 0-11%3; task_id=start_index*2+replicate. Every task reads the
  same frozen protocol and writes only results/tasks/{task_id}. The initial
  parameter vectors are unchanged paper starts; repetition streams differ.
- Each task receives 1 CPU, 8 GB, maximum 50 minutes. All BLAS/OpenMP pools
  are pinned to one thread; GPUs are hidden. Twelve allocations total at most
  10 CPU-hours. Run at most three tasks concurrently. No GPU pypfilt port is
  claimed. The original preflight used 892300 KB maximum RSS.
- Temporary observation files, TMPDIR and JAX cache are isolated per task.
- The task fits its endpoint first, then scores it. It stops launching scoring
  batches near the allocation limit and leaves pending results. A resumed task
  must pass already-used task allocation as --prior-allocation-seconds. There is
  no automatic resubmission. Parent reconciles overall allocation before retry.
- Preflight pypfilt PF T600/J5000 took about 9.7–10.1 seconds per call; the
  26-evaluation NM probe took 266.2 seconds. These are measurements, not claims
  of convergence or of superiority over another optimizer.
- Original PyPOMP scoring R32 took 61.8 seconds on the old 4-CPU allocation.
  That is not a measured one-CPU time; current single-CPU scoring may be slower.
  Pending or imprecise scores remain explicit, never silently discarded.
- CPU pypfilt and GPU JAX IF2/IFAD costs are reported separately. The experiment
  does not support a same-hardware speed comparison.

Necessary tests live in check_driver.py and checks/verification.json:
late high-score exclusion, complete cost accounting, checkpoint replay, endpoint
reuse, independent-score recovery, precision stopping and the R288 cap.
smoke_real/ contains actual native pypfilt PF and original-PyPOMP scoring at
T24/J64, one starting point, a .5-second fitting budget. It is an execution
check only. No result here establishes a full Dacca optimization advantage.

Before submission: verify manifest.json, review the exact frozen array command,
and obtain the parent dispatch confirmation. Preserve the old preflight source,
job and manifest history. Only protocol.json defines this formal run; the older
draft and small smoke protocols are not formal experiment settings.
