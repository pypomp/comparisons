# Dacca experiments

The files in these directories are unchanged copies of the saved protocols and principal programs. [Original paths](original_paths.json) locate each full experiment in the archive. They are provided for reading; reconstruct the original layout and environment before a numerical rerun.

| Package | Included revision | State in the saved records |
|---|---|---|
| [Cuthbert](cuthbert/) | `cuthbert_formal_v3` | Prepared; no formal job submitted |
| [PFJAX](pfjax/) | `jax_formal_v2` | Cancelled before fitting after failed GPU check |
| [PyDPF](pydpf/) | `formal_v1` | Cancelled before fitting after failed GPU check |
| [TFP](tfp/) | `formal_v2` | Cancelled before fitting after failed GPU check |
| [particles](particles/) | `formal_v1` | 24 fits and independent scores collected and audited |

All use the original Dacca model: 600 observations, 20 Gaussian Euler steps per month and J=5000. The four GPU protocols compare A: α=1 Adam from the original start; B: α=1 Adam after 175 IF2 passes; C: α=0.97 Adam after the same IF2 result. Each gradient stage has 400 updates. Six starts are repeated twice. B/C share parameters, initial Adam state, learning rates and training random streams. A/B do not have equal total computation. IF2 details differ across packages.

particles compares external Nelder-Mead for 1800 seconds against 900 seconds of IF2 followed by Nelder-Mead to the same total budget. It is a CPU experiment; independent GPU scoring is counted separately.

[score.py](score.py) is the unchanged fixed PyPOMP evaluator. It computes log(mean likelihood) with 36, 72, 144 or 288 repeats, independent training and evaluation streams, and MCSE/ESS checks. Per-experiment copies, models, input arrays, environments, launch scripts and all failures can be restored from the archive.

## Meeting basis

Ed asked for parameter perturbations within the package's framework and an extension of Cuthbert's ADPF (29 October, 15:27:20–15:28:52). Giles asked for a readable diff from the package's original code (15:29:17–15:29:59). Kevin distinguished correctness, running time and likelihood performance (15:22:47–15:23:41). The within-package comparison was discussed at 15:24:45–15:25:17 and 15:53:37–15:53:55. Exact particle counts, iteration counts and paired streams were chosen afterward. The full notes and checked excerpts are [archived](../../archive/README.md).

This repository cleanup changes file organization only. It does not submit, cancel, fit or score any experiment, or alter the live job directories.
