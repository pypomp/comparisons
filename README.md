# comparisons

Particle filtering and likelihood optimization for the Dacca cholera model. This repository contains changes to Cuthbert, PFJAX, PyDPF, TensorFlow Probability and particles, together with the programs and records used to compare them.

## Reports

| Question | Report |
|---|---|
| Does the modified code pass comparisons with the original package? | [Correctness](reports/correctness.md) |
| How long does an ordinary particle filter take? | [PF runtime](reports/timing.md) |
| Do IF2 initialization and gradient discounting improve optimization? | [Likelihood optimization](reports/optimization.md) |

[Study design](reports/dacca_comparison.md) describes the model and comparison. [Experiment status](reports/status.md) lists completed and unfinished work. The [completed Cuthbert comparison](reports/cuthbert_20261101/report.md) contains all 36 audited endpoints, including two failed A fits and three low-precision A scores. C exceeds B in all twelve observed pairs. The [completed particles comparison](reports/particles_20261101/report.md) includes all 24 estimates, independent scores, figures and a verified reproduction archive, with three low-precision scores retained.

The [completed PFJAX comparison](reports/pfjax_20261101/report.md) includes all 36 endpoints, one failed A fit and its density-floor score, and a verified archive of source, raw records and figures. The [completed TFP comparison](reports/tfp_20261101/report.md) includes all 36 endpoints, 35 complete fits and one failed A fit with a density-floor score, together with the verified full reproduction archive. The [completed PyDPF comparison](reports/pydpf_20261102/report.md) contains 36 complete fits and 1,548 raw PF values, retaining one low-ESS score and one density-floor score. All five new public-interface comparisons have final reports, figures and verified reproduction archives. The separate [maze pilot](reports/pydpf_maze_20261101/report.md) completed its forward checks and stopped at the original time-budget check, with no training or test-RMSE result; its failed allocation and full records are retained.

## Package changes

Each directory contains the changed files, a diff against the recorded original version, and validation programs. These changes are local and have not been submitted upstream.

| Package | Original code used | Our additions |
|---|---|---|
| [Cuthbert](packages/cuthbert/README.md) | PF updates and α=1 derivative correction | IF2, gradient discounting and the observation-time loop |
| [PFJAX](packages/pfjax/README.md) | Full PF and resampling | IF2 and derivative correction, including α=1 |
| [PyDPF](packages/pydpf/README.md) | `ParticleFilter` and `StopGradientDPF` | IF2 and gradient discounting |
| [TFP](packages/tfp/README.md) | SMC and differentiable `particle_filter` | IF2 and gradient discounting |
| [particles](packages/particles/README.md) | `SMC` and `Bootstrap` | IF2; the experiment uses external Nelder-Mead |

The [function table](docs/package_calls.md) gives the calls and external optimizers. Final scores use the fixed original PyPOMP PF. Earlier PyPOMP and pypfilt comparisons are retained with their own protocols. Dynamax is retained only in the historical records.

## Reproduce saved results

Python 3.9 or later and Git are sufficient for source recovery and table regeneration. Run from the repository root:

```bash
make help
make prepare PACKAGE=cuthbert
make tables
make check
```

These commands restore source, rebuild tables from saved scores and verify files. Numerical reruns require the recorded environments and launch paths; see [Reproduction](docs/reproduction.md). Large archives in `artifacts/` are excluded from Git and must be copied separately using [the catalog](archive/catalog.json).

## Files

| Directory | Contents |
|---|---|
| `packages/` | Package changes, diffs and validation programs |
| `experiments/dacca/` | Protocols, fitting programs and independent scoring |
| `results/` | Saved estimates, checks and summary tables |
| `reports/` | Study design, results and dated test reports |
| `scripts/` | Source recovery, table generation and file checks |
| `archive/` | Index and checksums for full sources, logs and earlier records |

[Repository conventions](docs/conventions.md) describe file organization and the examples used. [Source and license notices](THIRD_PARTY.md) identify the original packages. To browse locally, run `make browse` and open `http://127.0.0.1:8765/`.
