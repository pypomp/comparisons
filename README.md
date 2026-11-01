# comparisons

Particle filtering and likelihood optimization for the Dacca cholera model. This repository contains changes to five packages, experiment programs and a short report. The work is local and has not been submitted upstream.

[Report](reports/dacca_comparison.md) · [Functions used](docs/package_calls.md) · [Results and tests](reports/status.md) · [Reproduce](docs/reproduction.md)

| Package | Code used and added | Review |
|---|---|---|
| Cuthbert | Original PF updates and α=1 correction; added IF2, discounting and observation-time loop | [Code](packages/cuthbert/README.md) · [Diff](packages/cuthbert/review/full.patch) |
| PFJAX | Original full PF; added IF2 and gradient correction, including α=1 | [Code](packages/pfjax/README.md) · [Diff](packages/pfjax/review/full.patch) |
| PyDPF | Original `ParticleFilter` and `StopGradientDPF`; added IF2 and discounting | [Code](packages/pydpf/README.md) · [Diff](packages/pydpf/review/full.patch) |
| TFP | Original SMC and differentiable PF; added IF2 and discounting | [Code](packages/tfp/README.md) · [Diff](packages/tfp/review/full.patch) |
| particles | Original `SMC` and `Bootstrap`; added IF2, with external Nelder-Mead | [Code](packages/particles/README.md) · [Diff](packages/particles/review/full.patch) |

The new GPU optimization comparisons are unfinished. The new particles experiment has 24 fitted vectors and independent scores, including three low-precision scores. [Results](results/README.md) separates these from the earlier experiments. Final evaluation uses the fixed original PyPOMP PF.

Each package directory contains only its description, changed files, complete diff, validation programs and version. Full source trees, raw data, failures, test logs and earlier reports are preserved in the [archive](archive/README.md). Dynamax remains in that history and is excluded from the main comparison.

```bash
python3 scripts/prepare.py cuthbert --output build/cuthbert
python3 scripts/reproduce.py --output build/tables
python3 scripts/verify.py --archives
```

These commands reconstruct source code, rebuild tables from saved scores and check all files and archives. They do not fit models or submit jobs. Numerical reruns require the recorded environments and launch paths described in [Reproduction](docs/reproduction.md).

| Directory | Contents |
|---|---|
| `packages/` | Changes to the five packages |
| `experiments/dacca/` | Protocols and experiment programs |
| `results/` | Endpoint records and summary tables |
| `reports/` | Dacca report and dated test status |
| `archive/` | Archive index, file hashes and recovery instructions |
| `artifacts/` | Full archives stored locally, excluded from Git |

Run `python3 scripts/browse.py` to browse at `http://127.0.0.1:8765/`. Sharing Git alone does not include the archives; copy the files in [the catalog](archive/catalog.json) as well. See [source and license notices](THIRD_PARTY.md).
