# comparisons

Code and experiments comparing particle filters and likelihood optimization for the Dacca cholera model. We use each package's particle filter and document the changes needed for IF2 and IFAD.

Start with the [Dacca report](reports/dacca_comparison.md) for the model, methods, available results and unfinished work. The [function table](docs/package_calls.md) shows which functions come from each package and which we wrote. [Meeting notes](docs/meeting_decisions.md) explain the questions raised by Ed, Giles and Kevin on 29 October 2026.

## Code

| Package | What the comparison uses | Changes and tests |
|---|---|---|
| Cuthbert | Original PF update functions and ADPF derivative correction; added IF2, discounting and a loop over observation times | [Code](packages/cuthbert/README.md) · [Diff](packages/cuthbert/review/full.patch) |
| PFJAX | Original PF; added IF2 and resampling derivative correction, including at α=1 | [Code](packages/pfjax/README.md) · [Diff](packages/pfjax/review/full.patch) |
| PyDPF | Original `ParticleFilter` and `StopGradientDPF`; added IF2 and gradient discounting | [Code](packages/pydpf/README.md) · [Diff](packages/pydpf/review/full.patch) |
| TensorFlow Probability | Original SMC and differentiable PF; added IF2 and gradient discounting | [Code](packages/tfp/README.md) · [Diff](packages/tfp/review/full.patch) |
| particles | Original `SMC` and `Bootstrap`; added IF2, followed by external SciPy Nelder-Mead | [Code](packages/particles/README.md) · [Diff](packages/particles/review/full.patch) |
| PyPOMP | Package IF2 and training functions in the earlier comparison; fixed PF for independent evaluation | [Code and results](packages/pypomp/README.md) |
| pypfilt | Package PF with external SciPy Nelder-Mead in the earlier comparison | [Code and results](packages/pypfilt/README.md) |

The first five packages have local modifications. Each directory contains the original version, modified version, complete diff and test records. These changes have not been submitted to the package authors. The [source review](reports/native_pf_audit_20261031/report.md) explains how each filter is called and links to the code.

## Results

The [results and tests](reports/status.md) page separates correctness, ordinary PF runtime and optimization results. It records the date of the included files; it is not a live job monitor.

- The new Cuthbert, PFJAX, PyDPF and TFP comparisons have no optimization results in the included records. GPU validation is still required.
- The new particles comparison has 24 fitted parameter vectors and independent scores. Three scores remain below the required Monte Carlo precision. Its final figures and reproducibility archive are pending.
- [Earlier results](reports/legacy_dacca/report_CN.txt) used different experiment code. They have 968 parameter estimates and 37,584 individual PF evaluations, including 36 Dynamax estimates retained as history. [Figures with all points](reports/legacy_dacca/figures_all/) and [figures hiding points beyond 1.5×IQR](reports/legacy_dacca/figures_iqr/) belong to those earlier experiments.

Dynamax is excluded from the current comparison because that experiment used a PF we wrote ourselves. Its [earlier code and results](packages/dynamax/README.md) remain available. [Earlier Cuthbert experiments](experiments/history/README.md) are stored separately.

## Reproduce or inspect the work

See [reproduction instructions](docs/reproduction.md) for source versions, environments, raw records and commands. The [experiment directory](experiments/public_interfaces/README.md) contains the new A/B/C protocol.

```bash
python3 scripts/verify.py
python3 scripts/verify.py --archives
```

These commands check file hashes, apply the five diffs to copies of the original sources, compare the resulting code and check result counts. The second command also reads the archives. They do not run numerical experiments.

| Directory | Contents |
|---|---|
| `packages/` | Original and modified package code, diffs and tests |
| `experiments/` | Experiment programs, settings, job records and results |
| `reports/` | Reports, figures and source reviews |
| `artifacts/` | Local archives of source code and raw results, excluded from Git |
| `manifests/` | File hashes, source versions and archive locations |
| `scripts/` | File verification, archive extraction and local browser |

The files are copies, not links to the original workspace. [Import records](manifests/import.json) give their sources and hashes. Large archives listed in [archives.json](manifests/archives.json) must be copied separately from Git. Some experiment programs retain their original absolute paths; the reproduction instructions describe this limitation.

Run `python3 scripts/browse.py` to inspect the repository at `http://127.0.0.1:8765/`. See [browser instructions](docs/browsing.md). This repository is local and has no Git remote. English is used for review documents; [Chinese notes](README_CN.md) and original records are also available. Existing license notices are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
