# comparisons

Local code and evidence for comparisons of particle-filter implementations and parameter estimation on the original Dacca cholera model.

[Chinese notes](README_CN.md) · [Package calls and additions](docs/package_calls.md) · [Current evidence](reports/status.md) · [Reproduction](docs/reproduction.md) · [Meeting decisions](docs/meeting_decisions.md)

English is the primary language for review documents and this browser. Chinese notes are supplementary; frozen sources and historical evidence retain their recorded language.

This repository follows Ed's proposal to keep the comparison code, package changes and reports together. It has no remote. No material has been sent to package authors or submitted as a pull request.

For a local file and diff browser, run `python3 scripts/browse.py` and open `http://127.0.0.1:8765/`. See [browsing instructions](docs/browsing.md). The [31 October source audit](reports/native_pf_audit_20261031/report.md) distinguishes native PF time loops, native update kernels, and locally added gradient code; [source locations](reports/native_pf_audit_20261031/source_locations.md) link to the inspected calls.

## Packages

| Package | Code and review | Available experiments |
|---|---|---|
| Cuthbert | [v3 source, patch and checks](packages/cuthbert/README.md) | Completed earlier paper comparison; new GPU checks pending at the recorded scheduler check, A/B/C not started |
| PFJAX | [v2 source, patch and checks](packages/pfjax/README.md) | Completed earlier comparison with corrected independent scores; new GPU gradient check unresolved |
| PyDPF | [v1 source, patch and checks](packages/pydpf/README.md) | Completed earlier comparison; new GPU check unresolved |
| TensorFlow Probability | [v1 source, patch and checks](packages/tfp/README.md) | Completed earlier comparison; new GPU check unresolved and full local upstream testing incomplete |
| particles | [v1 source, patch and checks](packages/particles/README.md) | Earlier comparison complete; new 24 endpoints and scores audited, final figures/package pending |
| PyPOMP | [Reference source](packages/pypomp/README.md) | Paper comparator and independent endpoint evaluator |
| pypfilt | [Earlier experiment code](packages/pypfilt/README.md) | Completed earlier native-PF comparison with external Nelder–Mead |

Dynamax is excluded from the main comparison at the user's request on 31 October 2026. Its earlier experiment used a locally written PF with `SSM.fit_sgd`, so it does not answer the comparison of package PF implementations. Its code and results remain [historical records](packages/dynamax/README.md); no further Dynamax work is planned.

## Experiments

- [Earlier completed Dacca results (Chinese historical report)](reports/legacy_dacca/report_CN.txt): the unchanged historical archive has 968 endpoint records, including the now-excluded Dynamax experiment. Each result retains the code that actually produced it. This is not the current comparison scope or evidence validating later public interfaces.
- [New public-interface comparison](experiments/public_interfaces/README.md): correctness, native PF timing and optimization are separate questions. The four GPU comparisons have no optimization endpoints in this snapshot. The new particles comparison has 24 audited endpoints.
- [Earlier Cuthbert probes](experiments/history/README.md): retained separately, including earlier source revisions and superseded evidence.
- [Historical all-point figures](reports/legacy_dacca/figures_all/) and [historical 1.5-IQR display figures](reports/legacy_dacca/figures_iqr/): these unchanged earlier figures include Dynamax. They are retained as history, not current main-comparison figures. Future main tables, figures and conclusions exclude Dynamax.

The original Dacca benchmark uses 600 monthly observations and 20 Euler steps per month with Gaussian process noise. Particle counts, starts, iteration counts and budgets belong to each recorded protocol. CPU and GPU costs, older J=1024 probes and later J=5000 comparisons must remain distinguishable.

## Files

```text
packages/      reviewed candidate sources, corresponding upstream files, patches,
               validation programs and records, and exact function descriptions
experiments/   frozen experiment programs, protocols, job records, audits and results
reports/       completed reports/figures and a dated account of unresolved work
artifacts/     actual local reproduction archives and raw records (not Git-tracked)
manifests/     SHA256, original locations, archive inventory and review records
scripts/       offline verification, safe extraction and the one-time import program
```

All imported files are actual copies, not links to the original workspace. `manifests/import.json` records their original paths and checksums. Large archives are physically present under `artifacts/` and excluded from Git. A Git clone alone would therefore need those archives separately; [archives.json](manifests/archives.json) identifies the exact files. No remote storage is assumed.

## Verify locally

Python's standard library and Git are enough for repository verification:

```bash
python3 scripts/verify.py
python3 scripts/verify.py --archives
```

This checks copied files, applies each of the five patches to its preserved upstream source in a temporary directory, compares every resulting file to the candidate, checks result counts and the fixed scorer, and optionally reads all archive members. It does not run a PF, fit parameters, submit jobs or establish GPU correctness.

Historical experiment scripts are preserved byte for byte, including their original paths. Re-running those numerical programs requires their recorded environments and path setup. See [reproduction instructions](docs/reproduction.md); importing this repository does not make every historical runner portable automatically.

Third-party license and copyright notices remain with their sources or archives. See [THIRD_PARTY.md](THIRD_PARTY.md). No new project-wide license is assigned here.
