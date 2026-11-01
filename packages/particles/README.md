# particles

Local revision 1, based on the original version recorded in [source details](version.json). The changes have not been submitted upstream.

The new `iterated_filtering` and `_PerturbedBootstrap` give joint state and parameter particles to the original `SMC.run` loop and `Bootstrap` model calls. They add parameter perturbations, cooling and retention of the parameter population between IF2 passes. Existing Python source files are unchanged.

The experiment uses external SciPy Nelder-Mead. Method A runs it for 1,800 seconds. Method B uses IF2 for the first 900 seconds and then Nelder-Mead for the remaining time. This is not a differentiable-PF experiment.


## Inspect and restore

[Changed files](changes/) · [Complete diff](review/full.patch) · [Version](version.json) · [Validation programs](validation/)

From the repository root, restore the original code and apply the diff:

```bash
python3 scripts/prepare.py particles --output build/particles
```

The command checks every resulting file against the saved modified source. Original source trees, environments, test outputs and failures remain in the [archive](../../archive/README.md). [Results and tests](../../reports/status.md) describes the included evidence. [Reproduction instructions](../../docs/reproduction.md) distinguish restoring files from rerunning numerical experiments.
