# Likelihood optimization

The current comparison has twelve runs per method: six common starting vectors, each repeated twice. A starts at the original parameters; B adds IF2; C uses the same IF2 result and our discounted gradient. The four GPU packages use external Adam. particles has a separate CPU IF2/Nelder-Mead comparison. See the [design](dacca_comparison.md) and [original-function table](../docs/package_calls.md).

## Independent likelihood results

The medians include every finite score, including scores from failed fits, scores with low precision and scores at the density floor. All scores use the same original PyPOMP PF and log(mean likelihood). A failed fit can also have a floor score or low precision.

| Package | A median | B median | C median | Failed fits | Floor scores | Low precision |
|---|---:|---:|---:|---:|---:|---:|
| Cuthbert | -3814.966803 | -3747.557538 | -3744.501336 | 2 | 0 | 3 |
| PFJAX native A/B | -3859.159879 | -3763.926678 | -3745.089672 | 0 | 0 | 0 |
| TFP | -3809.127311 | -3746.926753 | -3744.294332 | 1 | 1 | 0 |
| PyDPF | -3806.946514 | -3746.804253 | -3744.445235 | 0 | 1 | 1 |
| particles (CPU) | -4169.070668 | -3784.099127 | — | 0 | 0 | 3 |

PFJAX A/B use automatic differentiation through the package's basic PF, giving a biased gradient. We compare these results with our earlier α=1 correction below. B and C differ in gradient calculation as well as α, so their difference cannot be attributed to discounting alone.

| Comparison | Positive / pairs | Median difference |
|---|---:|---:|
| Cuthbert C-B | 12/12 | 2.351684 |
| PFJAX C−native B | 12/12 | 18.879960 |
| TFP C-B | 12/12 | 2.391187 |
| PyDPF C-B | 12/12 | 2.240650 |
| PyDPF C-B (precise, nonfloor) | 11/11 | 2.031365 |
| particles (CPU) B-A | 12/12 | 377.263414 |

These differences describe six starts with two repetitions each. They do not establish performance at other starts or models, or at equal computational cost. The PyDPF pair with a floor score remains in the summary of all pairs and is shown separately in the plot. A floor score can have MCSE zero and ESS=36 because all evaluations return the same floor value, so these diagnostics do not establish useful scoring precision.

The [figures](figures.md) show half-violin plots with boxes and individual points, the full likelihood range, and paired C−B differences. All twelve points enter the distributions and summaries, including points outside the zoomed view. The [summary CSVs](../results/README.md) identify the results used.

## PFJAX native baseline and the earlier extension

| PFJAX method | Median likelihood, all twelve fits | Complete fits | Median seconds for 400 updates | Median seconds including IF2 where used |
|---|---:|---:|---:|---:|
| Native A | -3859.159879 | 12/12 | 523.299 | 523.299 |
| Added alpha=1 A | -3793.495698 | 11/12 | 529.630 | 529.630 |
| Native B | -3763.926678 | 12/12 | 523.158 | 563.246 |
| Added alpha=1 B | -3748.631853 | 12/12 | 529.519 | 569.591 |
| C: IFAD | -3745.089672 | 12/12 | 529.660 | 569.732 |

Likelihood medians include all twelve final estimates. Timing medians include only fits that completed all updates. With our α=1 correction, A case 8 stopped after 116 updates and has a floor score. Its elapsed time is retained in the records but excluded from the 400-update median. The comparison reuses C, with no additional IF2 or C evaluation. The small timing differences were measured in different A40 allocations, without a timing calibration that would establish a speed advantage. [All sixty comparison rows](../results/pfjax_native/native_and_extension_20261102.csv).

## Fitting time

[The full timing table](../results/overview_20261102/fit_times.csv) reports median time per fit for Cuthbert and PFJAX, and time for a batch of twelve fits for TFP and PyDPF. B and C times each include IF2. Total resource use counts their shared IF2 run once. Allocation time already includes compilation, I/O and scoring. The later PyDPF callback changes were not used in these fits.

## Retained failures and evaluation limits

- Cuthbert: two A fits stopped early; three A scores remain low precision.
- TFP: A case 6 stopped after 64 accepted updates with finite loss and NaN gradients; its score is at the density floor.
- PyDPF: every fit completed 400 updates. A case 9 remains low ESS after 288 repeats; B case 7 has a floor score despite completing optimization.
- Native PFJAX: all 24 new A/B fits completed, and all scores meet the precision criteria. The failed A fit from our earlier correction remains in the separate comparison.
- particles: three A scores remain low precision; there is no gradient C arm.

The [archived final reports](../archive/README.md) contain the checks of saved results, individual PF values, resource use for each allocation and reproduction files. See [Correctness](correctness.md) for the numerical checks and unresolved derivative questions, and [Runtime](timing.md) for ordinary PF timings.
