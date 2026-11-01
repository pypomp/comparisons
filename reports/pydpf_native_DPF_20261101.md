# PyDPF: full Dacca comparison with the original differentiable filter

The original `StopGradientDPF` and the added public interface return identical monthly log-likelihood factors on the tested Dacca inputs. This holds for both public discount values, 1 and 0.97, for twelve starting rows and two validation seeds: 48 endpoint comparisons at T=600 and J=5000. Starting parameters, seeds and final random-generator states also match exactly. All eight evaluations and four isolated Adam proposals are finite.

The original ordinary PF and original `StopGradientDPF` disagree on these same GPU inputs. The largest absolute difference in their summed log likelihood is 210.7509765625 for the first seed and 558.884765625 for the second. The public interface follows the original differentiable filter in both comparisons. The underlying GPU operation responsible for the ordinary-PF/DPF difference has not been identified. These results do not certify mathematical derivatives.

Job **65169994 remains FAILED, exit 1:0**, because its original test required ordinary-PF/DPF equality. The failed comparisons and raw arrays are retained. Its `COMPLETE.json` read an obsolete report filename; the separately saved `result.json` and all eight evaluation arrays are complete. The audit records this metadata error without changing the original files. Collection verified eight source files, 84 referenced files, 23 transferred files and 17 terminal records. The job used 571 A40 GPU seconds.

## Measured resource use

Three public IF2 passes with a complete 5,000-by-23 parameter swarm took 28.660 seconds, or 9.553 seconds per pass. The largest measured GPU allocation across the checks was 6,966,062,592 bytes. The estimates below count a twelve-filter gradient batch once. They include the original 5% margin and preparation/scoring reserves; they are forecasts, not measured complete fits.

| Phase | Estimated allocation needed (seconds) | Original limit (seconds) |
|---|---:|---:|
| Twelve shared IF2 prefixes, 175 passes each | 21,664.908 | 24,300 |
| A: 400 Adam updates, twelve filters | 26,416.410 | 27,900 |
| B: 400 Adam updates, twelve filters | 26,416.410 | 27,900 |
| C: 400 Adam updates, twelve filters | 26,045.145 | 27,900 |

The complete four-phase allocation remains 108,000 seconds, or 30 GPU hours. Validation has used 2,522 of its separate 3,600-second allowance, including earlier failed checks.

## Next experiment

Under the user's decision to proceed with empirical comparisons after likelihood agreement, the new revision uses the original differentiable PF as its forward reference. The ordinary-PF discrepancy remains an explicit limitation. The package, Dacca model, systematic resampling, float32 arithmetic, checkpoint setting and default nondeterministic execution are unchanged. The driver adds records of rejected proposals before rollback, including loss, gradient, natural parameters and Adam state. Small CPU tests verify three deliberately injected failure types and preservation of the surviving row; these are regression tests, not optimization results.

Only the planned shared IF2 phase has been submitted: **65170442**, one A40, four CPUs, 48 GiB host memory, at most 6 hours 45 minutes. It runs the twelve existing starting rows. The A/B/C gradient phases have not been submitted. Their submission follows review of the actual IF2 results and cost. No completed fit is repeated. At 10:09 UTC the IF2 job was pending Priority.

The frozen new source contains 158 files, with manifest SHA256 `af918da10aa5e0926b306ec0e6a3db6663d152fc4c3cc7e14fd384dd8e17a1b3`. The [saved review and submission](../results/pydpf/native_DPF_20261101.json) links this decision to the failed check and its complete raw-data audit. Full source, arrays, logs, failure records and CPU checks are retained in the dated archive catalog.
