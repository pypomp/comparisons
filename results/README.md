# Results

The main comparison contains 168 final estimates, with six starts × two repetitions per method. It uses native PFJAX A/B and reports our earlier α=1 correction separately.

| Table | Contents |
|---|---|
| [Likelihoods](overview_20261102/likelihood.csv) | Medians, failures, floors and precision |
| [Paired differences](overview_20261102/paired_differences.csv) | C−B; CPU particles B−A |
| [Fitting times](overview_20261102/fit_times.csv) | Times per fit or per batch, with units stated |
| [PFJAX five methods](overview_20261102/pfjax_five_methods.csv) | Native A/B, added-α=1 A/B and C |
| [PFJAX individual records](pfjax_native/native_and_extension_20261102.csv) | All sixty scores and fitting times |

Endpoint sources: [Cuthbert](cuthbert/endpoints.csv), [PFJAX native](pfjax_native/final_comparison_20261102.csv), [PFJAX extension](pfjax/endpoints.csv), [TFP](tfp/endpoints.csv), [PyDPF](pydpf/endpoints.csv), [particles](particles/endpoints.csv).

Run `make overview` to regenerate the summary and `make check` to verify it. Medians include all finite scores, including scores at the density floor and those from failed fits. Individual PF evaluations, checkpoints and older tables are [archived](../archive/README.md).
