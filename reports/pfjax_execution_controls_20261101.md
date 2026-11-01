# PFJAX: execution controls and the first A/B/C case

The completed controls show that original and extended PF likelihoods agree when the execution context is matched. The previously observed differences also occur between the original PF's own forward-only and differentiated calls. The specific GPU operation responsible has not been identified.

Job **65168970** completed on A40/gl1526 in **354 seconds**. We checked 8 frozen source files, 394 referenced files, the transferred records and all 164 terminal files. Saved arrays contain 144 finite evaluations at T=600/J=5000: cases 0, 2 and 11; original PF and public alpha=0, 1 and .97; forward-only and `value_and_grad`; three repeats; checkpoint enabled and disabled. All 1,656 saved comparisons were recomputed without calling a PF.

| Original PF case | Difference between forward-only and differentiated log likelihood, absolute value | Original/public likelihood agreement in the same execution context |
|---|---:|---|
| 0 | 11.716309 | Exact in every pair |
| 2 | 0 | Exact in every pair |
| 11 | 112.315430 | Exact in every pair |

Both checkpoint settings give this pattern. Across all original/public comparisons with matched execution context, **324/324 likelihood pairs are exactly equal**. Original self-repeats within each execution context also have identical likelihoods. Gradients remain recorded separately; comparisons between different alpha values are not gradient correctness tests.

The earlier [full Dacca check](pfjax_forward_20261101.md) remains a failed check: it compared native forward-only values to public differentiated values and failed in four of twelve case identities. These new controls cover three case identities, including two of those failures and one agreeing control. They support an execution-context explanation for the two reproduced discrepancies, not a claim that every numerical difference has been explained. No tolerance or package source was changed.

## Empirical A/B/C experiment

Under the user's acceptance of likelihood agreement for empirical comparison, **65169945_0** is the first of the planned twelve cases. It was running on A40/gl1511 at the saved 09:39 UTC query. This is a full formal case, not an additional pilot. The other eleven cases have not been submitted and will follow only after this case's parameters, pairing, independent scores and actual allocation time are reviewed.

The experiment keeps T=600, 20 Euler steps per month, J=5000, 175 public IF2 passes and 400 Adam updates per method. A starts directly; B and C start from the same IF2 endpoint and use alpha=1 and .97. B/C have identical zero Adam state, learning rates and training keys. Their total work exceeds A's. The added `particle_filter_mop` and `iterated_filter_step` use the original PF loop; alpha=1 is also an added correction. The optimizer is external Optax. Production checkpoint=True is unchanged.

The driver changes only add rejected IF2/Adam proposal records. The scorer is the unchanged original PyPOMP PF, with independent scoring keys and adaptive R=36/72/144/288. No failed fit is retried or selected by its best training loss. All low-precision, failed and nonfinite outcomes will be retained.

The previous full-scale check measured five finite IF2 passes and eight finite Adam proposals. Its conservative complete-case forecast is **2810.537 seconds**, below the existing 3120-second cost threshold. Each formal allocation is capped at 3360 seconds. Including the three completed diagnostics/checks, the upper total is **41161 of 42480 GPU seconds**. This is a resource budget, not a wall-clock waiting-time estimate or a gradient certification.

[Recorded analysis and submission](../results/pfjax/execution_controls_20261101.json). Full arrays, StableHLO, source, failed preparation attempts, CPU checks, submission records and saved-data audit programs are in the dated archive listed in [the archive catalog](../archive/catalog.json). Historical validation failures remain unchanged.
