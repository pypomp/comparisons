# Dacca study design

T=600 monthly observations, twenty Gaussian Euler steps per month, J=5000 particles, and six common starting vectors repeated twice. We update 23 coordinates: 18 process/observation coordinates and five normalized initial-value coordinates; the latter representation is redundant.

| Method | Initialization | Optimization |
|---|---|---|
| A | Original start | Native gradient, external Adam, 400 updates |
| B | 175 IF2 passes | Same gradient and Adam as A, 400 updates |
| C | Same IF2 result as B | Our α=0.97 gradient and Adam, 400 updates |

B and C use the same initial parameters, zero initial Adam state, learning rates and random-number streams during training. PFJAX A/B differentiate through the package's basic PF, giving a biased gradient. C changes both the gradient calculation and α. We report the earlier PFJAX α=1 correction separately. [Functions and additions](../docs/package_calls.md) identify each implementation.

particles has a separate CPU comparison. A uses Nelder-Mead for 1,800 seconds. B uses IF2 for 900 seconds and Nelder-Mead for the remainder. It has no gradient method C.

We evaluate each final parameter estimate independently with the same original PyPOMP PF. Scores are log(mean likelihood), with J5000 and 36/72/144/288 repeats as needed for MCSE≤1 and likelihood ESS≥10. We retain all failures, density-floor scores and scores that miss these precision criteria. PFJAX C uses its existing scores without choosing among alternative evaluations.

A and B do different amounts of work, and packages use different IF2 perturbations and cooling schedules. Total computation is therefore not held equal. Each B or C fitting time includes IF2, but their shared IF2 run is counted only once in total resource use. TFP and PyDPF times cover a batch of twelve fits, rather than one fit.

[Results](optimization.md) · [Figures](figures.md) · [Correctness](correctness.md) · [Reproduction](../docs/reproduction.md).
