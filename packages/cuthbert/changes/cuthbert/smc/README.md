# Sequential Monte Carlo Methods

The core atomic functions can be found in [`cuthbertlib.smc`](../../cuthbertlib/smc).

Parameter estimation for bootstrap models is available through
`iterated_filtering.fit` (IF2), `dmop.fit` (discounted MOP gradients), and
`ifad.fit` (IF2 followed by DMOP and independent PF scoring). These functions
resample at every observation and accept parameter PyTrees on an unconstrained
estimation scale. DMOP requires pathwise differentiable model sampling.
See the [IFAD example](../../docs/examples/parameter_estimation_ifad.md).

<!-- --8<-- [start:feynman_kac] -->
Sequential Monte Carlo methods provide particle approximations to general Feynman-Kac models:

$$
\mathbb{Q}_{t}(x_{0:t}) \propto \mathbb{M}_0(x_0) \, G_0(x_0) \prod_{s=1}^{t} M_s(x_s \mid x_{s-1}) \, G_s(x_{s-1}, x_s), \quad t \in \{ 0, \dots, T\},
$$

where $M_{s}$ are probability kernels and $G_{s}$ are positive and bounded functions.
<!-- --8<-- [end:feynman_kac] -->
