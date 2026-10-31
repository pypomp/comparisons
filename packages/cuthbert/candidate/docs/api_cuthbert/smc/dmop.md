# Discounted MOP

The original `stop_gradient_decorator` uses the selected ancestors' log weights
to correct derivatives through resampling. The added
`discounted_stop_gradient_decorator` uses the same resampling interface and
multiplies that derivative correction by `alpha`. The correction is zero in
the forward calculation. It does not change particle selection, propagation
or observation likelihoods.

`log_likelihood` constructs the original `particle_filter.build_filter` with
this decorator. Its internal loop calls the original PF update functions and
returns only the terminal state, with optional rematerialization for gradients.
Rematerialization is disabled by default. It saves intermediate storage but
can change floating-point gradients; comparisons with the original ADPF use
`checkpoint=False` on the public interface.
`fit` differentiates this objective and calls a caller-supplied optimizer.
The optimizer is not an ADPF implementation and does not replace the PF.

At `alpha=1`, this wrapper calls the original `stop_gradient_decorator`.
The existing ADPF function remains unchanged. For
`alpha<1`, it is a discounted derivative, not the ordinary derivative of the
unchanged forward likelihood. This extension is restricted to pathwise
differentiable bootstrap models with resampling at every observation.

::: cuthbert.smc.dmop

## Model factory and optimizer

::: cuthbert.smc._estimation.ModelFactory

::: cuthbert.smc._estimation.Optimizer
