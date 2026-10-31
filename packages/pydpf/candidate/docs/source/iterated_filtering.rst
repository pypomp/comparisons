IF2 and discounted particle-filter gradients
===========================================

This local extension adds parameter perturbations to the existing bootstrap
``ParticleFilter`` and an optional ``alpha`` argument to ``StopGradientDPF``.
It does not replace the particle filter or add an optimizer. Training still
uses an ordinary Torch training loop, as in the existing PyDPF tutorial.

Using an existing model
-----------------------

Provide ``model_factory(parameters)`` returning a normal ``FilteringModel``.
Its prior and dynamic components implement ``sample`` and its observation
component implements ``fitness`` (the existing ``score`` spelling also works).
The factory applies parameter transformations and captures the resulting
parameters in these components. It must support a parameter tensor with one
row per particle, of shape ``(batch, particles, coordinates)``. A module that
only reads one global ``torch.nn.Parameter`` needs a model adapter first;
the extension does not automatically rewrite arbitrary PyDPF models.

The IF2 module handles parameter perturbation, ancestor selection, and the
iteration over complete filtering passes. Model callbacks contain none of
these operations. They use the existing PyDPF argument conventions, including
time, control and series metadata. This version supports vector-valued states
and parameters, bootstrap proposals, and unconditional systematic or
multinomial resampling.

The example in ``examples/if2_and_discount.py`` uses the same factory for a
normal ``StopGradientDPF`` and ``IteratedFiltering``:

.. code-block:: python

    warm = pydpf.IteratedFiltering(
        model_factory,
        perturbation_scale=process_scales,
        initial_scale=initial_scales,
        generator=generator,
    )(initial_parameters, observations, n_particles=64, n_iterations=3)
    if not warm.success:
        raise RuntimeError(warm.failure_reason)

The parameter swarm is appended to the model state. The original filter
resamples this joined state, so state and parameters use the same ancestors.
After the last observation, a further call to the original resampler produces
an equally weighted parameter swarm for the next pass. The swarm is retained
in full; its mean is not used to recreate the next swarm.

``initial_scale`` is applied at the beginning of each pass. Process
perturbations are applied before sampling the prior at observation zero and
before subsequent transitions. A coordinate that should change only at
initialization must have process scale zero. If a model requires distinct
parameter values for its unobserved initial state and a transition before its
first observation, this single-prior interface may need a separate adapter.

``cooling_rate**(m/cooling_interval)`` controls the scales in pass ``m``.
With ``within_pass_cooling=True``, observation ``t`` instead uses
``cooling_rate**((m+t/T)/cooling_interval)``. The caller supplies the scales and
cooling; no Dacca dimensions, parameter names, or paper settings are built in.

Failure of one pass stops the whole batch, reports ``success=False`` and keeps
the preceding valid swarm. Successful-pass histories and counts remain
available. This interface does not perform the historical benchmark driver's
per-row isolation, checkpoint recovery, or automatic retries.

Adding the gradient discount
----------------------------

.. code-block:: python

    pf = pydpf.StopGradientDPF(
        SSM=model_factory(parameters),
        resampling_generator=generator,
        alpha=0.97,
    )

The default ``alpha=1`` retains the existing resampling computation.
For ``alpha<1``, the derivative inherited from the selected log weights is
multiplied by ``alpha``. Derivatives through the sampled states remain.
The forward arithmetic is retained from the original stop-gradient resampler,
including its floating-point rounding. Thus alpha zero does not disable all
gradients, and finite differences of the forward likelihood are not a test
for the discounted derivative.

Discounting is restricted to pathwise bootstrap models and unconditional
systematic/multinomial resampling at every observation. Combining it with
guided proposals, conditional resampling, other resamplers, or transition-score
REINFORCE is not validated here. The default alpha-one API keeps its original
supported configuration choices.

IFAD and endpoint evaluation
---------------------------

Use the IF2 parameters as the initial values in the normal Torch optimizer
loop, then use ``StopGradientDPF(alpha=0.97)`` for the likelihood gradient.
Call ``pf.update()`` as instructed by the original training tutorial when
reusing modules with cached derived quantities. Adam remains ``torch.optim.Adam``;
neither it nor the complete IFAD training loop is called a native PyDPF MLE
routine. No new general training-loop API is introduced in this patch.

After fitting, evaluate fixed parameters with the ordinary PF and fresh random
draws. Keep each log likelihood; use log(mean likelihood) for repeated PF
evaluation, not the arithmetic mean of the log likelihoods. IF2's perturbed
model likelihoods and training objectives are not endpoint scores. The small
example only demonstrates execution and does not assess score precision.
