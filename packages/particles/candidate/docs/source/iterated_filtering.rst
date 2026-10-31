Iterated filtering
==================

This proposed extension adds IF2 parameter perturbations to the existing
bootstrap particle filter. It is not an automatic differentiation method.
The implementation uses ``Bootstrap.M0``, ``Bootstrap.M`` and ``Bootstrap.logG``
inside the original ``SMC.run``. The SMC core, distributions and resamplers are
unchanged. The extra Feynman--Kac callbacks store state and parameter coordinates
in the same structured particle, so native ancestor indexing selects both.

Model interface
---------------

A factory takes a parameter vector and returns a normal ``StateSpaceModel``.
The model must implement ``PX0().rvs``, ``PX(t,xp).rvs`` and
``PY(t,xp,x).logpdf``. No transition ``logpdf`` or inverse CDF is needed.
Parameters are perturbed on the coordinate scale chosen by the caller; use
log/logit transforms in the factory for constrained parameters. There is no
automatic clipping, rejection or constraint transformation.

For example, the existing ``StochVol`` class needs no changes::

    import numpy as np
    from particles import state_space_models as ssms
    from particles.iterated_filtering import iterated_filtering

    def model_factory(theta):
        return ssms.StochVol(mu=theta[..., 0],
                            rho=np.tanh(theta[..., 1]),
                            sigma=np.exp(theta[..., 2]))

    theta0 = np.array([-0.5, np.arctanh(0.7), np.log(0.3)])
    np.random.seed(31)
    result = iterated_filtering(
        model_factory, data, np.tile(theta0, (100, 1)),
        scales=[0.1, 0.05, 0.05], niter=3, cooling=0.9,
        vectorized=True)

``vectorized=False`` (the default) constructs a scalar-parameter model for each
particle. It supports models that do not accept parameter arrays, but Python
loops and repeated model construction may be expensive. ``vectorized=True``
passes an ``(N,P)`` array; the factory must return a model whose distributions
correctly broadcast particle-specific parameters. Not all existing models
support this. Constructors and factories must not sample or mutate inputs.
States must be numeric arrays with leading particle axis; scalar and vector
states are supported, arbitrary nested objects are not.

Perturbations and repeated passes
--------------------------------

The initial input is the complete ``(N,P)`` equal-weight parameter cloud.
``scales`` is either ``(P,)`` or ``(T,P)``. Row zero controls perturbations
before ``PX0``; row t controls perturbations before ``PX(t, ...)``.
``PX0`` samples the state for the first observation, following the original
Bootstrap convention. For initial-value parameters, set their later rows to
zero. Setting an entire coordinate's scale to zero freezes perturbations, but
ancestor selection can still select among different initial values.

On pass m, every scale is multiplied by ``cooling**m``. Parameters perform
independent Gaussian random walks. Native resampling runs at every transition,
even with exactly uniform weights. Final native resampling returns the entire
equal-weight parameter cloud to the next pass, which reinitializes latent
states. The cloud is never replaced by copies of its mean. The reported mean
is in estimation coordinates; transform it explicitly to obtain model parameters.

``result.loglik`` contains perturbed-model normalizers. It is not the likelihood
of the mean parameter. For an unperturbed score, use a fresh ordinary
``Bootstrap`` and ``SMC`` at a fixed parameter, with separate random numbers.
The short executable example in ``book/mle/if2_stochvol.py`` does this, but its
three passes and two single scores do not establish optimization quality.

Continuation and limitations
----------------------------

To continue, retain ``result.theta_particles``, NumPy's complete RNG state,
and the iteration index. Restore the RNG and pass ``start_iteration`` to
continue the cooling schedule. This is an in-memory continuation interface,
not a checkpoint file format. If a model uses other random generators, the
caller must separately preserve their state. Models used in the example use
NumPy's global RNG. Concurrent threads sharing this RNG are not supported.

Supported resamplers are multinomial (default), systematic, stratified and
residual. SSP is excluded because its Numba-compiled sampler uses a separate
RNG. SQMC, guided/auxiliary filters, differentiation, Adam and IFAD are not
implemented by this module. Nonfinite parameters/states, NaN or positive
infinite weights, and all-negative-infinite weights raise exceptions. Individual
zero-weight particles are allowed. There is no automatic retry, best-iterate
selection, per-particle failure recovery or convergence claim.

Relationship to existing estimation methods
------------------------------------------

``mcmc.PMMH`` and ``smc_samplers.SMC2`` provide Bayesian parameter inference
using particle-filter likelihood estimates and a prior. ``mcmc.ParticleGibbs``
also requires a model-specific ``update_theta`` method. They estimate posterior
distributions and should not be relabeled as maximum-likelihood optimizers.
The book's ``mle_neuro.py`` contains model-specific EM and score formulas and
an example using external ``scipy.optimize.minimize(method='Nelder-Mead')``.
This IF2 module neither replaces those methods nor makes them differentiable.

.. automodule:: particles.iterated_filtering
   :members: if2_step, iterated_filtering, IF2Step, IF2Result
