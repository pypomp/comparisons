"""Iterated filtering using TFP's existing sequential Monte Carlo kernel.

Local candidate extension; this file is not part of the pinned TFP release.
"""

import collections
import importlib
import math

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import sequential_monte_carlo_kernel as smc
from tensorflow_probability.python.experimental.mcmc import weighted_resampling
from tensorflow_probability.python.internal import samplers

pf = importlib.import_module(
    'tensorflow_probability.python.experimental.mcmc.particle_filter')

__all__ = ['iterated_filter', 'iterated_filter_step', 'IteratedFilterResults']

IteratedFilterResults = collections.namedtuple(
    'IteratedFilterResults',
    ['parameters', 'parameter_particles', 'parameter_history',
     'perturbed_log_likelihoods'])


def _always_resample(state, particles_dim=0):
  return tf.ones_like(tf.reduce_sum(state.log_weights, axis=particles_dim,
                                    keepdims=True), dtype=tf.bool)


def _normal_sample(template, seed):
  return tf.random.stateless_normal(tf.shape(template), seed, dtype=template.dtype)


def iterated_filter_step(
    parameter_particles, observations, initial_state_prior_fn,
    transition_fn, observation_fn, perturbation_scale, *, initial_scale=0.,
    iteration=0, cooling_rate=.9, cooling_interval=1.,
    within_pass_cooling=False,
    resample_fn=weighted_resampling.resample_systematic,
    normal_sample_fn=_normal_sample, seed=None, name=None):
  """Run one perturbed bootstrap PF and retain its complete parameter cloud.

  Args:
    parameter_particles: Float Tensor [J, b1, ..., bN, P], in unconstrained
      estimation coordinates. Particle axis is zero; P is statically known.
    observations: Tensor [T, b1, ..., bN, ...]. T must be statically known
      and at least two. Observation zero weights the initial state, exactly
      as in particle_filter. Models with observations after transitions can
      explicitly supply a dummy time-zero observation of log probability zero.
    initial_state_prior_fn: parameters -> Distribution whose batch shape is
      [J, b1, ..., bN]. sample(seed=...) generates the initial state. It may
      be a nested-state Distribution; no transition density is required.
    transition_fn: (step, state, parameters) -> Distribution with batch shape
      [J, b1, ..., bN]. Its sample(seed=...) advances from step to step+1.
    observation_fn: (step, state, parameters) -> Distribution whose log_prob
      returns one weight per particle and batch member.
    perturbation_scale: Nonnegative finite scale broadcastable to the
      parameter batch shape [b1, ..., bN, P]. Applied before each transition.
    initial_scale: Scale added before sampling the prior, on every pass.
    iteration: Nonnegative scalar pass index, starting at zero.
    cooling_rate: Python real in (0, 1].
    cooling_interval: Positive Python real. Cooling at transition t is
      rate ** ((iteration + t/(T-1)) / interval) when within_pass_cooling
      is True, otherwise rate ** (iteration / interval).
    within_pass_cooling: Whether to cool within a pass.
    resample_fn: Original TFP resampling callback signature. Resampling is
      always performed, including at the initial and final observations.
    normal_sample_fn: Optional (template, stateless_seed) -> standard normal
      Tensor of the same shape/dtype, for a caller-controlled RNG layout.
    seed: Stateless seed (or TFP-compatible seed).
    name: Optional name scope.

  Returns:
    parameter_particles: Equally weighted terminal cloud, retained between
      passes rather than replaced with copies of its mean.
    perturbed_log_likelihood: Per-batch likelihood of the perturbed model;
      this is not a final score for the static model.

  Invalid/nonfinite passes raise InvalidArgumentError; no automatic retries,
  per-member failure isolation or job recovery are implemented. Pure model
  constructors must support per-particle parameters. This function does not
  rewrite arbitrary parameter Variables in an existing model.
  """
  if not math.isfinite(cooling_rate) or not 0 < cooling_rate <= 1:
    raise ValueError('cooling_rate must be in (0, 1]')
  if not math.isfinite(cooling_interval) or cooling_interval <= 0:
    raise ValueError('cooling_interval must be finite and positive')
  with tf.name_scope(name or 'iterated_filter_step'):
    cloud = tf.stop_gradient(tf.convert_to_tensor(parameter_particles))
    observations = tf.convert_to_tensor(observations)
    if (not cloud.dtype.is_floating or cloud.shape.rank is None or
        cloud.shape.rank < 2 or cloud.shape[-1] is None or cloud.shape[-1] < 1):
      raise ValueError('parameter_particles must have shape [J, ..., P]')
    if observations.shape.rank is None or observations.shape.rank < 1:
      raise ValueError('observations must have a leading time dimension')
    length = observations.shape[0]
    if length is None or length < 2:
      raise ValueError('observations require a static length of at least two')
    index = tf.cast(tf.convert_to_tensor(iteration), cloud.dtype)
    checks = [tf.debugging.assert_all_finite(cloud, 'nonfinite initial cloud'),
              tf.debugging.assert_positive(tf.shape(cloud)[0]),
              tf.debugging.assert_rank(index, 0),
              tf.debugging.assert_non_negative(index),
              tf.debugging.assert_equal(index, tf.floor(index)),
              tf.debugging.assert_all_finite(index, 'nonfinite iteration')]

    def scale(value):
      value = tf.broadcast_to(tf.convert_to_tensor(value, dtype=cloud.dtype),
                              tf.shape(cloud)[1:])
      with tf.control_dependencies([
          tf.debugging.assert_non_negative(value),
          tf.debugging.assert_all_finite(value, 'nonfinite perturbation scale')]):
        return tf.identity(value)

    process_scale, start_scale = scale(perturbation_scale), scale(initial_scale)

    def perturb(parameters, std, cooling, key):
      noise = normal_sample_fn(parameters, key)
      with tf.control_dependencies([
          tf.debugging.assert_equal(tf.shape(noise), tf.shape(parameters)),
          tf.debugging.assert_all_finite(noise, 'nonfinite perturbation noise')]):
        return parameters + noise * std * cooling

    # Match the two explicit streams of the existing Dacca IF2 pass. The
    # prior receives a separate folded seed; constructors must not draw RNGs.
    keys = tf.random.experimental.stateless_split(samplers.sanitize_seed(seed), 2)
    with tf.control_dependencies(checks):
      parameters = perturb(cloud, start_scale,
                           cooling_rate ** (index / cooling_interval), keys[0])
    prior = initial_state_prior_fn(parameters)
    state = prior.sample(seed=tf.random.experimental.stateless_fold_in(keys[0], 1))
    weights = tf.fill(tf.shape(parameters)[:-1],
                      -tf.math.log(tf.cast(tf.shape(cloud)[0], cloud.dtype)))
    initial_log_prob = pf._compute_observation_log_weights(
        0, state, observations,
        lambda step, x: observation_fn(step, x, parameters), particles_dim=0)
    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.shape(initial_log_prob), tf.shape(weights))]):
      initial = smc.WeightedParticles(
          {'state': state, 'parameters': parameters}, weights + initial_log_prob)

    def propose(step, weighted, seed):
      step_keys = tf.random.experimental.stateless_split(seed, 2)
      within = tf.cast(step, cloud.dtype) / (length - 1) if within_pass_cooling else 0.
      new_parameters = perturb(
          weighted.particles['parameters'], process_scale,
          cooling_rate ** ((index + within) / cooling_interval), step_keys[0])
      transition = transition_fn(step, weighted.particles['state'], new_parameters)
      new_state = transition.sample(seed=step_keys[1])
      log_prob = pf._compute_observation_log_weights(
          step + 1, new_state, observations,
          lambda t, x: observation_fn(t, x, new_parameters), particles_dim=0)
      with tf.control_dependencies([
          tf.debugging.assert_equal(tf.shape(log_prob), tf.shape(weights))]):
        return smc.WeightedParticles(
            {'state': new_state, 'parameters': new_parameters},
            weighted.log_weights + log_prob)

    final, log_likelihood = pf.sequential_monte_carlo(
        initial, propose, num_steps=length, resample_fn=resample_fn,
        resample_criterion_fn=_always_resample, unbiased_gradients=False,
        trace_fn=lambda s, r: (s.particles, r.accumulated_log_marginal_likelihood),
        trace_criterion_fn=None, seed=keys[1])
    assertions = [tf.debugging.assert_all_finite(x, 'nonfinite IF2 terminal state')
                  for x in tf.nest.flatten(final)]
    assertions.append(tf.debugging.assert_all_finite(
        log_likelihood, 'nonfinite IF2 likelihood'))
    with tf.control_dependencies(assertions):
      return (tf.stop_gradient(final['parameters']),
              tf.stop_gradient(log_likelihood))


def iterated_filter(
    initial_parameters, observations, initial_state_prior_fn, transition_fn,
    observation_fn, num_particles, num_iterations, perturbation_scale, *,
    seed=None, name=None, **step_kwargs):
  """Run complete IF2 passes with a TensorFlow while loop.

  initial_parameters has shape [b1, ..., bN, P]. num_particles and
  num_iterations are Python integers (positive/nonnegative respectively).
  Other arguments follow iterated_filter_step. Pass seeds are stateless
  fold_in(root_seed, pass_index). An invalid pass raises; this function does
  not return a selected best iterate or replay a failed pass.
  """
  for label, value, lower in [('num_particles', num_particles, 1),
                              ('num_iterations', num_iterations, 0)]:
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
      raise ValueError(label + ' must be an integer >= ' + str(lower))
  if 'iteration' in step_kwargs:
    raise ValueError('iteration is managed by iterated_filter')
  with tf.name_scope(name or 'iterated_filter'):
    parameters = tf.convert_to_tensor(initial_parameters)
    if (not parameters.dtype.is_floating or parameters.shape.rank is None or
        parameters.shape.rank < 1 or parameters.shape[-1] is None or
        parameters.shape[-1] < 1):
      raise ValueError('initial_parameters must have shape [..., P]')
    with tf.control_dependencies([tf.debugging.assert_all_finite(
        parameters, 'nonfinite initial parameters')]):
      cloud = tf.broadcast_to(parameters[None],
                              tf.concat([[num_particles], tf.shape(parameters)], 0))
    root_seed = samplers.sanitize_seed(seed)
    history = tf.TensorArray(parameters.dtype, size=num_iterations + 1,
                             element_shape=parameters.shape).write(0, parameters)
    likelihoods = tf.TensorArray(parameters.dtype, size=num_iterations,
                                 element_shape=parameters.shape[:-1])

    def body(i, cloud, history, likelihoods):
      cloud, ll = iterated_filter_step(
          cloud, observations, initial_state_prior_fn, transition_fn,
          observation_fn, perturbation_scale,
          iteration=tf.cast(i, parameters.dtype),
          seed=tf.random.experimental.stateless_fold_in(root_seed, i), **step_kwargs)
      return i + 1, cloud, history.write(i + 1, tf.reduce_mean(cloud, 0)), likelihoods.write(i, ll)

    _, cloud, history, likelihoods = tf.while_loop(
        lambda i, *_: i < num_iterations, body,
        (tf.constant(0), cloud, history, likelihoods), parallel_iterations=1)
    return IteratedFilterResults(history.read(num_iterations), cloud,
                                 history.stack(), likelihoods.stack())
