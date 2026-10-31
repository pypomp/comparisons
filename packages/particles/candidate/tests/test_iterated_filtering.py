import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import logsumexp

import particles
from particles import distributions as dists, resampling as rs
from particles import state_space_models as ssms
from particles.iterated_filtering import if2_step, iterated_filtering


class Gaussian(ssms.StateSpaceModel):
    def __init__(self, theta):
        self.theta = theta

    def PX0(self):
        return dists.Normal(loc=self.theta[..., 0], scale=0.4)

    def PX(self, t, xp):
        return dists.Normal(loc=0.6 * xp + self.theta[..., 0], scale=0.4)

    def PY(self, t, xp, x):
        previous = 0.0 if xp is None else xp
        return dists.Normal(loc=x + 0.2 * previous + self.theta[..., 1], scale=0.7)


DATA = np.array([0.2, -0.4, 0.7, 1.1])
CLOUD = np.arange(24).reshape(12, 2) / 100
SCALES = np.array([[0.1, 0.2], [0.05, 0], [0.02, 0], [0.01, 0]])


def reference(seed, cloud, scales, niter=1, cooling=1.0, resampling="multinomial"):
    """Explicit Gaussian recursion; no wrapper, Bootstrap or SMC time loop."""
    np.random.seed(seed)
    theta = cloud.copy()
    records = []
    for m in range(niter):
        total = 0
        for t, y in enumerate(DATA):
            if t:
                ancestors = rs.resampling(resampling, weights, M=len(theta))
                theta, previous = theta[ancestors], x[ancestors]
            else:
                previous = 0
            active = scales[t] > 0
            current_scale = scales[t, active] * cooling**m
            theta[:, active] += (np.random.normal(size=(len(theta), active.sum()))
                                  * current_scale)
            loc = theta[:, 0] + (0.6 * previous if t else 0)
            x = loc + 0.4 * np.random.normal(size=len(theta))
            mu = x + 0.2 * previous + theta[:, 1]
            logw = -0.5 * ((y - mu) / 0.7)**2 - np.log(0.7) - 0.5 * np.log(2*np.pi)
            total += logsumexp(logw) - np.log(len(theta))
            weights = np.exp(logw - logsumexp(logw))
        theta = theta[rs.resampling(resampling, weights, M=len(theta))].copy()
        records.append(total)
    return theta, np.array(records)


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("scheme", ["multinomial", "systematic", "stratified", "residual"])
def test_independent_recursion(vectorized, scheme):
    expected, loglik = reference(105, CLOUD, SCALES, 3, 0.6, scheme)
    np.random.seed(105)
    got = iterated_filtering(Gaussian, DATA, CLOUD, SCALES, 3, cooling=0.6,
                            vectorized=vectorized, resampling=scheme)
    assert_array_equal(got.theta_particles, expected)
    assert_allclose(got.loglik, loglik, rtol=0, atol=2e-14)
    assert_array_equal(got.means[-1], expected.mean(axis=0))
    assert_array_equal(CLOUD, np.arange(24).reshape(12, 2) / 100)


class AlwaysBootstrap(ssms.Bootstrap):
    def time_to_resample(self, smc):
        return True


@pytest.mark.parametrize("vectorized", [False, True])
def test_zero_perturbation_native_model_and_smc(vectorized, monkeypatch):
    theta = np.array([-1.0, 0.7, np.log(0.2)])
    def factory(p):
        return ssms.StochVol(mu=p[..., 0], rho=p[..., 1], sigma=np.exp(p[..., 2]))
    np.random.seed(17)
    native = particles.SMC(fk=AlwaysBootstrap(ssm=factory(theta), data=DATA),
                           N=12, resampling="multinomial", collect="off")
    native.run()
    calls = []
    old = particles.SMC.run
    def record(self):
        calls.append(type(self))
        return old(self)
    monkeypatch.setattr(particles.SMC, "run", record)
    np.random.seed(17)
    got = if2_step(factory, DATA, np.tile(theta, (12, 1)), np.zeros(3),
                   vectorized=vectorized)
    assert calls == [particles.SMC]
    assert_array_equal(got.smc.X["x"], native.X)
    assert_array_equal(got.smc.W, native.W)
    assert got.loglik == native.logLt
    assert_array_equal(got.theta_particles, np.tile(theta, (12, 1)))


def test_split_continuation_preserves_cloud_and_rng():
    np.random.seed(88)
    whole = iterated_filtering(Gaussian, DATA, CLOUD, SCALES, 4, cooling=0.8)
    final_rng = np.random.get_state()
    np.random.seed(88)
    first = iterated_filtering(Gaussian, DATA, CLOUD, SCALES, 2, cooling=0.8)
    checkpoint = first.theta_particles.copy(), np.random.get_state()
    np.random.normal(size=99)
    np.random.set_state(checkpoint[1])
    second = iterated_filtering(Gaussian, DATA, checkpoint[0], SCALES, 2,
                                start_iteration=2, cooling=0.8)
    assert_array_equal(second.theta_particles, whole.theta_particles)
    assert_array_equal(np.r_[first.loglik, second.loglik], whole.loglik)
    assert_array_equal(np.random.get_state()[1], final_rng[1])
    assert np.random.get_state()[2:] == final_rng[2:]


def test_fixed_coordinate_and_initial_only_perturbations():
    # With distinct original values, verify inheritance rather than a false
    # assertion that row identities must survive ancestor selection.
    scales = SCALES.copy()
    scales[:, 1] = 0
    np.random.seed(94)
    got = if2_step(Gaussian, DATA, CLOUD, scales)
    assert set(got.theta_particles[:, 1]) <= set(CLOUD[:, 1])
    expected, _ = reference(94, CLOUD, scales)
    assert_array_equal(got.theta_particles, expected)


class VectorGaussian(ssms.StateSpaceModel):
    def __init__(self, theta):
        self.theta = theta
    def PX0(self):
        return dists.MvNormal(loc=np.array([self.theta[0], 0]), cov=np.eye(2))
    def PX(self, t, xp):
        return dists.MvNormal(loc=0.5*xp, cov=np.eye(2))
    def PY(self, t, xp, x):
        return dists.Normal(loc=x[:, 0], scale=1)


def test_vector_state_and_one_observation():
    got = if2_step(VectorGaussian, [0.2], np.zeros((5, 1)), [0.1])
    assert got.smc.X["x"].shape == (5, 2)
    assert got.theta_particles.shape == (5, 1)
    assert np.isfinite(got.loglik)


def test_zero_passes_does_not_sample_or_alias():
    np.random.seed(63)
    before = np.random.get_state()
    got = iterated_filtering(Gaussian, DATA, CLOUD, SCALES, 0)
    assert_array_equal(got.theta_particles, CLOUD)
    assert not np.shares_memory(got.theta_particles, CLOUD)
    assert got.loglik.shape == (0,)
    assert_array_equal(np.random.get_state()[1], before[1])
    assert np.random.get_state()[2:] == before[2:]


class BadObservation(Gaussian):
    value = np.nan
    def PY(self, t, xp, x):
        value = self.value
        class Law:
            def logpdf(self, y):
                return np.full(len(x), value)
        return Law()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bad_weights_raise(value):
    class Model(BadObservation):
        pass
    Model.value = value
    with pytest.raises(FloatingPointError):
        if2_step(Model, DATA, CLOUD, SCALES)


def test_some_impossible_particles_are_allowed():
    class Model(Gaussian):
        def PY(self, t, xp, x):
            class Law:
                def logpdf(self, y):
                    return np.where(x < 0, -np.inf, 0.0)
            return Law()
    np.random.seed(1)
    got = if2_step(Model, [0], CLOUD, [0, 0], vectorized=True)
    assert np.isfinite(got.loglik)


@pytest.mark.parametrize("kwargs", [dict(niter=-1), dict(niter=1.5), dict(niter=True),
    dict(start_iteration=-1), dict(cooling=0), dict(cooling=np.nan), dict(cooling=1.1),
    dict(scales=[-1, 0]), dict(scales=[np.nan, 0]), dict(scales=[0]),
    dict(data=[]), dict(theta_particles=[[np.nan, 0]]), dict(theta_particles=[0, 0]),
    dict(resampling="unknown"), dict(resampling="ssp")])
def test_invalid_inputs(kwargs):
    arguments = dict(model_factory=Gaussian, data=DATA, theta_particles=CLOUD,
                     scales=SCALES, niter=1)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        iterated_filtering(**arguments)


def test_uniform_weights_still_resample_every_transition():
    class Model(BadObservation):
        value = 0.0
    np.random.seed(2)
    got = if2_step(Model, DATA, CLOUD, [0, 0])
    assert got.smc.rs_flag
    assert not np.array_equal(got.smc.A, np.arange(len(CLOUD)))
    assert got.loglik == 0.0
