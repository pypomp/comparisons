"""Full original Dacca integration, two IF2 passes, not a fitting benchmark."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('PARTICLES_IF2_OUTPUT', str(HERE / 'results')))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
os.environ['PFB_DACCA_INPUTS'] = str(HERE / 'frozen.npz')
import numpy as np
from scipy.special import logsumexp
import particles
from particles import resampling as rs
from particles import state_space_models as ssms
from particles.iterated_filtering import iterated_filtering
import dacca_common as dc
from dacca_numpy import Dacca
from bench_particles import DaccaSSM

# Execute only the unchanged coordinate transform, not a historical driver.
node = next(n for n in ast.parse((HERE / 'frozen_support.py').read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'natural')
inp = dc.load_inputs()
NAMES = inp['paramnames']
FIXED = ['rho', 'c', 'alpha', 'delta', 'Y_0']
IVPS = ['S_0', 'I_0', 'R1_0', 'R2_0', 'R3_0']
FREE = [n for n in NAMES if n not in FIXED]
scope = dict(np=np, FREE=FREE, NAMES=NAMES, IVPS=IVPS,
             POSITIVE=['gamma', 'epsilon', 'm', 'sigma', 'tau'])
exec(compile(ast.Module(body=[node], type_ignores=[]), 'frozen_natural', 'exec'), scope)
natural = scope['natural']


def factory(theta):
    return DaccaSSM(Dacca(natural(theta), inp))


def reference(cloud, scales, seed):
    """Independent time loop; uses original model laws, not candidate FK or SMC."""
    np.random.seed(seed)
    # Public IF2 explicitly uses float64 estimation coordinates. The historical
    # starting-point file is float32; retaining that dtype here would round every
    # in-place perturbation and compare two different arithmetic precisions.
    theta = np.array(cloud, dtype=float, copy=True)
    endpoints, lls = [], []
    for m in range(2):
        total = 0.0
        for t, y in enumerate(inp['ys']):
            if t:
                a = rs.resampling('multinomial', weights, M=len(theta))
                theta, xp = theta[a], x[a]
            else:
                xp = None
            sd = scales[t] * 0.8**m
            active = sd > 0
            theta[:, active] += np.random.normal(size=(len(theta), active.sum())) * sd[active]
            states = []
            models = [factory(th) for th in theta]
            for i, model in enumerate(models):
                law = model.PX0() if t == 0 else model.PX(t, xp[i:i+1])
                states.append(law.rvs(size=1))
            x = np.concatenate(states, axis=0)
            lw = np.array([model.PY(t, None if xp is None else xp[i:i+1], x[i:i+1]).logpdf(y)[0]
                           for i, model in enumerate(models)])
            total += logsumexp(lw) - np.log(len(theta))
            weights = np.exp(lw - logsumexp(lw))
        theta = theta[rs.resampling('multinomial', weights, M=len(theta))].copy()
        endpoints.append(theta.copy())
        lls.append(total)
    return endpoints, lls


assert inp['T'] == 600 and inp['nstep_const'] == 20 and len(FREE) == 23
cloud = np.tile(np.load(HERE / 'paper_starts_estimation_free.npy')[0], (8, 1))
scales = np.full((600, 23), 0.002)
for i, name in enumerate(FREE):
    if name in IVPS:
        scales[:, i] = 0.0
        scales[0, i] = 0.008
print('Running full Dacca public IF2 interface', flush=True)
np.random.seed(20261030)
actual = iterated_filtering(factory, inp['ys'], cloud, scales, 2, cooling=0.8)
print('Running independent explicit recursion', flush=True)
expected, ref_ll = reference(cloud, scales, 20261030)
np.testing.assert_array_equal(actual.theta_particles, expected[-1])
np.testing.assert_array_equal(actual.means[1:], np.mean(np.asarray(expected), axis=1))
np.testing.assert_allclose(actual.loglik, ref_ll, rtol=0, atol=1e-8)
print('Running unperturbed original PF twice at the fixed output', flush=True)
scores = []
for seed in [20261031, 20261032]:
    np.random.seed(seed)
    pf = particles.SMC(fk=ssms.Bootstrap(ssm=factory(actual.means[-1]), data=inp['ys']),
                       N=8, resampling='systematic', ESSrmin=1.0, collect='off')
    pf.run()
    scores.append({'seed': seed, 'loglik': float(pf.logLt)})
assert np.isfinite([x['loglik'] for x in scores]).all()
record = dict(passed=True, T=600, euler_steps_per_month=20, J=8, parameters=23,
    niter=2, dtype='float64', device='CPU', model_source='unchanged historical Dacca NumPy model and StateSpaceModel',
    public_interface='particles.iterated_filtering.iterated_filtering',
    parameter_max_abs_error=float(np.max(np.abs(actual.theta_particles - expected[-1]))),
    perturbed_loglik_max_abs_error=float(np.max(np.abs(actual.loglik - ref_ll))),
    perturbed_loglik=actual.loglik.tolist(), parameter_means=actual.means.tolist(),
    terminal_cloud=actual.theta_particles.tolist(), reference_loglik=ref_ll,
    unperturbed_PF_values=scores, independent_score_precision_claimed=False,
    fitting_performance_claimed=False, speed_comparison_performed=False,
    same_seed_as_historical_fitting_claimed=False,
    versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','numba','joblib']},
    copied_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [HERE / n for n in ['bench_particles.py','dacca_numpy.py','dacca_common.py',
                            'frozen_support.py','frozen.npz','paper_starts_estimation_free.npy']]})
(OUT / 'dacca_check.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['terminal_cloud','parameter_means']}))
