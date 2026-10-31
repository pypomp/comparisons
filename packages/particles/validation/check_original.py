"""Compare unchanged original algorithms in separate baseline/candidate processes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get('PARTICLES_IF2_OUTPUT', str(HERE / 'validation/results')))
OUT.mkdir(parents=True, exist_ok=True)
code = '''
import hashlib,json,numpy as np,particles
from particles import state_space_models as ssms
try:
    import particles.iterated_filtering
except ModuleNotFoundError:
    pass
out={}
data=np.array([.1,-.3,.2,.7,-.5,.9])
for cls in [ssms.Bootstrap,ssms.GuidedPF,ssms.AuxiliaryPF]:
    np.random.seed(81)
    pf=particles.SMC(fk=cls(ssm=ssms.StochVol(),data=data),N=32,collect="off")
    pf.run()
    out[cls.__name__]={'loglik':float(pf.logLt),'X':pf.X.tolist(),'W':pf.W.tolist(),
      'rng_array_sha256':hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest(),
      'rng_position':np.random.get_state()[2]}
print(json.dumps(out))
'''
values = {}
for name in ['upstream','candidate']:
    env = dict(os.environ, PYTHONPATH=str(HERE / name))
    p = subprocess.run([sys.executable, '-c', code], env=env, text=True,
                       capture_output=True, check=True)
    values[name] = json.loads(p.stdout)
assert values['upstream'] == values['candidate']

# Confirm the diagnosed SSP RNG issue without altering the candidate sampler.
import numpy as np
import numba
from particles import resampling as rs
weights = np.arange(1.0, 101.0)
weights /= weights.sum()
@numba.njit
def seed_numba(seed):
    np.random.seed(seed)

seed_numba(17)
np.random.seed(15)
a = rs.resampling('ssp', weights, M=100)
np.random.seed(15)
b = rs.resampling('ssp', weights, M=100)
seed_numba(17)
np.random.seed(15)
c = rs.resampling('ssp', weights, M=100)
assert not np.array_equal(a,b)
np.testing.assert_array_equal(a,c)
record = {'passed': True, 'original_algorithms_compared': list(values['candidate']),
          'separate_process_outputs': values,
          'SSP_same_numpy_seed_repeats': False,
          'SSP_also_reseed_numba_repeats': True,
          'candidate_supports_SSP': False, 'full_original_suite_executed': False}
(OUT / 'original_checks.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='separate_process_outputs'}))
