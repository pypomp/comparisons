"""Persistent, idle-between-requests PF worker; upstream/candidate are isolated."""
import sys,json,time,hashlib
import numpy as np
import model as m
z=dict(zip(m.NAMES,m.INPUT['theta_published'][0].astype(float)))
for k in m.POSITIVE+m.IVPS:z[k]=np.log(z[k])
z['beta_trend']*=100
point=np.array([z[k] for k in m.FREE])
for line in sys.stdin:
    request=json.loads(line);begin=time.perf_counter()
    pf=m.native_pf(point,request['J'],request['seed'],T=request['T'])
    elapsed=time.perf_counter()-begin
    print(json.dumps(dict(seed=request['seed'],seconds=elapsed,loglik=float(pf.logLt),
        X_sha256=hashlib.sha256(pf.X.tobytes()).hexdigest(),
        W_sha256=hashlib.sha256(pf.W.tobytes()).hexdigest(),
        rng_sha256=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest(),
        rng_position=np.random.get_state()[2],observations=int(pf.t))),flush=True)
