"""Independent adapter checks plus actual public-IF2 recursion and continuation."""
import json
import hashlib
import time
import numpy as np
from scipy.special import logsumexp
from numpy.testing import assert_allclose, assert_array_equal
import particles
from particles import resampling as rs
from particles.state_space_models import Bootstrap
from particles.iterated_filtering import if2_step, iterated_filtering
import model as m

def run():
    begin=time.perf_counter(); N=8
    z=m.STARTS[:N].astype(float).copy();th=m.natural(z)
    vec=m.VectorDacca(th,m.INPUT)
    scalar=[m.Dacca({n:th[n][i] for n in m.NAMES},m.INPUT) for i in range(N)]
    x=vec.rinit(N)
    expected={k:np.concatenate([s.rinit(1)[k] for s in scalar]) for k in x}
    for k in x:assert_allclose(x[k],expected[k],rtol=2e-15,atol=1e-8)
    rng=np.random.RandomState(81001);max_abs=0.
    # Distinct parameters, common explicit Gaussian draws, unchanged scalar step.
    for interval in [0,1,113,599]:
        x=vec.rinit(N);expected={k:v.copy() for k,v in x.items()}
        for k in range(20):
            noise=rng.normal(size=N)
            x=vec.step(x,interval,k,noise)
            ref=[s.step({key:v[i:i+1] for key,v in expected.items()},interval,k,noise[i:i+1]) for i,s in enumerate(scalar)]
            expected={key:np.concatenate([v[key] for v in ref]) for key in x}
            for key in x:
                max_abs=max(max_abs,float(np.max(abs(x[key]-expected[key]))))
                assert_allclose(x[key],expected[key],rtol=1e-11,atol=1e-7)
        got=vec.dmeas(m.INPUT['ys'][interval],x['Mn'],x['count'])
        ref=np.array([s.dmeas(m.INPUT['ys'][interval],expected['Mn'][i:i+1],expected['count'][i:i+1])[0] for i,s in enumerate(scalar)])
        assert_allclose(got,ref,rtol=1e-11,atol=1e-8)
    class Always(Bootstrap):
        def time_to_resample(self,smc):return True
    cloud=np.tile(m.STARTS[0].astype(float),(16,1))
    np.random.seed(81002)
    baseline=particles.SMC(fk=Always(ssm=m.factory(cloud),data=m.INPUT['ys']),N=16,
        resampling='multinomial',collect='off');baseline.run()
    np.random.seed(81002)
    zero=if2_step(m.factory,m.INPUT['ys'],cloud,np.zeros((600,23)),vectorized=True)
    assert_array_equal(zero.smc.X['x'],baseline.X)
    assert_array_equal(zero.smc.W,baseline.W)
    assert zero.loglik==baseline.logLt
    # Independent explicit IF2 recursion. Adapter laws separately checked above.
    small=m.dc.load_inputs(4);factory=lambda z:m.factory(z,small)
    sc=m.scales(4);seed=81003
    np.random.seed(seed)
    whole=iterated_filtering(factory,small['ys'],cloud,sc,3,cooling=.8,vectorized=True)
    whole_rng=np.random.get_state()
    np.random.seed(seed);theta=cloud.copy();lik=[]
    for step in range(3):
        total=0.
        for t,y in enumerate(small['ys']):
            if t:
                a=rs.resampling('multinomial',weights,M=len(theta));theta,xp=theta[a],x[a]
            else:xp=None
            active=sc[t]>0
            theta[:,active]+=np.random.normal(size=(len(theta),active.sum()))*sc[t,active]*(.8**step)
            obj=factory(theta);law=obj.PX0() if t==0 else obj.PX(t,xp)
            x=law.rvs(size=len(theta));lw=obj.PY(t,xp,x).logpdf(y)
            total+=logsumexp(lw)-np.log(len(theta));weights=np.exp(lw-logsumexp(lw))
        theta=theta[rs.resampling('multinomial',weights,M=len(theta))].copy();lik.append(total)
    assert_array_equal(theta,whole.theta_particles)
    assert_allclose(lik,whole.loglik,rtol=0,atol=1e-10)
    np.random.seed(seed)
    a=iterated_filtering(factory,small['ys'],cloud,sc,1,cooling=.8,vectorized=True)
    saved_rng=np.random.get_state();np.random.normal(size=47);np.random.set_state(saved_rng)
    b=iterated_filtering(factory,small['ys'],a.theta_particles,sc,2,cooling=.8,start_iteration=1,vectorized=True)
    assert_array_equal(b.theta_particles,whole.theta_particles)
    assert_array_equal(np.random.get_state()[1],whole_rng[1]);assert np.random.get_state()[2:]==whole_rng[2:]
    return dict(passed=True,adapter_max_state_abs_error=max_abs,zero_perturbation_full_T600_J16_exact=True,
        independent_perturbed_recursion_T4_J16_passes3=True,full_cloud_RNG_continuation_exact=True,
        euler_steps=20,coordinates=23,seconds=time.perf_counter()-begin,
        original_equations_reused=['Dacca.step','Dacca.interval','Dacca.dmeas','DaccaSSM'],
        actual_IF2_SMC_class=type(zero.smc).__module__+'.'+type(zero.smc).__name__)

if __name__=='__main__':print(json.dumps(run(),indent=2))
