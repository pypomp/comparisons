"""Original Dacca simulator supplied to TensorFlow Probability's native PF.

The monthly transition has a sampler, deliberately no log_prob. A real
deterministic initial prior and an uninformative t0 observation avoid any
invented monthly density. All 600 original observations are then weighted.
"""
from contextlib import contextmanager
import importlib
import math
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd=tfp.distributions
ROOT=Path(__file__).resolve().parent
DATA=dict(np.load(ROOT/'input/frozen.npz'))
NAMES=DATA['paramnames'].tolist()
P={n:i for i,n in enumerate(NAMES)}
FIXED=['rho','c','alpha','delta','Y_0']
FREE=[n for n in NAMES if n not in FIXED]
IVPS=['S_0','I_0','R1_0','R2_0','R3_0']
IVPF=[FREE.index(n) for n in IVPS]
POSITIVE=['gamma','epsilon','m','sigma','tau']
LTOL=math.log(1e-18)
pfmod=importlib.import_module('tensorflow_probability.python.experimental.mcmc.particle_filter')
wr=importlib.import_module('tensorflow_probability.python.experimental.mcmc.weighted_resampling')

def to_est(theta):
    x=np.asarray(theta,dtype=np.float64)
    e=x[..., [P[n] for n in FREE]].copy()
    for n in POSITIVE:e[...,FREE.index(n)]=np.log(x[...,P[n]])
    e[...,FREE.index('beta_trend')]*=100
    iv=x[..., [P[n] for n in IVPS]]
    e[...,IVPF]=np.log(iv/iv.sum(-1,keepdims=True))
    return e.astype(np.float32)

def natural(e):
    values={n:e[...,i] for i,n in enumerate(FREE)}
    for n in POSITIVE:values[n]=tf.exp(values[n])
    values['beta_trend']/=100
    iv=tf.exp(tf.gather(e,IVPF,axis=-1));iv/=tf.reduce_sum(iv,-1,keepdims=True)
    values.update(zip(IVPS,tf.unstack(iv,axis=-1)))
    z=tf.zeros_like(e[...,0])
    values.update(rho=z,c=z+1,alpha=z+1,delta=z+.02,Y_0=z)
    return tf.stack([values[n] for n in NAMES],-1)

@contextmanager
def discounted_weights(alpha):
    """Local, trace-time gradient extension; restore native function afterwards.

    Native TFP owns propagation, resampling and likelihood accumulation.
    Only the derivative of its target resampling weights is discounted.
    This context is for a single tracing thread, not concurrent graph builds.
    """
    original=wr.resample
    def resample(*args,target_log_weights=None,**kwargs):
        if target_log_weights is not None:
            stop=tf.stop_gradient(target_log_weights)
            target_log_weights=stop+alpha*(target_log_weights-stop)
        return original(*args,target_log_weights=target_log_weights,**kwargs)
    wr.resample=resample
    try:yield
    finally:wr.resample=original

def always_resample(weighted_particles,particles_dim=0):
    return tf.ones_like(tf.reduce_sum(weighted_particles.log_weights,
        axis=particles_dim,keepdims=True),dtype=tf.bool)

class Dacca:
    def __init__(self,T=600,recompute=True):
        self.T=T;self.recompute=recompute
        self.cov=tf.constant(DATA['covars'][:20*T].reshape(T,20,9),tf.float32)
        self.dt=tf.constant(DATA['dt'][:20*T].reshape(T,20),tf.float32)
        self.y=tf.constant(np.r_[0.,DATA['ys'][:T]].astype(np.float32))
        self.pop0=float(DATA['covars'][0,2])

    def initial(self,theta):
        iv=tf.gather(theta,[P[n] for n in ['S_0','I_0','Y_0','R1_0','R2_0','R3_0']],axis=-1)
        iv=self.pop0*iv/tf.reduce_sum(iv,-1,keepdims=True)
        s,i,y,r1,r2,r3=tf.unstack(iv,axis=-1);z=tf.zeros_like(s)
        return tf.stack([s,i,y,z,r1,r2,r3,z],-1)

    def interval(self,state,theta,t,seed):
        noise=tf.random.stateless_normal(tf.concat([tf.shape(state)[:-1],[20]],0),seed=seed,dtype=state.dtype)
        cov=tf.gather(self.cov,t);dt=tf.gather(self.dt,t)
        if self.recompute:
            return tf.recompute_grad(self._interval)(state,theta,noise,cov,dt)
        return self._interval(state,theta,noise,cov,dt)

    def _interval(self,state,theta,noise,cov,dtall):
        get=lambda n:theta[...,P[n]]
        beta=tf.exp(get('beta_trend')[...,None]*cov[:,0]+tf.einsum('...i,ki->...k',theta[...,6:12],cov[:,3:9]))
        omega=tf.exp(tf.einsum('...i,ki->...k',theta[...,22:28],cov[:,3:9]))
        S,I,Y,_,R1,R2,R3,count=tf.unstack(state,axis=-1)
        Mn=tf.zeros_like(S)
        for k in range(20):
            dt=dtall[k];pop=cov[k,2];dpop=cov[k,1]
            dw=noise[...,k]*tf.sqrt(dt)
            inf=(omega[...,k]+(beta[...,k]+get('sigma')*dw/dt)*(I/pop)**get('alpha'))*S
            a,b,c,d=get('gamma')*I,3*get('epsilon')*R1,3*get('epsilon')*R2,3*get('epsilon')*R3
            disease=get('m')*I;delta=get('delta')
            ns=S+(dpop+delta*pop-inf-delta*S+d+get('rho')*Y)*dt
            ni=I+(get('c')*inf-disease-delta*I-a)*dt
            ny=Y+((1-get('c'))*inf-delta*Y-get('rho')*Y)*dt
            n1=R1+(a-b-delta*R1)*dt;n2=R2+(b-c-delta*R2)*dt;n3=R3+(c-d-delta*R3)*dt
            nm=Mn+disease*dt
            count+=tf.cast((ns<0)|(ni<0)|(ny<0)|(nm<0)|(n1<0)|(n2<0)|(n3<0),count.dtype)
            S,I,Y,Mn,R1,R2,R3=[tf.maximum(x,0) for x in [ns,ni,ny,nm,n1,n2,n3]]
        return tf.stack([S,I,Y,Mn,R1,R2,R3,count],-1)

    def observation(self,state,theta,y):
        deaths,count=state[...,3],state[...,7]
        v=theta[...,P['tau']]*deaths
        good=tf.math.is_finite(v)&(count<=0)
        mu=tf.where(good,deaths,tf.ones_like(deaths));scale=tf.where(good,v,tf.ones_like(v))+1e-18
        z=tf.clip_by_value((y-mu)/scale,-1e10,1e10)
        lp=-.5*z*z-tf.math.log(scale)-.5*math.log(2*math.pi)
        lp=tf.reduce_logsumexp(tf.stack([lp,tf.fill(tf.shape(lp),tf.cast(LTOL,lp.dtype))]),axis=0)
        return tf.where(good,lp,tf.cast(LTOL,lp.dtype))

    def likelihood(self,e,J,seed,method='native',alpha=.97,return_increments=False):
        theta=natural(e);prior=tfd.Independent(tfd.Deterministic(self.initial(theta)),1)
        observations=tf.broadcast_to(self.y[:,None],[self.T+1,tf.shape(e)[0]])
        def transition(step,state):return Transition(self,state,theta,step)
        def observation(step,state):return Observation(self,state,theta,step)
        def execute():
            return tfp.experimental.mcmc.particle_filter(
                observations,prior,transition,observation,num_particles=J,
                resample_criterion_fn=always_resample,
                unbiased_gradients=(method!='naive'),
                trace_fn=lambda state,result:result.incremental_log_marginal_likelihood,
                seed=seed)
        if method=='discounted':
            with discounted_weights(alpha):increments=execute()
        elif method in ['native','naive']:increments=execute()
        else:raise ValueError(method)
        tf.debugging.assert_equal(tf.shape(increments)[0],self.T+1)
        tf.debugging.assert_near(increments[0],tf.zeros_like(increments[0]),atol=1e-6)
        if return_increments:return increments
        return tf.reduce_sum(increments,0)

class Transition(tfd.Distribution):
    def __init__(self,model,state,theta,t):
        self.model=model;self.state=state;self.theta=theta;self.t=t
        super().__init__(dtype=state.dtype,reparameterization_type=tfd.FULLY_REPARAMETERIZED,
            validate_args=False,allow_nan_stats=False,parameters={},name='DaccaMonthlySimulator')
    def _batch_shape(self):return self.state.shape[:-1]
    def _batch_shape_tensor(self):return tf.shape(self.state)[:-1]
    def _event_shape(self):return tf.TensorShape([8])
    def _event_shape_tensor(self):return tf.constant([8],tf.int32)
    def _sample_n(self,n,seed=None):
        state=tf.broadcast_to(self.state,tf.concat([[n],tf.shape(self.state)],0))
        return self.model.interval(state,self.theta,self.t,seed)
    def _log_prob(self,value):
        raise NotImplementedError('Dacca monthly transition density is unavailable; bootstrap PF must not call it')

class Observation(tfd.Distribution):
    """Original published observation log potential, including numerical floor."""
    def __init__(self,model,state,theta,step):
        self.model=model;self.state=state;self.theta=theta;self.step=step
        super().__init__(dtype=state.dtype,reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=False,allow_nan_stats=False,parameters={},name='DaccaObservationPotential')
    def _batch_shape(self):return self.state.shape[:-1]
    def _batch_shape_tensor(self):return tf.shape(self.state)[:-1]
    def _event_shape(self):return tf.TensorShape([])
    def _event_shape_tensor(self):return tf.constant([],tf.int32)
    def _log_prob(self,y):
        return tf.cond(tf.equal(self.step,0),lambda:tf.zeros(tf.shape(self.state)[:-1],self.state.dtype),
            lambda:self.model.observation(self.state,self.theta,y))

def make_pf(model,J,method='native',alpha=.97):
    @tf.function(autograph=False)
    def run(e,seed):return model.likelihood(e,J,seed,method,alpha)
    return run

def make_gradient(model,J,method='native',alpha=.97):
    @tf.function(autograph=False)
    def run(e,seed):
        with tf.GradientTape() as tape:
            tape.watch(e);loss=-model.likelihood(e,J,seed,method,alpha)/model.T
            total=tf.reduce_sum(loss)
        return loss,tape.gradient(total,e)
    return run

def if2(model,e,J,M,cooling=.5,seed=7401,timing=None):
    """Local IF2 extension using TFP's actual native SMC kernel."""
    swarm=tf.broadcast_to(e[None,...],[J,tf.shape(e)[0],len(FREE)])
    init_scale=tf.constant([.16 if n in IVPS else 0. for n in FREE],tf.float32)
    proc_scale=tf.constant([0. if n in IVPS else .02 for n in FREE],tf.float32)
    trace=[e];likelihood=[]
    @tf.function(autograph=False)
    def iteration(swarm,i,key):
        keys=tf.random.experimental.stateless_split(key,2)
        init=swarm+tf.random.stateless_normal(tf.shape(swarm),keys[0])*init_scale*cooling**(i/50.)
        state=model.initial(natural(init))
        wp=tfp.experimental.mcmc.WeightedParticles(particles={'x':state,'p':init},
                log_weights=tf.fill(tf.shape(init)[:-1],-tf.math.log(tf.cast(J,tf.float32))))
        def propose(step,state,seed):
            seeds=tf.random.experimental.stateless_split(seed,2)
            theta=state.particles['p']+tf.random.stateless_normal(tf.shape(init),seeds[0])*proc_scale*cooling**((i+tf.cast(step,tf.float32)/model.T)/50.)
            natural_theta=natural(theta)
            newx=model.interval(state.particles['x'],natural_theta,step,seeds[1])
            logw=state.log_weights+model.observation(newx,natural_theta,tf.gather(model.y,step+1))
            return tfp.experimental.mcmc.WeightedParticles(particles={'x':newx,'p':theta},log_weights=logw)
        final,score=pfmod.sequential_monte_carlo(
            initial_weighted_particles=wp,propose_and_update_log_weights_fn=propose,num_steps=model.T+1,
            resample_fn=tfp.experimental.mcmc.resample_systematic,resample_criterion_fn=always_resample,
            unbiased_gradients=False,trace_fn=lambda s,r:(s.particles['p'],r.accumulated_log_marginal_likelihood),
            trace_criterion_fn=None,seed=keys[1])
        return final,score
    compiled_seconds=0.;execution=[]
    for i in range(M):
        index=tf.constant(float(i));key=tf.constant([seed,i],tf.int32)
        if i==0:
            before=time.perf_counter();iteration.get_concrete_function(swarm,index,key)
            compiled_seconds=time.perf_counter()-before
        before=time.perf_counter();swarm,ll=iteration(swarm,index,key)
        if timing is not None:
            swarm.numpy();ll.numpy();execution.append(time.perf_counter()-before)
        trace.append(tf.reduce_mean(swarm,0));likelihood.append(ll)
    if timing is not None:timing.update(graph_build_seconds=compiled_seconds,execution_seconds=execution)
    return tf.stack(trace),tf.stack(likelihood)

def train(model,e,J,M,method='native',alpha=.97,seed=7501,timing=None):
    coords=[tf.Variable(e[:,i]) for i in range(len(FREE))]
    optimizers=[tf.keras.optimizers.Adam(.001,beta_1=.9,beta_2=.999,epsilon=1e-8) for _ in FREE]
    base=np.array([.05 if n in IVPS or n in ['gamma','beta_trend','sigma','tau'] else .1 for n in FREE])
    warm=np.r_[np.linspace(.1,1,min(10,M)),np.ones(max(0,M-10))]
    eta=warm[:,None]*(.05+.95*.5*(1+np.cos(np.pi*np.arange(M)/M)))[:,None]*base
    grad=make_gradient(model,J,method,alpha);trace=[e];losses=[]
    before=time.perf_counter();grad.get_concrete_function(tf.stack(coords,-1),tf.constant([seed,0],tf.int32))
    graph_seconds=time.perf_counter()-before;executions=[]
    for i in range(M):
        before=time.perf_counter()
        loss,g=grad(tf.stack(coords,-1),tf.constant([seed,i],tf.int32))
        tf.debugging.assert_all_finite(g,'nonfinite gradient')
        for k,opt in enumerate(optimizers):
            opt.learning_rate.assign(float(eta[i,k]));opt.apply_gradients([(g[:,k],coords[k])])
        trace.append(tf.stack(coords,-1));losses.append(loss)
        if timing is not None:
            trace[-1].numpy();executions.append(time.perf_counter()-before)
    if timing is not None:timing.update(graph_build_seconds=graph_seconds,update_seconds=executions,
        optimizer='tf.keras.optimizers.Adam',epsilon_convention='Keras epsilon_hat',
        optimizer_initial_state='fresh zero for each continuation')
    return tf.stack(trace),tf.stack(losses)
