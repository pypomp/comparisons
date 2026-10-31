"""Independent per-filter keys around TFP's native PF, SMC and resampler.

Only the RNG layout is adapted. The original Dacca equations, native TFP
importance weights, stop-gradient correction and resampling algorithm remain.
"""
import tensorflow as tf
import tensorflow_probability as tfp
import model as m

def combined(seed,member):
 seed=tf.random.experimental.stateless_fold_in(seed,member[0])
 return tf.random.experimental.stateless_fold_in(seed,member[1])
def keyed_normal(template,seed,members,last):
 # template axes [..., particles, batch, coordinates]; each member gets its
 # own native stateless stream, independent of batch shape or member order.
 shape=tf.concat([tf.shape(template)[:-2],[last]],0)
 spec=tf.TensorSpec(template.shape[:-2].concatenate([last]),template.dtype)
 values=tf.map_fn(lambda k:tf.random.stateless_normal(shape,combined(seed,k),dtype=template.dtype),members,
                  fn_output_signature=spec,parallel_iterations=12)
 rank=values.shape.rank
 return tf.transpose(values,list(range(1,rank-1))+[0,rank-1])
def resampler(members):
 def sample(log_probs,event_size,sample_shape,seed=None,name=None):
  batch=tf.transpose(log_probs,[1,0])
  # TFP's native PF uses no extra resampling sample dimensions here.
  tf.debugging.assert_equal(tf.size(sample_shape),0)
  values=tf.map_fn(lambda pair:tfp.experimental.mcmc.resample_systematic(
      pair[0],event_size=event_size,sample_shape=(),seed=combined(seed,pair[1])),
      (batch,members),fn_output_signature=tf.TensorSpec([log_probs.shape[0]],tf.int32),parallel_iterations=12)
  return tf.transpose(values,[1,0])
 return sample

class Transition(m.Transition):
 def __init__(self,model,state,theta,t,members):
  self.members=members;super().__init__(model,state,theta,t)
 def _sample_n(self,n,seed=None):
  state=tf.broadcast_to(self.state,tf.concat([[n],tf.shape(self.state)],0))
  noise=keyed_normal(state,seed,self.members,20)
  cov=tf.gather(self.model.cov,self.t);dt=tf.gather(self.model.dt,self.t)
  if self.model.recompute:return tf.recompute_grad(self.model._interval)(state,self.theta,noise,cov,dt)
  return self.model._interval(state,self.theta,noise,cov,dt)

class Dacca(m.Dacca):
 def likelihood_keyed(self,e,J,seed,members,method='native',alpha=.97):
  theta=m.natural(e);prior=tfp.distributions.Independent(tfp.distributions.Deterministic(self.initial(theta)),1)
  observations=tf.broadcast_to(self.y[:,None],[self.T+1,tf.shape(e)[0]])
  def transition(step,state):return Transition(self,state,theta,step,members)
  def observation(step,state):return m.Observation(self,state,theta,step)
  def execute():
   return tfp.experimental.mcmc.particle_filter(observations,prior,transition,observation,num_particles=J,
     resample_fn=resampler(members),resample_criterion_fn=m.always_resample,unbiased_gradients=True,
     trace_fn=lambda state,result:result.incremental_log_marginal_likelihood,seed=seed)
  if method=='discounted':
   with m.discounted_weights(alpha):inc=execute()
  else:inc=execute()
  tf.debugging.assert_equal(tf.shape(inc)[0],self.T+1)
  tf.debugging.assert_near(inc[0],tf.zeros_like(inc[0]),atol=1e-6)
  return tf.reduce_sum(inc,axis=0)

def gradient(model,J,method='native',alpha=.97,xla=False):
 @tf.function(autograph=False,jit_compile=xla)
 def run(e,root,members):
  with tf.GradientTape() as tape:
   tape.watch(e);loss=-model.likelihood_keyed(e,J,root,members,method,alpha)/model.T
   total=tf.reduce_sum(loss)
  return loss,tape.gradient(total,e)
 return run

def if2_step(model,J,cooling=.5,xla=False,return_cloud=False):
 init_scale=tf.constant([.16 if n in m.IVPS else 0. for n in m.FREE],tf.float32)
 proc_scale=tf.constant([0. if n in m.IVPS else .02 for n in m.FREE],tf.float32)
 @tf.function(autograph=False,jit_compile=xla)
 def run(swarm,index,root,members):
  keys=tf.random.experimental.stateless_split(root,2)
  p=swarm+keyed_normal(swarm,keys[0],members,len(m.FREE))*init_scale*cooling**(index/50.)
  initial=tfp.experimental.mcmc.WeightedParticles({'x':model.initial(m.natural(p)),'p':p},
      tf.fill(tf.shape(p)[:-1],-tf.math.log(tf.cast(J,tf.float32))))
  def propose(month,state,seed):
   ks=tf.random.experimental.stateless_split(seed,2)
   newp=state.particles['p']+keyed_normal(state.particles['p'],ks[0],members,len(m.FREE))*proc_scale*cooling**((index+tf.cast(month,tf.float32)/model.T)/50.)
   natural=m.natural(newp);oldx=state.particles['x'];noise=keyed_normal(oldx,ks[1],members,20)
   newx=model._interval(oldx,natural,noise,tf.gather(model.cov,month),tf.gather(model.dt,month))
   weights=state.log_weights+model.observation(newx,natural,tf.gather(model.y,month+1))
   return tfp.experimental.mcmc.WeightedParticles({'x':newx,'p':newp},weights)
  final,ll=m.pfmod.sequential_monte_carlo(initial,propose,num_steps=model.T+1,
      resample_fn=resampler(members),resample_criterion_fn=m.always_resample,unbiased_gradients=False,
      trace_fn=lambda s,r:(s.particles['p'],r.accumulated_log_marginal_likelihood),trace_criterion_fn=None,seed=keys[1])
  return (final if return_cloud else tf.reduce_mean(final,axis=0)),ll
 return run


def particle_filter(model,J,xla=False):
 @tf.function(autograph=False,jit_compile=xla)
 def run(e,root,members):
  return model.likelihood_keyed(e,J,root,members,method='native')
 return run
