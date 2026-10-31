"""Dacca callbacks for the unmodified PyDPF 1.2.3 ParticleFilter interface.

Derived from quant 4e1cb811's Dacca adapter. The 23 free coordinates and data
match the pinned DMOP paper experiment. Only the discounted resampler and
the perturbed-parameter IF2 model below are new experimental extensions.
"""
from pathlib import Path
import math
import numpy as np
import torch
import pydpf
from torch.utils.checkpoint import checkpoint
from pydpf.utils import batched_select

ROOT = Path(__file__).resolve().parent
DATA = dict(np.load(ROOT / 'input/frozen.npz'))
NAMES = DATA['paramnames'].tolist()
P = {n:i for i,n in enumerate(NAMES)}
FIXED = ['rho','c','alpha','delta','Y_0']
FREE = [n for n in NAMES if n not in FIXED]
IVPS = ['S_0','I_0','R1_0','R2_0','R3_0']
IVP = [P[n] for n in ['S_0','I_0','Y_0','R1_0','R2_0','R3_0']]
IVPF = [FREE.index(n) for n in IVPS]
POSITIVE = ['gamma','epsilon','m','sigma','tau']
BS = [P[f'bs{i}'] for i in range(1,7)]
OM = [P[f'omegas{i}'] for i in range(1,7)]
LTOL = math.log(1e-18)

def to_est(theta):
    theta = np.asarray(theta, dtype=np.float64)
    e = theta[..., [P[n] for n in FREE]].copy()
    for n in POSITIVE:
        e[...,FREE.index(n)] = np.log(theta[...,P[n]])
    e[...,FREE.index('beta_trend')] *= 100
    iv = theta[..., [P[n] for n in IVPS]]
    e[...,IVPF] = np.log(iv / iv.sum(-1,keepdims=True))
    return e.astype(np.float32)

def natural(e):
    v = {n:e[...,i] for i,n in enumerate(FREE)}
    for n in POSITIVE:
        v[n] = torch.exp(v[n])
    v['beta_trend'] = v['beta_trend']/100
    props = torch.exp(e[...,IVPF])
    props = props/props.sum(-1,keepdim=True)
    v.update(zip(IVPS,props.unbind(-1)))
    z = torch.zeros_like(e[...,0])
    v.update(rho=z,c=z+1,alpha=z+1,delta=z+.02,Y_0=z)
    return torch.stack([v[n] for n in NAMES],-1)

class Dacca:
    def __init__(self, device='cpu', T=600, checkpoint_intervals=False):
        self.device = torch.device(device)
        self.T = T
        self.checkpoint_intervals = checkpoint_intervals
        self.generator = torch.Generator(device=self.device)
        self.cov = torch.tensor(DATA['covars'][:20*T].reshape(T,20,9),device=self.device,dtype=torch.float32)
        self.dt = DATA['dt'][:20*T].reshape(T,20).astype(np.float64).tolist()
        self.dpop = DATA['covars'][:20*T,1].reshape(T,20).astype(np.float64).tolist()
        self.pop = DATA['covars'][:20*T,2].reshape(T,20).astype(np.float64).tolist()
        self.pop0 = float(DATA['covars'][0,2])
        self.y = torch.tensor(DATA['ys'][:T],device=self.device,dtype=torch.float32).reshape(T,1,1)

    def initial(self, theta):
        iv = theta[...,IVP]
        S,I,Y,R1,R2,R3 = (self.pop0*iv/iv.sum(-1,keepdim=True)).unbind(-1)
        z = torch.zeros_like(S)
        return torch.stack([S,I,Y,z,R1,R2,R3,z],-1)

    def interval(self, state, theta, t):
        noise = torch.randn((*state.shape[:-1],20),device=self.device,generator=self.generator)
        if self.checkpoint_intervals and torch.is_grad_enabled():
            return checkpoint(self._interval,state,theta,noise,t,use_reentrant=False,preserve_rng_state=False)
        return self._interval(state,theta,noise,t)

    def _interval(self,state,theta,noise,t):
        if theta.ndim==2:
            theta=theta.unsqueeze(1)
        get=lambda n:theta[...,P[n]]
        gamma,m,rho,c=get('gamma'),get('m'),get('rho'),get('c')
        sigma,alpha,delta,neps=get('sigma'),get('alpha'),get('delta'),3*get('epsilon')
        cov=self.cov[t]
        beta=torch.exp(get('beta_trend')[...,None]*cov[:,0] + theta[...,BS] @ cov[:,3:9].T)
        omega=torch.exp(theta[...,OM] @ cov[:,3:9].T)
        S,I,Y,_,R1,R2,R3,cnt=state.unbind(-1)
        Mn=torch.zeros_like(S)
        for k in range(20):
            dt,dpopdt,pop=self.dt[t][k],self.dpop[t][k],self.pop[t][k]
            dw=noise[...,k]*math.sqrt(dt)
            effI=(I/pop)**alpha
            inf=(omega[...,k]+(beta[...,k]+sigma*dw/dt)*effI)*S
            pass0,pass1,pass2,pass3=gamma*I,neps*R1,neps*R2,neps*R3
            disease=m*I
            ns=S+(dpopdt+delta*pop-inf-delta*S+pass3+rho*Y)*dt
            ni=I+(c*inf-disease-delta*I-pass0)*dt
            ny=Y+((1-c)*inf-delta*Y-rho*Y)*dt
            n1=R1+(pass0-pass1-delta*R1)*dt
            n2=R2+(pass1-pass2-delta*R2)*dt
            n3=R3+(pass2-pass3-delta*R3)*dt
            nm=Mn+disease*dt
            neg=(ns<0)|(ni<0)|(ny<0)|(nm<0)|(n1<0)|(n2<0)|(n3<0)
            cnt=cnt+neg
            S,I,Y=ns.clamp_min(0),ni.clamp_min(0),ny.clamp_min(0)
            R1,R2,R3,Mn=n1.clamp_min(0),n2.clamp_min(0),n3.clamp_min(0),nm.clamp_min(0)
        return torch.stack([S,I,Y,Mn,R1,R2,R3,cnt],-1)

    def observation(self,state,theta,y):
        if theta.ndim==2:
            theta=theta.unsqueeze(1)
        Mn,cnt=state[...,3],state[...,7]
        v=theta[...,P['tau']]*Mn
        valid=torch.isfinite(v)&(cnt<=0)
        mu=torch.where(valid,Mn,torch.ones_like(Mn))
        sc=torch.where(valid,v,torch.ones_like(v))+1e-18
        z=((y[:,0:1]-mu)/sc).clamp(-1e10,1e10)
        lp=-.5*z*z-torch.log(sc)-.5*math.log(2*math.pi)
        ll=torch.logaddexp(lp,torch.full_like(lp,LTOL))
        return torch.where(valid,ll,torch.full_like(ll,LTOL))

    def ssm(self):
        model=self
        class Prior(pydpf.Module):
            def sample(self,batch_size,n_particles,series_metadata,**kw):
                x=model.initial(series_metadata).unsqueeze(1).expand(batch_size,n_particles,8)
                return model.interval(x,series_metadata,0)
        class Dyn(pydpf.Module):
            def sample(self,prev_state,t,series_metadata,**kw):
                return model.interval(prev_state,series_metadata,t)
        class Obs(pydpf.Module):
            def score(self,state,observation,series_metadata,**kw):
                return model.observation(state,series_metadata,observation)
        return pydpf.FilteringModel(dynamic_model=Dyn(),observation_model=Obs(),prior_model=Prior())

    def filter(self, method='native', alpha=.97):
        if method=='native':
            return pydpf.StopGradientDPF(SSM=self.ssm(),resampling_generator=self.generator)
        base=pydpf.SystematicResampler(self.generator)
        if method=='discounted':
            base=DiscountedResampler(base,alpha)
        elif method!='pf':
            raise ValueError(method)
        return pydpf.ParticleFilter(resampler=base,SSM=self.ssm())

    def likelihood(self,e,J,seed,method='native',alpha=.97):
        self.generator.manual_seed(seed)
        p=natural(e)
        out=self.filter(method,alpha)(n_particles=J,time_extent=self.T-1,
            aggregation_function=pydpf.LogLikelihoodFactors(),
            observation=self.y.expand(self.T,p.shape[0],1),series_metadata=p)
        if out.shape[0]!=self.T:
            raise RuntimeError(f'Incomplete PyDPF trajectory: {out.shape[0]}/{self.T}')
        return out.sum(0).reshape(p.shape[0])

class DiscountedResampler(pydpf.Module):
    """Experimental alpha extension; alpha=1 is tested against native PyDPF."""
    def __init__(self,base,alpha):
        super().__init__()
        self.base_resampler=base
        self.alpha=float(alpha)
        self.cache={}
        self._need_weight_output=True
    def forward(self,state,weight,**data):
        self.base_resampler._need_weight_output=True
        new_state,new_weight=self.base_resampler(state,weight)
        self.cache=self.base_resampler.cache
        if torch.is_grad_enabled():
            selected=batched_select(self.cache['used_weight'],self.cache['sampled_indices'])
            new_weight=new_weight+self.alpha*(selected-selected.detach())
        return new_state,new_weight

class LastState(pydpf.Module):
    def forward(self,state,weight,likelihood,**kw):
        self.state=state
        self.weight=weight
        return likelihood

def if2(model,e,J,M,cooling=.5,seed=741):
    """New IF2 extension: PyDPF performs each augmented-state PF pass."""
    model.generator.manual_seed(seed)
    B=e.shape[0]
    swarm=e[:,None,:].expand(B,J,len(FREE)).clone()
    init_scale=e.new_tensor([.16 if n in IVPS else 0. for n in FREE])
    proc_scale=e.new_tensor([0. if n in IVPS else .02 for n in FREE])
    trace=[e.clone()]
    ll=[]
    with torch.no_grad():
        for iteration in range(M):
            init_swarm=swarm+torch.randn(swarm.shape,device=model.device,generator=model.generator)*init_scale*cooling**(iteration/50)
            class Prior(pydpf.Module):
                def sample(self,batch_size,n_particles,**kw):
                    state=model.initial(natural(init_swarm))
                    p=init_swarm+torch.randn(swarm.shape,device=model.device,generator=model.generator)*proc_scale*cooling**(iteration/50)
                    state=model.interval(state,natural(p),0)
                    return torch.cat([state,p],-1)
            class Dyn(pydpf.Module):
                def sample(self,prev_state,t,**kw):
                    p=prev_state[...,8:]+torch.randn(swarm.shape,device=model.device,generator=model.generator)*proc_scale*cooling**((iteration+t/model.T)/50)
                    return torch.cat([model.interval(prev_state[...,:8],natural(p),t),p],-1)
            class Obs(pydpf.Module):
                def score(self,state,observation,**kw):
                    return model.observation(state[...,:8],natural(state[...,8:]),observation)
            pf=pydpf.ParticleFilter(resampler=pydpf.SystematicResampler(model.generator),
                SSM=pydpf.FilteringModel(dynamic_model=Dyn(),observation_model=Obs(),prior_model=Prior()))
            last=LastState()
            out=pf(n_particles=J,time_extent=model.T-1,aggregation_function=last,observation=model.y.expand(model.T,B,1))
            if out.shape[0]!=model.T:
                raise RuntimeError('Incomplete IF2 pass')
            final,_=pydpf.SystematicResampler(model.generator)(last.state,last.weight)
            swarm=final[...,8:]
            trace.append(swarm.mean(1))
            ll.append(out.sum(0).reshape(B))
    return torch.stack(trace),torch.stack(ll)

def learning_rates(M,device):
    base=np.array([.05 if n in IVPS or n in ['gamma','beta_trend','sigma','tau'] else .1 for n in FREE])
    warm=np.r_[np.linspace(.1,1,min(10,M)),np.ones(max(0,M-10))]
    decay=.05+.95*.5*(1+np.cos(np.pi*np.arange(M)/M))
    return torch.tensor(warm[:,None]*decay[:,None]*base,device=device,dtype=torch.float32)

def train(model,e,J,M,method='native',alpha=.97,beta1=.9,seed=751):
    # Each coordinate has its own paper learning rate; torch Adam owns updates.
    coordinates=[torch.nn.Parameter(e[:,i].clone()) for i in range(e.shape[1])]
    rates=learning_rates(M,model.device)
    opt=torch.optim.Adam([{'params':[p],'lr':float(rates[0,i])} for i,p in enumerate(coordinates)],betas=(beta1,.999),eps=1e-8)
    trace=[e.detach().clone()]
    losses=[]
    for iteration in range(M):
        opt.zero_grad(set_to_none=True)
        params=torch.stack(coordinates,-1)
        loss=-model.likelihood(params,J,seed+iteration,method,alpha)/model.T
        if not torch.isfinite(loss).all():
            raise RuntimeError('Nonfinite training loss')
        loss.sum().backward()
        if not all(torch.isfinite(p.grad).all() for p in coordinates):
            raise RuntimeError('Nonfinite gradient')
        for i,g in enumerate(opt.param_groups):
            g['lr']=float(rates[iteration,i])
        opt.step()
        trace.append(torch.stack(coordinates,-1).detach().clone())
        losses.append(loss.detach())
    return torch.stack(trace),torch.stack(losses)
