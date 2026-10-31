"""PyDPF optimizer sessions with complete state and per-row validity flags."""
import copy
import numpy as np
import torch
import pydpf
import dacca_model as m

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

class IF2Session:
    kind='IF2'
    def __init__(self,model,p,J,cooling,seed):
        self.model,self.J,self.a,self.seed=model,J,cooling,seed
        self.swarm=p[:,None,:].expand(p.shape[0],J,len(m.FREE)).clone()
        self.iteration=0;self.parameters=p.clone()
        self.initial=p.new_tensor([.16 if n in m.IVPS else 0 for n in m.FREE])
        self.process=p.new_tensor([0 if n in m.IVPS else .02 for n in m.FREE])

    def step(self):
        model=self.model;J=self.J;B=self.swarm.shape[0];i=self.iteration;a=self.a
        model.generator.manual_seed(self.seed+i)
        noise=lambda:torch.randn(self.swarm.shape,device=model.device,generator=model.generator)
        with torch.no_grad():
            initial=self.swarm+noise()*self.initial*a**(i/50)
            session=self
            class Prior(pydpf.Module):
                def sample(self,batch_size,n_particles,**kw):
                    state=model.initial(m.natural(initial))
                    p=initial+noise()*session.process*a**(i/50)
                    return torch.cat([model.interval(state,m.natural(p),0),p],-1)
            class Dyn(pydpf.Module):
                def sample(self,prev_state,t,**kw):
                    p=prev_state[...,8:]+noise()*session.process*a**((i+t/model.T)/50)
                    return torch.cat([model.interval(prev_state[...,:8],m.natural(p),t),p],-1)
            class Obs(pydpf.Module):
                def score(self,state,observation,**kw):
                    return model.observation(state[...,:8],m.natural(state[...,8:]),observation)
            pf=pydpf.ParticleFilter(resampler=pydpf.SystematicResampler(model.generator),
                SSM=pydpf.FilteringModel(prior_model=Prior(),dynamic_model=Dyn(),observation_model=Obs()))
            last=m.LastState()
            factors=pf(n_particles=J,time_extent=model.T-1,aggregation_function=last,
                       observation=model.y.expand(model.T,B,1))
            if factors.shape[0]!=model.T:raise RuntimeError('Incomplete IF2 trajectory')
            state,_=pydpf.SystematicResampler(model.generator)(last.state,last.weight)
            proposal=state[...,8:];values=factors.sum(0).reshape(B)
            good=torch.isfinite(proposal).all((1,2))&torch.isfinite(values)
            self.swarm=torch.where(good[:,None,None],proposal,self.swarm)
            self.parameters=self.swarm.mean(1);self.iteration+=1
            return values.detach(),good.detach(),['ok' if g else 'nonfinite IF2 state or value' for g in good.tolist()]

    def state_dict(self):
        return cpu(dict(kind=self.kind,J=self.J,cooling=self.a,seed=self.seed,iteration=self.iteration,
                        parameters=self.parameters,swarm=self.swarm,generator_state=self.model.generator.get_state()))

class AdamSession:
    kind='Adam'
    def __init__(self,model,p,J,method,alpha,seed,iterations):
        self.model,self.J,self.method,self.alpha,self.seed=model,J,method,alpha,seed
        self.iterations=iterations;self.iteration=0
        self.coords=[torch.nn.Parameter(p[:,i].clone()) for i in range(p.shape[1])]
        self.opt=torch.optim.Adam([{'params':[v],'lr':.1} for v in self.coords],betas=(.9,.999),eps=1e-8)
        self.parameters=p.clone()
        self.base=np.array([.05 if n in m.IVPS or n in ['gamma','beta_trend','sigma','tau'] else .1 for n in m.FREE])

    def step(self):
        self.opt.zero_grad(set_to_none=True)
        before=torch.stack(self.coords,-1).detach().clone()
        loss=-self.model.likelihood(torch.stack(self.coords,-1),self.J,
                                    self.seed+self.iteration,self.method,self.alpha)/self.model.T
        good_loss=torch.isfinite(loss)
        objective=torch.where(good_loss,loss,torch.zeros_like(loss)).sum()
        gradients=torch.autograd.grad(objective,self.coords,allow_unused=True)
        gradients=[torch.zeros_like(p) if g is None else g for p,g in zip(self.coords,gradients)]
        good_grad=torch.stack([torch.isfinite(g) for g in gradients],-1).all(-1)
        good=good_loss&good_grad
        for p,g in zip(self.coords,gradients):p.grad=torch.where(good,g,torch.zeros_like(g))
        warmup=min(10,self.iterations)
        warm=.1 if warmup==1 else .1+.9*min(self.iteration,warmup-1)/(warmup-1)
        factor=warm*(.05+.95*.5*(1+np.cos(np.pi*self.iteration/self.iterations)))
        for i,g in enumerate(self.opt.param_groups):g['lr']=float(self.base[i]*factor)
        self.opt.step()
        after=torch.stack(self.coords,-1).detach()
        good_parameter=torch.isfinite(after).all(-1);good=good&good_parameter
        with torch.no_grad():
            for i,p in enumerate(self.coords):p.copy_(torch.where(good,p,before[:,i]))
        self.parameters=torch.stack(self.coords,-1).detach().clone();self.iteration+=1
        reasons=[]
        for a,b,c in zip(good_loss.tolist(),good_grad.tolist(),good_parameter.tolist()):
            reasons.append('nonfinite Adam loss' if not a else 'nonfinite Adam gradient' if not b else 'nonfinite Adam parameters' if not c else 'ok')
        return loss.detach(),good.detach(),reasons

    def state_dict(self):
        return cpu(dict(kind=self.kind,J=self.J,method=self.method,alpha=self.alpha,seed=self.seed,
            iterations=self.iterations,iteration=self.iteration,parameters=self.parameters,
            optimizer=self.opt.state_dict(),generator_state=self.model.generator.get_state()))

def restore(model,state):
    p=state['parameters'].to(model.device)
    if state['kind']=='IF2':
        result=IF2Session(model,p,state['J'],state['cooling'],state['seed'])
        result.swarm=state['swarm'].to(model.device)
    else:
        result=AdamSession(model,p,state['J'],state['method'],state['alpha'],state['seed'],state['iterations'])
        result.opt.load_state_dict(state['optimizer'])
    result.iteration=state['iteration'];result.parameters=p.clone()
    model.generator.set_state(state['generator_state'])
    return result

def select_state(state,indices):
    state=cpu(state);B=state['parameters'].shape[0]
    idx=torch.tensor(indices,dtype=torch.long)
    state['parameters']=state['parameters'][idx]
    if state['kind']=='IF2':state['swarm']=state['swarm'][idx]
    else:
        for values in state['optimizer']['state'].values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor) and value.ndim>0 and value.shape[0]==B:values[key]=value[idx]
    return state

def combine_states(states,base_seed):
    assert states and len({s['iteration'] for s in states})==1
    result=cpu(states[0]);result['seed']=base_seed
    result['parameters']=torch.cat([s['parameters'] for s in states])
    if result['kind']=='IF2':result['swarm']=torch.cat([s['swarm'] for s in states])
    else:
        for key,values in result['optimizer']['state'].items():
            for name,value in values.items():
                if isinstance(value,torch.Tensor) and value.ndim>0:
                    values[name]=torch.cat([s['optimizer']['state'][key][name] for s in states])
                elif isinstance(value,torch.Tensor):
                    assert all(torch.equal(value,s['optimizer']['state'][key][name]) for s in states)
    return result
