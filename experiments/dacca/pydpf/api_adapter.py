"""Model-only adapter: original Dacca equations; native/public filter classes."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import torch
import pydpf
import dacca_model as frozen
ROOT=Path(__file__).resolve().parent

def original_package():
    name='_pydpf_original';base=ROOT/'upstream/pydpf/pydpf'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,base/'__init__.py',submodule_search_locations=[str(base)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def seed(root,*keys):return int(np.random.SeedSequence([root,*keys]).generate_state(1)[0])
def factory(model,parameters,package=pydpf):
    theta=frozen.natural(parameters)
    class Prior(package.Module):
        def sample(self,batch_size,n_particles,**data):
            state=model.initial(theta)
            if state.ndim==2:state=state[:,None,:].expand(batch_size,n_particles,-1)
            return model.interval(state,theta,0)
    class Dynamic(package.Module):
        def sample(self,prev_state,t,**data):return model.interval(prev_state,theta,t)
    class Observation(package.Module):
        def fitness(self,state,observation,**data):return model.observation(state,theta,observation)
    return package.FilteringModel(prior_model=Prior(),dynamic_model=Dynamic(),observation_model=Observation())
def factors(model,parameters,J,random_seed,alpha=None,package=pydpf):
    model.generator.manual_seed(random_seed)
    if alpha is None:pf=package.ParticleFilter(SSM=factory(model,parameters,package),resampler=package.SystematicResampler(model.generator))
    else:
        if package is not pydpf and alpha!=1:raise ValueError('Pristine upstream has no discount argument')
        kw={'alpha':alpha} if package is pydpf else {}
        pf=package.StopGradientDPF(SSM=factory(model,parameters,package),resampling_generator=model.generator,**kw)
    pf.update()
    values=pf(n_particles=J,time_extent=model.T-1,aggregation_function=package.LogLikelihoodFactors(),observation=model.y.expand(model.T,parameters.shape[0],1))
    if values.shape[0]!=model.T:raise RuntimeError(f'Incomplete PF trajectory {values.shape[0]}/{model.T}')
    return values

def warm_if2(model,initial,J,M,random_seed):
    model.generator.manual_seed(random_seed)
    return pydpf.IteratedFiltering(lambda p:factory(model,p),generator=model.generator,
        perturbation_scale=initial.new_tensor([0 if n in frozen.IVPS else .02 for n in frozen.FREE]),
        initial_scale=initial.new_tensor([.16 if n in frozen.IVPS else 0 for n in frozen.FREE]),
        cooling_rate=.5,cooling_interval=50,within_pass_cooling=True)(initial,model.y.expand(model.T,initial.shape[0],1),n_particles=J,n_iterations=M)
