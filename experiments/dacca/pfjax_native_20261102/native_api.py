"""Dacca model callbacks and the unmodified upstream basic PF objective."""
import jax
import pfjax
from pfjax.particle_resamplers import resample_multinomial
import dacca_model as m

class Dacca(pfjax.BaseModel):
    def __init__(self):
        super().__init__(bootstrap=True)

    def pf_init(self, key, y_init, theta):
        init, prop, pot = m.factory(theta)
        x0 = init(key)
        x = prop(key, x0, y_init)
        return x, pot(x0, x, y_init)

    def pf_step(self, key, x_prev, y_curr, theta):
        _, prop, pot = m.factory(theta)
        x = jax.checkpoint(prop)(key, x_prev, y_curr)
        return x, pot(x_prev, x, y_curr)

    def meas_lpdf(self, y_curr, x_curr, theta):
        return m.observation(m.from_est(theta), x_curr, y_curr['y'])

def objective(J=5000,T=600):
    model=Dacca();data=m.inputs(T)
    def loss(p,key):
        return -pfjax.particle_filter(model,key,data,p,J,
            resampler=resample_multinomial,score=False,fisher=False,history=False)['loglik']/T
    return loss
