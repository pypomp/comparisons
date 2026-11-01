"""Short executable IF2 example, not an optimization-quality comparison.

Run from a checkout with that checkout on PYTHONPATH. The original StochVol
class and its PX0/PX/PY methods are used without changes. All parameters below
are in estimation coordinates (mu, atanh(rho), log(sigma)).
"""
import json

import numpy as np
import particles
from particles import state_space_models as ssms
from particles.iterated_filtering import iterated_filtering


def model_factory(theta):
    return ssms.StochVol(mu=theta[..., 0], rho=np.tanh(theta[..., 1]),
                        sigma=np.exp(theta[..., 2]))


def main():
    np.random.seed(30)
    _, data = ssms.StochVol().simulate(50)
    theta = np.array([-0.5, np.arctanh(0.7), np.log(0.3)])
    np.random.seed(31)
    result = iterated_filtering(model_factory, data, np.tile(theta, (100, 1)),
                                scales=[0.1, 0.05, 0.05], niter=3, cooling=0.9,
                                vectorized=True)
    # Fresh, unperturbed PF calls. These two individual values do not establish
    # precision or improvement and are not used to choose the fitted endpoint.
    scores = []
    for seed, point in [(32, theta), (33, result.means[-1])]:
        np.random.seed(seed)
        pf = particles.SMC(fk=ssms.Bootstrap(ssm=model_factory(point), data=data),
                           N=200, collect="off")
        pf.run()
        scores.append(float(pf.logLt))
    print(json.dumps({"completed": True, "niter": 3, "T": 50, "N": 100,
        "parameters": result.means.tolist(), "perturbed_loglik": result.loglik.tolist(),
        "single_unperturbed_PF_values": scores,
        "optimization_quality_or_speed_claimed": False}))


if __name__ == "__main__":
    main()
