"""Use the unchanged BMModel with IF2, then an explicitly external optimizer.

Three updates demonstrate each interface; this is not an accuracy comparison.
Optax is needed only to run this example, not to import the extensions.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pfjax as pf
from pfjax.models import BMModel


def main():
    model = BMModel(dt=0.5, unconstrained_theta=True)
    true_theta = jnp.array([0.02, -0.8, -1.2])
    y, _ = pf.simulate(model, jax.random.PRNGKey(100), 12, jnp.asarray(0.0), true_theta)
    initial = jnp.array([0.1, -0.5, -0.5])
    population = jnp.broadcast_to(initial, (24, 3))
    warm = jax.jit(
        lambda p: pf.iterated_filter(
            model, jax.random.PRNGKey(200), y, p, 0.04, 3, initial_scale=0.04
        )
    )(population)
    if not np.all(warm["history"]["valid"]):
        raise RuntimeError("IF2 returned invalid values")
    results = {}
    for name, start in [
        ("original_RB_score", initial),
        ("IF2_then_RB_score", warm["theta"]),
        ("IF2_then_added_MOP", warm["theta"]),
    ]:
        optimizer = optax.adam(0.001)
        theta, state = start, optimizer.init(start)
        if name.endswith("MOP"):
            get_gradient = jax.jit(
                jax.grad(
                    lambda p, k: pf.particle_filter_mop(model, k, y, p, 24, alpha=0.97)[
                        "loglik"
                    ]
                )
            )
        else:
            get_gradient = jax.jit(
                lambda p, k: pf.particle_filter_rb(model, k, y, p, 24, score=True)[
                    "score"
                ]
            )
        for i in range(3):
            gradient = get_gradient(
                theta, jax.random.fold_in(jax.random.PRNGKey(300), i)
            )
            if not np.all(np.isfinite(gradient)):
                raise RuntimeError("nonfinite gradient in " + name)
            updates, state = optimizer.update(-gradient / len(y), state, theta)
            theta = optax.apply_updates(theta, updates)
        score_keys = jax.random.split(jax.random.PRNGKey(400), 4)
        raw = jax.jit(
            jax.vmap(lambda k: pf.particle_filter(model, k, y, theta, 24)["loglik"])
        )(score_keys)
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("nonfinite evaluation")
        results[name] = {
            "theta": np.asarray(theta).tolist(),
            "raw_evaluation": np.asarray(raw).tolist(),
        }
    print(
        json.dumps(
            {
                "completed": True,
                "package": pf.__file__,
                "model": "unchanged PFJAX BMModel",
                "optimizer": "external Optax Adam",
                "IF2_passes": 3,
                "updates_per_example": 3,
                "results": results,
                "accuracy_comparison": False,
            }
        )
    )


if __name__ == "__main__":
    main()
