"""Compare native and public alpha=1 adjoints without changing tolerances."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import dacca_model as model
from cuthbert import filter as run_filter
from cuthbert.smc import dmop, particle_filter as pf
from cuthbertlib.resampling import autodiff, multinomial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = model.inputs(600)
    p = jnp.asarray(model.PUBLISHED, dtype=jnp.float32)
    key = jax.random.split(jax.random.key(20261030), 3)[0]

    def native(p, checkpoint=False):
        obj = pf.build_filter(*model.factory(p), 32,
            autodiff.stop_gradient_decorator(multinomial.resampling))
        if checkpoint:
            obj = obj._replace(filter_combine=jax.checkpoint(obj.filter_combine))
        ik, fk = jax.random.split(key)
        return run_filter(obj, inputs, obj.init_prepare(key=ik), key=fk).log_normalizing_constant[-1]

    variants = {'native': native, 'native_repeat': lambda p: native(p),
                'native_checkpoint_combine': lambda p: native(p, True)}
    for checkpoint in (False, True):
        variants[f'public_checkpoint_{checkpoint}'] = lambda p, c=checkpoint: dmop.log_likelihood(
            model.factory, p, inputs, key=key, n_particles=32, alpha=1.,
            resampling='multinomial', checkpoint=c)
    arrays = {}
    records = {}
    for name, fn in variants.items():
        start = time.monotonic()
        value, grad = jax.jit(jax.value_and_grad(fn))(p)
        value, grad = np.asarray(value), np.asarray(grad)
        arrays[name+'_gradient'] = grad
        arrays[name+'_value'] = value
        reference = arrays['native_gradient']
        difference = np.abs(grad-reference)
        tolerance = 1e-4 + 1e-5*np.abs(reference)
        records[name] = dict(value=float(value), gradient=grad.tolist(),
            difference=(grad-reference).tolist(), exceeds_tolerance=np.flatnonzero(difference>tolerance).tolist(),
            max_abs=float(difference.max()), seconds=time.monotonic()-start)
        (args.output/'diagnostic.json').write_text(json.dumps(dict(
            backend=jax.default_backend(), jax=jax.__version__, rtol=1e-5, atol=1e-4,
            J=32,T=600,resampling='multinomial', records=records), indent=2)+'\n')
        print(name, records[name]['exceeds_tolerance'], records[name]['max_abs'], flush=True)
    np.savez_compressed(args.output/'gradients.npz', **arrays)


if __name__ == '__main__':
    main()
