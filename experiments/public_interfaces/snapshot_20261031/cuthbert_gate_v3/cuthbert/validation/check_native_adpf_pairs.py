"""Paired original/public ADPF checks at the twelve planned Dacca starts.

These are likelihood and gradient evaluations, not optimization runs.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dacca_model as model
from cuthbert import filter as run_filter
from cuthbert.smc import dmop, particle_filter as pf
from cuthbertlib.resampling import autodiff, multinomial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--starts', type=Path, required=True)
    parser.add_argument('--particles', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = model.inputs(600)

    def native(q, key):
        obj = pf.build_filter(*model.factory(q), args.particles,
            autodiff.stop_gradient_decorator(multinomial.resampling))
        ik, fk = jax.random.split(key)
        return run_filter(obj, data, obj.init_prepare(key=ik), key=fk).log_normalizing_constant[-1]

    def public(q, key):
        return dmop.log_likelihood(model.factory, q, data, key=key,
            n_particles=args.particles, alpha=1., resampling='multinomial', checkpoint=False)

    native_vg = jax.jit(jax.value_and_grad(native))
    public_vg = jax.jit(jax.value_and_grad(public))
    starts = np.load(args.starts)
    records = []
    for case in range(12):
        sid, replicate = divmod(case, 2)
        q = jnp.asarray(starts[sid], jnp.float32)
        key = jax.random.fold_in(jax.random.key(20261101), case)
        vn, gn = jax.device_get(native_vg(q, key))
        vp, gp = jax.device_get(public_vg(q, key))
        finite = bool(np.isfinite([vn, vp]).all() and np.isfinite(gn).all() and np.isfinite(gp).all())
        passed = finite and bool(np.allclose(vn,vp,rtol=1e-5,atol=1e-4) and np.allclose(gn,gp,rtol=1e-5,atol=1e-4))
        records.append(dict(case=case,start_id=sid,replicate=replicate,passed=passed,
            finite=finite,native_value=float(vn),public_value=float(vp),
            native_gradient=gn.tolist(),public_gradient=gp.tolist(),
            gradients_bitwise_equal=bool(np.array_equal(gn,gp))))
        args.output.write_text(json.dumps(dict(passed=all(x['passed'] for x in records),
            expected_pairs=12,actual_pairs=len(records),J=args.particles,T=600,
            rtol=1e-5,atol=1e-4,backend=jax.default_backend(),jax=jax.__version__,
            checkpoint=False,formal_fit_calls=0,records=records),indent=2)+'\n')
    assert all(x['passed'] for x in records), 'Original/public ADPF comparison failed; see all twelve records'


if __name__ == '__main__':
    main()
