from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'source/candidate'),str(ROOT/'source')]
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import model as m, keyed, public_api as api
model=keyed.Dacca(T=3)
p=tf.constant(np.repeat(m.to_est(m.DATA['theta_published']).reshape(1,23),2,0));members=tf.constant([[0,0],[0,1]])
key=tf.constant([24881007,0]);cloud=tf.broadcast_to(p[None],[8,2,23])
a=api.pf_kernel(model,8,pristine=True,differentiate=True)(p,key,members)
b=api.pf_kernel(model,8,alpha=1.,differentiate=True)(p,key,members)
for x,y in zip(a,b):np.testing.assert_array_equal(x.numpy(),y.numpy())
new,ll=api.if2_kernel(model,8)(cloud,tf.constant(0.),key,members)
assert np.isfinite(new.numpy()).all() and np.isfinite(ll.numpy()).all()
rates=tf.constant(np.load(ROOT/'source/input/learning_rates_400.npy')[0])
engine=api.Adam(model,p,8,1.);out=engine.step(key,members,rates)
assert all(np.isfinite(x.numpy()).all() for x in out)
summary={'passed':True,'T':3,'J':8,'batch':2,'PF_alpha1_exact':True,'gradient_alpha1_exact':True,'IF2_public_finite':True,'Adam_public_finite':True,'TFP_path':tfp.__file__,'purpose':'driver smoke, not full Dacca correctness or GPU evidence'}
(ROOT/'local_smoke.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
