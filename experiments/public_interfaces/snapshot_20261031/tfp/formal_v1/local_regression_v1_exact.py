from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'source/candidate'),str(ROOT/'source')]
import numpy as np
import tensorflow as tf
import model as m,keyed,public_api as api
model=keyed.Dacca(T=3);p=np.repeat(m.to_est(m.DATA['theta_published']).reshape(1,23),12,0)
ids=tf.constant([[i//2,i%2] for i in range(12)],tf.int32);key=tf.constant([24881000,4500]);rates=tf.constant(np.load(ROOT/'source/input/learning_rates_400.npy')[0])
results={}
for alpha,method in [(1.,'native'),(.97,'discounted')]:
 a=keyed.gradient(model,8,method,alpha,False)(tf.constant(p),key,ids)
 b=api.gradient_kernel(model,8,alpha)(tf.constant(p),key,ids)
 for x,y in zip(a,b):np.testing.assert_array_equal(x.numpy(),y.numpy())
 results[str(alpha)]={'old_wrapper_loss_and_gradient_exact':True}
normal=api.Adam(model,tf.constant(p),8,1.)
bad=p.copy();bad[3,m.FREE.index('sigma')]=1000.
poison=api.Adam(model,tf.constant(bad),8,1.)
na=[x.numpy() for x in normal.step(key,ids,rates)];ba=[x.numpy() for x in poison.step(key,ids,rates)]
assert na[-1].all() and ba[-1].tolist()==[i!=3 for i in range(12)]
np.testing.assert_array_equal(na[0][np.arange(12)!=3],ba[0][np.arange(12)!=3])
np.testing.assert_array_equal(ba[0][3],bad[3])
B=api.Adam(model,tf.constant(p),8,1.);C=api.Adam(model,tf.constant(p),8,.97)
bs=B.snapshot();cs=C.snapshot();assert all(np.array_equal(bs[k],cs[k]) for k in bs)
keys={};seen={}
for domain,count in [(10,175),(20,400),(30,400)]:
 for i in range(count):
  root=tf.random.experimental.stateless_fold_in(tf.constant([24881000,domain]),i)
  for s in range(6):
   for rep in range(2):
    k=tuple(keyed.combined(root,tf.constant([s,rep])).numpy().tolist());identity=(domain,i,s,rep)
    assert k not in seen,(identity,seen.get(k));seen[k]=identity
results.update(passed=True,purpose='actual tiny-model driver regression; not full Dacca or GPU evidence',T=3,J=8,batch=12,
 genuine_nonfinite_row_isolation=True,remaining_11_updates_exact=True,BC_identical_initial_Adam=True,
 derived_training_member_keys=len(seen),training_stream_collisions=0,BC_intentionally_share_domain30=True,
 score_root=94881000,score_different_library_and_root=True)
(ROOT/'local_regression.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
