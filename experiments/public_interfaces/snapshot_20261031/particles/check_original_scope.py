"""Actual bounded upstream example and unchanged algorithms; not a full test suite."""
import ast, hashlib, json, os
from pathlib import Path
import numpy as np
import particles
from particles import state_space_models as ssms

root=Path(particles.__file__).resolve().parents[1]
syntax=[]
for path in sorted(root.rglob('*.py')):
    compile(path.read_text(),str(path),'exec');syntax.append(str(path.relative_to(root)))
controls={}
for cls in [ssms.Bootstrap,ssms.GuidedPF,ssms.AuxiliaryPF]:
    np.random.seed(81004)
    pf=particles.SMC(fk=cls(ssm=ssms.StochVol(),data=np.array([.1,-.3,.2,.7,-.5,.9])),N=32,collect='off')
    pf.run()
    controls[cls.__name__]=dict(loglik=float(pf.logLt),states=pf.X.tolist(),weights=pf.W.tolist(),
        rng_sha256=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest(),
        rng_position=np.random.get_state()[2])
path=root/'tests/smc_samplers/linear_reg.py';tree=ast.parse(path.read_text())
overrides={'N':100,'lc':5,'nruns':2};changed=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name in overrides:node.value=ast.Constant(value=overrides[name]);changed.append(name)
assert set(changed)==set(overrides)
np.random.seed(81005);scope={};exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),scope)
out=dict(passed=True,syntax_checked_files=len(syntax),syntax_only_not_executed=syntax,
    pytest_collected=0,full_upstream_suite_passed_claim=False,original_algorithm_controls=controls,
    actual_upstream_linear_reg=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        overrides=overrides,upstream_default=dict(N=10000,lc=50,nruns=1000),
        loglik=[float(v['output'].logLt) for v in scope['results']],
        means=[v['mean_theta'].tolist() for v in scope['results']],
        purpose='bounded execution of original script, not posterior accuracy benchmark'),
    other_upstream_examples='inventoried and syntax compiled, not all executed')
print('RECORD='+json.dumps(out))
