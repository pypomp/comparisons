"""Short orchestration regression, including real original-PyPOMP scoring."""
from pathlib import Path
import tempfile,json,copy
import numpy as np
import engine as e
import worker as w
R=Path(__file__).resolve().parent
def main():
    cfg=w.read(R/'config.DRAFT.json');cfg['model'].update(T=12,J=16);cfg['start_indices']=[0];cfg['repetitions']=1;cfg['batch_size']=1
    cfg['IF2_chunk']=2;cfg['gradient_chunk']=2
    for name,spec in cfg['methods'].items():
        spec['IF2_passes']=4 if name=='IF2' else 2 if name=='IFAD_097' else 0
        spec['gradient_updates']=0 if name=='IF2' else 3
    cfg['evaluation'].update(J=16,initial_R=4,maximum_R=8,batch_R=4,minimum_ESS=100)
    with tempfile.TemporaryDirectory(dir=R/'results') as temp:
        root=Path(temp);p=root/'test.json';e.write(p,cfg);checks={}
        for method in w.METHODS:
            out=w.run(p,method,0,local=True,output=root/method)
            status=w.read(out/'status.json');assert status['complete'] and status['points']==1,status
            endpoint=w.read(out/'endpoints.json');before=endpoint['parameters_hash']
            for score in out.glob('score_*.json'):
                data=w.read(score);assert data['complete'] and data['R']==8 and data['extended'] and len(data['raw'])==8
                assert not data['precision_passed'] # test threshold100 exceeds maximumR8
            w.run(p,method,0,local=True,output=root/method)
            again=w.read(out/'endpoints.json');assert before==again['parameters_hash']
            assert endpoint['batch_fit_seconds']==again['batch_fit_seconds']
            checks[method]=dict(complete=True,points=1,original_PyPOMP_scoring=True,R=8,low_precision_kept=True,no_refit_on_second_invocation=True)
        result=dict(passed=True,T=12,J=16,methods=checks,scope='Small orchestration regression; no formal full-Dacca result')
        e.write(R/'results/worker_validation.json',result);print(json.dumps(result))
if __name__=='__main__':main()
