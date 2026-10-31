"""One allocation, independent processes; all failures retained."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
old=Path('/scratch/ionides_root/ionides1/hkunyang/model024_20261006')
deps=[old/'cuthbert_dacca_20261027/deps',old/'competitor_estimation_20261028/pfjax/deps']
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for n,h in manifest['files'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
results={}
def run(pkg,label,command,source='candidate',x64=False,timeout=900,cwd=None):
 env=os.environ.copy();paths=[ROOT/pkg/source]
 if pkg=='pfjax':paths[0]/='src'
 paths += [ROOT/pkg/'validation',*deps]
 env.update(PYTHONPATH=':'.join(map(str,paths)),JAX_PLATFORMS='cuda',JAX_ENABLE_X64=str(x64),PYTHONDONTWRITEBYTECODE='1')
 target=OUT/f'{pkg}_{label}.process.json'
 cmd=[sys.executable,str(ROOT/'run_recorded.py'),'--record',str(target),'--cwd',str(cwd or ROOT),'--timeout',str(timeout),'--',sys.executable,*map(str,command)]
 r=subprocess.run(cmd,env=env);results[f'{pkg}_{label}']=r.returncode
 (OUT/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
for pkg in ('cuthbert','pfjax'):
 test=['-m','pytest','-q','tests','-o','cache_dir=/tmp/dacca-new-api-'+pkg]
 if pkg=='cuthbert':test+=['--ignore=tests/test_examples_scripts.py']
 run(pkg,'tests',test,x64=True,cwd=ROOT/pkg/'candidate',timeout=1200)
 check='check_dacca_api.py' if pkg=='cuthbert' else 'check_dacca.py'
 run(pkg,'dacca',[ROOT/pkg/'validation'/check,'--output',OUT/f'{pkg}_dacca.json'],timeout=600)
 run(pkg,'native_timing',[ROOT/'measure.py',pkg,'native','--output',OUT/f'{pkg}_native_timing.json'],source='upstream',timeout=600)
 if results[f'{pkg}_dacca']==0 and results[f'{pkg}_tests']==0:
  run(pkg,'public_cost',[ROOT/'measure.py',pkg,'public','--output',OUT/f'{pkg}_public_cost.json'],timeout=600)
evidence={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name not in ('terminal_manifest.json','COMPLETE.json')}
(OUT/'terminal_manifest.json').write_text(json.dumps({'files':evidence,'source_manifest_sha256':hashlib.sha256((ROOT/'source_manifest.json').read_bytes()).hexdigest()},indent=2)+'\n')
(OUT/'COMPLETE.json').write_text(json.dumps({'completed':True,'all_passed':all(v==0 for v in results.values()),'exit_codes':results},indent=2)+'\n')
sys.exit(0 if all(v==0 for v in results.values()) else 1)
