"""Use the existing isolated Python 3.12 dependency directory without editing it."""
from pathlib import Path
import os,sys,subprocess,time,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
out=HERE/'local_checks';out.mkdir(exist_ok=True)
name,variant,script,*args=sys.argv[1:]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(HERE/'source'/variant),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',
    NUMBA_CACHE_DIR=str(out/'numba'))
code='import sys,runpy;sys.path.append('+repr(str(ROOT/'work/panel-cute-cpu/lib/python3.12/site-packages'))+');'
if script=='pytest':
    code+='sys.argv='+repr(['pytest']+args)+';runpy.run_module("pytest",run_name="__main__")'
else:code+='sys.argv='+repr([str(HERE/script)]+args)+';sys.path.insert(0,'+repr(str(HERE))+');runpy.run_path(sys.argv[0],run_name="__main__")'
t=time.perf_counter();p=subprocess.run(['/opt/homebrew/bin/python3.12','-c',code],env=env,cwd=HERE,
    text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(out/(name+'.log')).write_text(p.stdout)
record=dict(exit_code=p.returncode,seconds=time.perf_counter()-t,command=code,variant=variant)
(out/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));print(p.stdout[-5000:]);sys.exit(p.returncode)
