"""Run a local check and preserve stdout, stderr and actual process exit code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs/competitor_estimation/particles_implementation_v1'
name, mode, *args = sys.argv[1:]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
           NUMBA_CACHE_DIR='/private/tmp/particles-if2-numba',
           PYTHONPATH=str(HERE / 'candidate'), OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PARTICLES_IF2_OUTPUT=str(OUT))
bootstrap = ('import sys,runpy; sys.path.append(' + repr(str(ROOT / 'work/panel-cute-cpu/lib/python3.12/site-packages')) + '); ')
if mode == 'pytest':
    bootstrap += 'sys.argv=' + repr(['pytest', '-q', str(HERE / 'candidate/tests/test_iterated_filtering.py')]) + '; runpy.run_module("pytest", run_name="__main__")'
else:
    bootstrap += 'sys.argv=' + repr(args) + '; runpy.run_path(sys.argv[0], run_name="__main__")'
start = time.time()
process = subprocess.run(['/opt/homebrew/bin/python3.12', '-c', bootstrap], env=env,
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(OUT / (name + '.log')).write_text(process.stdout)
record = {'command': ['/opt/homebrew/bin/python3.12', '-c', bootstrap], 'exit_code': process.returncode,
          'elapsed_seconds': time.time() - start, 'purpose': 'local correctness/integration; not benchmark timing'}
(OUT / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
print(process.stdout[-2500:])
sys.exit(process.returncode)
