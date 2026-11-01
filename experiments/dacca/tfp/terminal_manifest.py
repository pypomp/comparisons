from pathlib import Path
import argparse,json,hashlib,os,time
p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['prefix','A','B','C']);a=p.parse_args()
r=Path(__file__).resolve().parent;o=r/'results'/a.method
if a.method=='prefix':assert (o/'COMPLETE.json').exists()
else:assert (o/'FITS_COMPLETE.json').exists() and (o/'independent_scores/COMPLETE.json').exists()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'method':a.method,'job':os.environ.get('SLURM_JOB_ID'),'node':os.environ.get('SLURMD_NODENAME'),'time':time.time(),'source_manifest_sha256':h(r/'manifest.json'),'sha256':{str(p.relative_to(o)):h(p) for p in sorted(o.rglob('*')) if p.is_file() and p.name!='terminal_manifest.json'}}
(o/'terminal_manifest.json').write_text(json.dumps(d,indent=2)+'\n')
