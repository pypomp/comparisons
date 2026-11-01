"""Read all 24 terminal fits, verify source/terminal hashes, never fit or rescore."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=ROOT/'manifest.json';source_hash=sha(manifest)
    for n,h in json.loads(manifest.read_text())['sha256'].items():assert sha(ROOT.parent/n)==h,n
    rows=[]
    for i in range(24):
        folder=ROOT/'results'/f'task_{i:02d}'
        terminal=json.loads((folder/'terminal_manifest.json').read_text())
        assert terminal['source_manifest_sha256']==source_hash
        for n,h in terminal['sha256'].items():assert sha(folder/n)==h,(i,n)
        row=json.loads((folder/'endpoint.json').read_text())
        pair,arm=divmod(i,2);start,rep=divmod(pair,2)
        assert (row['task_id'],row['method'],row['start_id'],row['replicate'])==(i,['A','B'][arm],start,rep)
        rows.append(row)
    for i in range(0,24,2):assert rows[i]['initial_estimation_sha256']==rows[i+1]['initial_estimation_sha256']
    dest=ROOT/'endpoints.json';text=json.dumps(rows,indent=2,allow_nan=False)+'\n'
    if dest.exists():assert dest.read_text()==text
    else:dest.write_text(text)
    print('Verified 24 terminal fits. Failed fits retained:',sum(x['fit_failed'] for x in rows),flush=True)
if __name__=='__main__':main()
