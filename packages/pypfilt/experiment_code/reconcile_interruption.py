"""Explicitly account for an unclean process before resuming its saved NM calls.

Never runs automatically: require terminal Slurm evidence and an audited total
fit-time charge. Completed work and unknown lost work cannot be treated as free.
"""
from pathlib import Path
import argparse,hashlib,time
import fit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit-directory',required=True)
    p.add_argument('--charged-fit-seconds',required=True,type=float)
    p.add_argument('--terminal-accounting-evidence',required=True)
    a=p.parse_args();folder=Path(a.fit_directory).resolve()
    assert folder.is_relative_to(fit.ROOT)
    path=folder/'fit_state.json';old=fit.read(path)
    assert old['phase']=='running' and not (folder/'endpoint.json').exists()
    evidence=Path(a.terminal_accounting_evidence).resolve();assert evidence.is_file()
    rows=[fit.read(q) for q in (folder/'calls').glob('call_*.json')]
    minimum=max([old['charged_fit_seconds']]+[row['finish_fit_seconds'] for row in rows])
    assert a.charged_fit_seconds>=minimum
    audit=dict(old_state=old,recovered_charged_fit_seconds=a.charged_fit_seconds,
        evidence=str(evidence),evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        time_unix=time.time(),note='Explicitly reconciled lost segment; not an automatic free restart')
    fit.write(folder/'recovery'/f'recovery_{time.time_ns()}.json',audit)
    segment=int(old['segments'])-1
    target=folder/'segments'/f'segment_{segment:04d}.json'
    if not target.exists():
        fit.write(target,dict(interrupted=True,prior_fit_seconds=None,
            charged_fit_seconds=a.charged_fit_seconds,replay_preparation_seconds=None,
            active_call=old['active_call'],interrupted_PF_attempt=old['active_call'] is not None,
            accounting_reconciliation=audit))
    fit.write(path,dict(old,phase='interrupted',charged_fit_seconds=a.charged_fit_seconds))

if __name__=='__main__':main()
