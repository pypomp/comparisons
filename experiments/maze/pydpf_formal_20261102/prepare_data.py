"""New formal cache: original transformations with train-only image statistics."""
import argparse, time
from pathlib import Path
import torch, pydpf
from common import ROOT, write, sha, ref
import json

def main(a):
    out = Path(a.output); assert not out.exists()
    torch.set_num_threads(1); start = time.perf_counter()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert sha(a.csv) == protocol['data']['csv_sha256']
    dataset = pydpf.StateSpaceDataset(a.csv, state_prefix='state', control_prefix='control',
                                    device=torch.device('cpu'))
    assert len(dataset) == 2000
    ids = torch.randperm(2000, generator=torch.Generator().manual_seed(20261101))
    split = dict(train=ids[:900], validation=ids[900:1300], test=ids[1300:])
    train = dataset.observation[:, split['train']]
    mean, std = train.mean(), train.std(); del train
    assert torch.isfinite(mean) and torch.isfinite(std) and std > 0
    dataset.apply(lambda observation, **kw: (observation-mean)/std, 'observation')
    dataset.apply(lambda state, **kw: (state-torch.tensor([[[1000.,650.,0.]]]))/torch.tensor([[[1000.,1000.,1.]]]), 'state')
    dataset.apply(lambda control, **kw: control/torch.tensor([[[1000.,1000.,1.]]]), 'control')
    record = {k:getattr(dataset,k).permute(1,0,2).contiguous() for k in ('state','observation','control')}
    assert record['observation'].shape == (2000,100,1728) and ref.finite(record)
    record.update(split=split, normalization=dict(mean=float(mean),std=float(std), training_only=True))
    torch.save(record, out)
    write(out.with_suffix('.json'), dict(csv_sha256=sha(a.csv), cache_sha256=sha(out),
          cache_bytes=out.stat().st_size, split={k:v.tolist() for k,v in split.items()},
          normalization=record['normalization'], elapsed_seconds=time.perf_counter()-start,
          original_loader=True, sample_only=False))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
