import copy, hashlib, json, os, time, traceback
from pathlib import Path
import torch
import pydpf
import reference_worker as ref

ROOT = Path(__file__).resolve().parent
ARMS = ('native_dpf', 'native_stop', 'discount097')

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')
        f.flush()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()

def rng(gen):
    return dict(private=gen.get_state().clone(), cpu=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state().clone() if gen.device.type == 'cuda' else None)

def set_rng(gen, saved):
    gen.set_state(saved['private']); torch.set_rng_state(saved['cpu'])
    if saved['cuda'] is not None: torch.cuda.set_rng_state(saved['cuda'])

def seed_call(gen, seed):
    gen.manual_seed(seed)
    torch.manual_seed(seed+1000000000)
    if gen.device.type == 'cuda': torch.cuda.manual_seed_all(seed+1000000000)

def state(dpf, opt, sched, gen):
    return dict(model=ref.cpu(dpf.SSM.state_dict()), optimizer=ref.cpu(opt.state_dict()),
                scheduler=sched.state_dict(), rng=rng(gen))

def restore(dpf, opt, sched, gen, saved):
    dpf.SSM.load_state_dict(saved['model'])
    # load_state_dict may retain CPU step/moment storage. Never let subsequent
    # updates mutate the immutable checkpoint used by another paired execution.
    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    sched.load_state_dict(copy.deepcopy(saved['scheduler']))
    dpf.update(); set_rng(gen, saved['rng'])

def ordinary(dpf, gen):
    return pydpf.ParticleFilter(SSM=dpf.SSM, resampler=pydpf.SystematicResampler(gen))

def environment(device):
    return dict(torch=torch.__version__, cuda=torch.version.cuda, pydpf=pydpf.__file__,
                job=os.getenv('SLURM_JOB_ID'), node=os.getenv('SLURMD_NODENAME'),
                device=str(device), gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                matmul_precision=torch.get_float32_matmul_precision(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                strict_determinism=torch.are_deterministic_algorithms_enabled())

def step(dpf, opt, sched, gen, batch, particles, seed, reject=None):
    """Original objective and AdamW; retain full rejected proposal before rollback."""
    dpf.train(); dpf.update(); opt.zero_grad(set_to_none=True); seed_call(gen, seed)
    before = state(dpf, opt, sched, gen)
    ref.sync(gen.device); begin = time.perf_counter()
    try:
        loss, outputs, components = ref.objective(dpf, batch, particles)
        loss.backward(); ref.sync(gen.device)
    except Exception:
        if reject:
            torch.save(dict(last_valid=before, exception=traceback.format_exc(),
                            state_at_exception=state(dpf,opt,sched,gen)), reject)
        restore(dpf,opt,sched,gen,before)
        raise
    kernel = time.perf_counter()-begin
    gradient = ref.gradients(dpf)
    record = dict(loss=ref.cpu(loss), outputs=ref.cpu(outputs), components=ref.cpu(components),
                  gradient=gradient, model_after_forward=ref.cpu(dpf.SSM.state_dict()),
                  rng=rng(gen), seed=seed, kernel_seconds=kernel)
    if not ref.finite(record):
        if reject: torch.save(dict(last_valid=before, rejected=record), reject)
        restore(dpf, opt, sched, gen, before)
        raise FloatingPointError('Nonfinite loss/gradient; no AdamW update; original rejection saved')
    opt.step()
    record['after_step'] = state(dpf, opt, sched, gen)
    if not ref.finite(record['after_step']):
        if reject: torch.save(dict(last_valid=before, rejected=record), reject)
        restore(dpf, opt, sched, gen, before)
        raise FloatingPointError('Nonfinite AdamW proposal; full proposal saved, restored last valid')
    record['before_sha256'] = ref.digest(before)
    record['gradient_sha256'] = ref.digest(gradient)
    record['after_sha256'] = ref.digest(record['after_step'])
    return record
