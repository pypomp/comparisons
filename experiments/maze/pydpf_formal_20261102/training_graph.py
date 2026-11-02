"""Route exact training shapes to the previously validated FCNN implementation.

All runners are created before replacing module.forward, so every runner owns
the original bound native forward method. Inference never uses custom autograd.
"""
import torch
from network_graph import NetworkRunner


class TrainingRouter:
    def __init__(self, module, batches, particles, cuda):
        self.original = module.forward
        self.runners = {b: NetworkRunner(module, (b, particles, 128), cuda)
                        for b in batches}
        self.inference_calls = 0

    def __call__(self, x):
        if not torch.is_grad_enabled():
            self.inference_calls += 1
            return self.original(x)
        return self.runners[x.shape[0]](x)


def install(dpf, batches=(64, 4), particles=100, cuda=True):
    modules = [(n, m) for n, m in dpf.named_modules() if type(m).__name__ == 'FCNN']
    assert len(modules) == 4
    routers = {n: TrainingRouter(m, batches, particles, cuda) for n, m in modules}
    for n, m in modules:
        m.forward = routers[n]
    return routers
