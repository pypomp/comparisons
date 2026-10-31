"""Test-only repair for the upstream module/function import-name collision.

The official test bodies are unchanged. Only their local name particle_filter
is bound to the submodule they already expect. This does not change TFP's
public exports, runtime algorithm, assertions, or numerical tolerances.
"""
import importlib

def pytest_collection_modifyitems(session,config,items):
    seen=set()
    for item in items:
        module=getattr(item,'module',None)
        if module is None or module.__name__ in seen:continue
        seen.add(module.__name__)
        if module.__name__.endswith('.experimental.mcmc.particle_filter_test'):
            value=getattr(module,'particle_filter',None)
            if callable(value) and not hasattr(value,'particle_filter'):
                module.particle_filter=importlib.import_module('tensorflow_probability.python.experimental.mcmc.particle_filter')
