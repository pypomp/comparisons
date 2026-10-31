#!/usr/bin/env python3
"""Check source reuse and record call sites; never run a filter or optimizer."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/native_pf_audit_20261031'
SNAP = 'experiments/public_interfaces/snapshot_20261031/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fragment(path, symbol=None):
    data = path.read_bytes()
    if symbol is None:
        return data, 1
    source = data.decode()
    node = ast.parse(source)
    for name in symbol.split('.'):
        node = next(n for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return ast.get_source_segment(source, node).encode(), node.lineno


def main():
    # Exact text comparison, not merely matching function names or AST shapes.
    checks = [
        ('cuthbert', 'cuthbert/smc/particle_filter.py', None),
        ('cuthbert', 'cuthbertlib/resampling/multinomial.py', None),
        ('cuthbert', 'cuthbertlib/resampling/systematic.py', None),
        ('cuthbert', 'cuthbertlib/resampling/autodiff.py', 'stop_gradient_decorator'),
        ('pfjax', 'src/pfjax/particle_filter.py', 'particle_filter.filter_step'),
        ('pfjax', 'src/pfjax/particle_filter.py', 'particle_filter_rb'),
        ('pfjax', 'src/pfjax/particle_resamplers.py', None),
        ('pydpf', 'pydpf/pydpf/filtering.py', 'SIS'),
        ('pydpf', 'pydpf/pydpf/filtering.py', 'ParticleFilter'),
        ('pydpf', 'pydpf/pydpf/model_based_api.py', None),
        ('pydpf', 'pydpf/pydpf/utils.py', None),
        ('tfp', 'tensorflow_probability/python/experimental/mcmc/weighted_resampling.py', None),
        ('particles', 'particles/core.py', None),
        ('particles', 'particles/state_space_models.py', None),
        ('particles', 'particles/resampling.py', None),
    ]
    unchanged = []
    for package, path, symbol in checks:
        base = ROOT / 'packages' / package
        old, oldline = fragment(base / 'upstream' / path, symbol)
        new, newline = fragment(base / 'candidate' / path, symbol)
        assert old == new, (package, path, symbol)
        unchanged.append(dict(package=package, path=path, symbol=symbol,
                              upstream_line=oldline, candidate_line=newline,
                              sha256=sha(new), exact_text_equal=True))

    # These locations were read together with their imports, callbacks and driver.
    # Static call-site existence does not by itself establish numerical correctness.
    locations = {
        'cuthbert': [
            ('packages/cuthbert/candidate/cuthbert/smc/iterated_filtering.py', 'obj = pf.build_filter('),
            ('packages/cuthbert/candidate/cuthbert/smc/iterated_filtering.py', 'final = util.scan_final('),
            ('packages/cuthbert/candidate/cuthbert/smc/_estimation.py', 'filter_obj.filter_combine('),
            ('packages/cuthbert/candidate/cuthbert/smc/dmop.py', 'obj = particle_filter.build_filter('),
            ('packages/cuthbert/candidate/cuthbertlib/resampling/autodiff.py', 'return stop_gradient_decorator(func)'),
            (SNAP + 'cuthbert_formal_v3/run.py', 'iterated_filtering.fit('),
            (SNAP + 'cuthbert_formal_v3/run.py', 'dmop.log_likelihood('),
        ],
        'pfjax': [
            ('packages/pfjax/candidate/src/pfjax/iterated_filter.py', 'result = particle_filter('),
            ('packages/pfjax/candidate/src/pfjax/particle_filter_mop.py', 'result = particle_filter('),
            (SNAP + 'jax_formal_v2/run.py', 'pf.iterated_filter_step('),
            (SNAP + 'jax_formal_v2/run.py', 'pf.particle_filter_mop('),
            ('reports/native_pf_audit_20261031/evidence/pfjax_old_bridge.py', 'jax.lax.scan(month,'),
        ],
        'pydpf': [
            ('packages/pydpf/candidate/pydpf/pydpf/iterated_filtering.py', 'pf = ParticleFilter('),
            (SNAP + 'pydpf/formal_v1/api_adapter.py', 'package.StopGradientDPF('),
            (SNAP + 'pydpf/formal_v1/api_adapter.py', 'pydpf.IteratedFiltering('),
        ],
        'tfp': [
            ('packages/tfp/candidate/tensorflow_probability/python/experimental/mcmc/iterated_filter.py', 'pf.sequential_monte_carlo('),
            ('reports/native_pf_audit_20261031/evidence/tfp_public_api.py', 'tfp.experimental.mcmc.particle_filter'),
            ('reports/native_pf_audit_20261031/evidence/tfp_public_api.py', 'tfp.experimental.mcmc.iterated_filter_step('),
            ('reports/native_pf_audit_20261031/evidence/tfp_keyed.py', 'tfp.experimental.mcmc.resample_systematic('),
        ],
        'particles': [
            ('packages/particles/candidate/particles/iterated_filtering.py', 'pf = SMC('),
            ('packages/particles/candidate/particles/iterated_filtering.py', 'pf.run()'),
            (SNAP + 'particles/model.py', 'particles.SMC('),
            ('reports/native_pf_audit_20261031/evidence/particles_fit.py', 'if2_step('),
        ],
        'pypfilt': [
            ('packages/pypfilt/experiment_code/adapter/bench_pypfilt.py', 'pypfilt.fit('),
            ('reports/native_pf_audit_20261031/evidence/pypfilt/__init__.py', "result = pfilter.run(ctx, start, until, ctx.data['obs'])"),
        ],
    }
    calls = []
    for package, items in locations.items():
        for path, text in items:
            content = (ROOT / path).read_bytes()
            matches = [(i, line.strip()) for i, line in enumerate(content.decode().splitlines(), 1) if text in line]
            assert matches, (path, text)
            calls.extend(dict(package=package, path=path, line=i, source=line,
                              file_sha256=sha(content)) for i, line in matches)
    evidence = json.loads((OUT / 'evidence_sources.json').read_text())
    for entry in evidence:
        assert sha((ROOT / entry['path']).read_bytes()) == entry['sha256'], entry
    # Ensure the old local IF2 loop is not called by the new PFJAX driver.
    driver = (ROOT / (SNAP + 'jax_formal_v2/run.py')).read_text()
    assert 'bridge.if2(' not in driver and 'bridge.objective(' not in driver
    result = dict(source_audit_passed=True, unchanged_native_code=unchanged,
                  call_sites=calls, evidence_sources_checked=len(evidence),
                  legacy_pfjax_if2_uses_local_time_loop=True,
                  current_cuthbert_uses_local_time_loop_over_native_updates=True,
                  current_pfjax_alpha1_gradient_correction_added_locally=True,
                  numerical_tests_run=False, gpu_tests_run=False, optimizations_run=False,
                  limitation='Static source and exact-code comparison only. GPU correctness and optimization remain separately assessed.')
    (OUT / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Source locations checked', '',
             'Generated by `python3 scripts/audit_native_pf.py`. No numerical experiments run.', '',
             '| Package | Call site |', '|---|---|']
    for c in calls:
        lines.append(f"| {c['package']} | [{c['path']}:{c['line']}](../../{c['path']}#L{c['line']}) |")
    lines += ['', 'Exact unchanged functions and file hashes are in [audit.json](audit.json).']
    (OUT / 'source_locations.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(source_audit_passed=True, exact_code_checks=len(unchanged),
                          call_sites=len(calls), evidence_sources=len(evidence))))


if __name__ == '__main__':
    main()
