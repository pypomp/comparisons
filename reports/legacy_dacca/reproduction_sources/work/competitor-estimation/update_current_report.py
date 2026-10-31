"""Build the current report from collected, audited results, without remote work."""
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work/competitor-estimation'
OUT = ROOT / 'outputs/competitor_estimation'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_score_failure(package, row):
    """Resolve actual package schemas without treating fit failure as score failure."""
    flags = [row[k] for k in ('evaluation_failed', 'score_failure', 'score_failed') if k in row]
    if flags:
        assert all(isinstance(v, bool) for v in flags) and len(set(flags)) == 1
        return flags[0], 'explicit saved evaluation failure flag'
    if package == 'PyDPF':
        # The released PyDPF row schema records score completion as finished.
        assert isinstance(row.get('finished'), bool)
        return (not row['finished'] or not finite_number(row['logmeanlik'])), 'saved finished flag and finite score; checked against method score_failures'
    if package == 'Dynamax':
        # Its final auditor refuses missing/failed scores and preserves all raw PF draws.
        raw = row['score_raw']
        assert len(raw) == row['R'] and len(raw) >= 2 and all(finite_number(v) for v in raw)
        assert finite_number(row['logmeanlik'])
        return False, 'released auditor requires complete scoring; finite raw PF repetitions checked'
    raise AssertionError(f'{package}: score failure provenance is missing')


def full_delivery_costs(package, data):
    """Keep measured, nested timing scopes; unavailable components remain None."""
    methods = sorted({r['method'] for r in data['rows']})
    by_method = {}
    for method in methods:
        row = dict(fitting_seconds=None, fitting_scope=None, IF2_seconds=None,
                   gradient_seconds=None, fit_compile_seconds=None, preparation_seconds=None,
                   score_execution_seconds=None, score_compile_seconds=None,
                   score_process_seconds=None, allocation_seconds=None)
        if package == 'TFP':
            t = data['timing_by_method'][method]
            row.update(fitting_seconds=t['fitting_synchronized_calls_seconds'],
                       fitting_scope='Synchronized whole batch of12; excludes separate preparation/control/scoring/I/O',
                       IF2_seconds=t['IF2_synchronized_calls_seconds'], gradient_seconds=t['gradient_synchronized_calls_seconds'],
                       score_execution_seconds=t['evaluation_PF_execution_seconds'],
                       score_compile_seconds=t['evaluation_explicit_compile_seconds'],
                       score_process_seconds=t['evaluation_process_wall_seconds'],
                       allocation_seconds=t['Slurm_allocation_seconds'], raw_timing=t,
                       missing_timing_note='Pure fitting compile time is not separable from first discarded execution; detailed graph-build/first-call times are retained in raw_timing.')
        elif package == 'PyDPF':
            t = data['timing'][method]
            row.update(fitting_seconds=t['batch_fitting_seconds'],
                       fitting_scope='Unique whole-batch fitting wall time, including failure attempts; synchronized subset retained separately',
                       IF2_seconds=t['batch_fitting_seconds'] if method == 'IF2' else t.get('warm_IF2_batch_seconds', 0.),
                       gradient_seconds=t.get('gradient_batch_seconds', t['batch_fitting_seconds'] if method == 'ADPF' else 0.),
                       preparation_seconds=t['initialization_seconds'],
                       score_execution_seconds=t['score_execution_seconds'], score_compile_seconds=t['score_compilation_seconds'],
                       score_process_seconds=t['score_process_wall_seconds'], allocation_seconds=t['allocation_seconds'],
                       raw_timing=t, missing_timing_note='No isolated fitting compile timer is reported. Score process wall contains its preparation/compile/execution; do not add nested fields.')
        elif package == 'Dynamax':
            t = data['summary'][method]
            row.update(fitting_seconds=t['fit_seconds_sum'],
                       fitting_scope='Sum of12 separate scalar fits; native SGD internal compilation stays inside fitting',
                       preparation_seconds=t['prepare_seconds_sum'], score_execution_seconds=t['score_seconds_sum'],
                       score_compile_seconds=t['score_compile_seconds_sum'], raw_timing=t,
                       missing_timing_note='Three methods share each task allocation. Per-method allocation and whole scoring-process time are not separately measured; internal SGD compile time is not isolated.')
        else:
            raise AssertionError(package)
        by_method[method] = row
    return dict(resource='A40 GPU seconds', formal_allocation_seconds=data['total_allocation_seconds'],
                timing_scope=data['timing_scope'], by_method=by_method,
                prior_package_and_gate_allocation_seconds=data.get('prior_package_and_gate_allocation_seconds'),
                prior_cost_note='Prior cost is included only when separately recorded; None is unavailable, not zero. Formal allocation is not an all-project compute total.',
                task_allocation_records=data.get('tasks', data.get('scheduler')))


def verify_scores(rows):
    for r in rows:
        raw = r.get('score_raw')
        if raw is None:
            continue
        assert len(raw) == r['score_repetitions']
        m = max(raw)
        w = [math.exp(v - m) for v in raw]
        mean = statistics.mean(w)
        calculated = {
            'logmeanlik': m + math.log(mean),
            'mcse': statistics.stdev(w) / (math.sqrt(len(w)) * mean),
            'likelihood_ess': sum(w) ** 2 / sum(v*v for v in w),
        }
        for k, v in calculated.items():
            assert math.isclose(v, r[k], abs_tol=1e-9, rel_tol=1e-10)
        assert r['precision_ok'] == (calculated['mcse'] <= 1 and calculated['likelihood_ess'] >= 10)


def load_paper_delivery():
    """Use the complete paper result only after its hashes and visual review pass."""
    folder = ROOT / 'outputs/cuthbert_dacca_paper/audited_v2'
    manifest_path = folder / 'delivery_manifest.json'
    audit_path = folder / 'audit.json'
    if not manifest_path.exists() or not audit_path.exists():
        return None, '完整交付文件尚未生成'
    try:
        manifest = read(manifest_path)
        if not (manifest.get('audit_passed') and manifest.get('all_800_endpoints_present')
                and manifest.get('all_three_slurm_jobs_completed') and manifest.get('release_ready')
                and manifest.get('independent_second_review_passed') and manifest.get('bundle_verified')):
            return None, '完整审计或作业完成确认尚未通过'
        if not manifest.get('visual_review_passed') or manifest.get('visual_review_required', True):
            return None, '完整数值审计已有记录，图的视觉检查尚未确认通过'
        for name, digest in manifest['sha256'].items():
            assert sha(folder / name) == digest, f'delivery hash mismatch: {name}'
        assert 'audit.json' in manifest['sha256']
        audit = read(audit_path)
        assert audit['passed'] and audit['endpoints'] == len(audit['scores']) == 800
        rows = audit['scores']
        counts = {
            'all_endpoints': len(rows),
            'finite_parameter_and_score_endpoints': sum(r['fit_finite'] and r['finite'] for r in rows),
            'nonfinite_fit_failures': sum(r['fit_failure'] for r in rows),
            'score_failures': sum(r['score_failure'] for r in rows),
            'score_floor_saturated': sum(r['score_floor_saturated'] for r in rows),
            'finite_low_precision': sum(r['low_precision_finite'] for r in rows),
        }
        assert counts == dict(all_endpoints=800, finite_parameter_and_score_endpoints=793,
                              nonfinite_fit_failures=7, score_failures=0,
                              score_floor_saturated=2, finite_low_precision=9), counts
        assert sum(r['total_starts'] for r in audit['summary']) == 800
        assert sum(r['successful_finite_n'] for r in audit['summary']) == counts['finite_parameter_and_score_endpoints']
        assert sum(r['score_floor_saturated_n'] for r in audit['summary']) == counts['score_floor_saturated']
        assert all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0'
                   for r in audit['scheduler'].values())
        cost = sum(r['allocation_seconds'] for r in audit['physical_costs'])
        return dict(counts=counts, summary=audit['summary'], physical_costs=audit['physical_costs'],
                    actual_GPU_allocation_seconds=cost, actual_GPU_allocation_hours=cost/3600,
                    scope=audit['scope'], timing_limit=audit['timing_limit'],
                    hardware_evidence=audit['hardware_evidence'], audit_passed=True,
                    visual_review_passed=True, delivery_files_verified=len(manifest['sha256']),
                    source_file=str(audit_path), source_file_sha256=sha(audit_path),
                    delivery_manifest=str(manifest_path), delivery_manifest_sha256=sha(manifest_path)), None
    except (AssertionError, KeyError, OSError, ValueError) as exc:
        # A writer may still be replacing the bundle. Do not promote partial evidence.
        return None, f'完整交付核验尚未通过：{exc}'


def load_tfp_IF2_audit():
    # Exact path supplied by the TFP auditor; no inference from runtime status.
    path = WORK / 'tfp/formal_IF2_audit_64995117/audit.json'
    if not path.exists():
        return None
    data = read(path)
    assert data['passed'] and data['state'] == 'COMPLETED' and data['exit_code'] == '0:0'
    assert data['scored_endpoints'] == len(data['endpoints']) == 12
    assert data['all_updates_verified'] == 1950 and data['accepted_each_filter'] == [1950]*12
    assert data['fit_failures'] == data['score_failures'] == data['imprecise_final_scores'] == 0
    rows = data['endpoints']
    assert len({(r['start'], r['replicate']) for r in rows}) == 12
    assert all(r['precision_passed'] and not r['fitting_failed'] and not r['evaluation_failed'] for r in rows)
    assert sum(r['R'] for r in rows) == data['raw_PF_draws'] == 468
    assert math.isclose(statistics.median(r['logmeanlik'] for r in rows),
                        data['score_summary']['median_12_of_12'], abs_tol=1e-9)
    return dict(data, source_file=str(path), source_file_sha256=sha(path), whole_comparison_complete=False)


def load_pydpf_IF2_audit():
    folder = OUT / 'pydpf/formal_IF2_audit_64994967'
    path = folder / 'audit.json'
    if not path.exists():
        return None
    data = read(path)
    review = read(folder / 'root_review.json')
    assert data['passed'] and review['passed'] and data['method_complete']
    assert not data['whole_comparison_complete']
    assert data['scheduler']['State'] == 'COMPLETED' and data['scheduler']['ExitCode'] == '0:0'
    rows = data['rows']
    assert data['endpoint_count'] == len(rows) == 12
    assert len({(r['start'], r['replicate']) for r in rows}) == 12
    assert all(r['fit_passed'] and r['accepted_steps'] == 1950 and r['precision_passed'] for r in rows)
    assert sum(r['repetitions'] for r in rows) == data['score_audit']['raw_repetitions_recomputed'] == 432
    assert data['fit_failures'] == data['score_failures'] == data['low_precision_count'] == 0
    assert data['verified_terminal_collection_files'] == review['collection_file_hashes_verified'] == 2003
    assert sha(folder / 'raw/collection_evidence.json') == review['collection_sha256'] == data['terminal_collection_evidence_sha256']
    assert sha(folder / 'completed_IF2_raw.tar.gz') == data['raw_archive_sha256']
    assert math.isclose(statistics.median(r['logmeanlik'] for r in rows), data['median_logmeanlik'], abs_tol=1e-9)
    assert math.isclose(data['median_logmeanlik'], review['median_logmeanlik'], abs_tol=1e-9)
    assert data['timing']['allocation_seconds'] == review['Slurm_allocation_seconds'] == 18014
    return dict(data, source_file=str(path), source_file_sha256=sha(path))


def load_pydpf_IFAD_audit():
    folder = OUT / 'pydpf/formal_IFAD_audit_64994968'
    path = folder / 'audit.json'
    if not path.exists() or not (folder / 'root_review.json').exists():
        return None
    data = read(path)
    review = read(folder / 'root_review.json')
    assert data['passed'] and review['passed'] and review['package_audit_sha256'] == sha(path)
    assert data['method_complete'] and not data['whole_comparison_complete']
    assert data['scheduler']['State'] == 'COMPLETED' and data['scheduler']['ExitCode'] == '0:0'
    rows = data['rows']
    assert len(rows) == data['endpoint_count'] == 12
    assert len({(r['start'], r['replicate']) for r in rows}) == 12
    complete = [r for r in rows if r['fit_passed']]
    assert len(complete) == data['completed_fit_count'] == 11
    assert all(r['accepted_IF2_steps'] == 175 for r in rows)
    assert all(r['accepted_steps'] == 400 and r['precision_passed'] for r in complete)
    failed = [r for r in rows if not r['fit_passed']]
    assert len(failed) == data['fit_failures'] == 1
    assert failed[0]['row_id'] == 11 and failed[0]['accepted_steps'] == 13
    assert failed[0]['score_floor_saturated'] and failed[0]['logmeanlik'] == -24867.998046875
    assert sum(r['repetitions'] for r in rows) == review['raw_scores_recomputed'] == 432
    assert data['verified_terminal_collection_files'] == review['collection_file_hashes_verified'] == 631
    assert sha(folder / 'raw/collection_evidence.json') == review['collection_sha256']
    assert sha(folder / 'completed_IFAD_raw.tar.gz') == data['raw_archive_sha256']
    for name, subset in [('median_all_retained_endpoints', rows), ('median_complete_fits', complete)]:
        assert math.isclose(statistics.median(r['logmeanlik'] for r in subset), data[name], abs_tol=1e-9)
        assert math.isclose(data[name], review[name], abs_tol=1e-9)
    assert data['timing']['allocation_seconds'] == review['Slurm_allocation_seconds'] == 21352
    return dict(data, source_file=str(path), source_file_sha256=sha(path))


def load_tfp_IFAD_audit():
    folder = WORK / 'tfp/formal_IFAD_audit_64995118'
    path = folder / 'audit.json'
    if not path.exists() or not (folder / 'root_review.json').exists():
        return None
    data = read(path); review = read(folder / 'root_review.json')
    assert data['passed'] and review['passed'] and review['package_audit_sha256'] == sha(path)
    assert data['state'] == 'COMPLETED' and data['exit_code'] == '0:0'
    rows = data['endpoints']
    assert data['scored_endpoints'] == len(rows) == 12
    assert len({(r['start'], r['replicate']) for r in rows}) == 12
    assert data['accepted_each_filter'] == [400]*12 and data['accepted_prefix_each_filter'] == [175]*12
    assert data['fit_failures'] == data['score_failures'] == data['imprecise_final_scores'] == 0
    assert all(r['precision_passed'] and not r['fitting_failed'] and not r['evaluation_failed'] for r in rows)
    assert sum(r['R'] for r in rows) == data['raw_PF_draws'] == review['raw_scores_recomputed'] == 468
    assert data['completed_output_files_sha_verified'] == review['collection_file_hashes_verified'] == 59
    assert sha(folder / 'raw/retrieval_manifest.json') == review['retrieval_manifest_sha256']
    assert math.isclose(statistics.median(r['logmeanlik'] for r in rows), data['score_summary']['median_12_of_12'], abs_tol=1e-9)
    assert math.isclose(data['score_summary']['median_12_of_12'], review['median_logmeanlik'], abs_tol=1e-9)
    assert data['timing']['Slurm_allocation_seconds'] == review['Slurm_allocation_seconds'] == 10913
    return dict(data, source_file=str(path), source_file_sha256=sha(path), whole_comparison_complete=False)


def load_remaining_full_delivery(package, relative):
    """Only promote an audited, visually checked package with all36 original rows."""
    folder = OUT / relative
    manifest_path = folder / 'delivery_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = read(manifest_path)
    if not all(manifest.get(k) for k in ('numeric_audit_passed', 'visual_review_passed',
                                        'root_review_passed', 'release_ready')):
        return None
    hashes = manifest['sha256']
    assert {'summary.json', 'root_review.json', 'visual_check.json'} <= set(hashes)
    assert 'reproduction.tar.gz' in hashes, 'Top-level complete reproduction archive required'
    for name, digest in hashes.items():
        p = (folder / name).resolve()
        assert p.is_relative_to(folder.resolve()), name
        assert sha(p) == digest, f'{package} delivery hash mismatch: {name}'
    data = read(folder / 'summary.json')
    root_review = read(folder / 'root_review.json')
    visual = read(folder / 'visual_check.json')
    assert root_review.get('passed') and visual.get('passed')
    bound_summaries = [v for v in (root_review.get('summary_sha256'),
                       root_review.get('sha256', {}).get('summary.json')) if v is not None]
    assert bound_summaries and all(v == sha(folder / 'summary.json') for v in bound_summaries), 'Root review does not bind the current summary'
    visual_maps = [visual[k] for k in ('figure_sha256', 'sha256', 'figures') if visual.get(k)]
    assert visual_maps, 'Visual review must identify the checked figures'
    for visual_hashes in visual_maps:
        for name, digest in visual_hashes.items():
            assert name in hashes and hashes[name] == digest, f'Stale visual review: {name}'
    rows = []
    for original in data['rows']:
        failed, basis = normalize_score_failure(package, original)
        row = dict(original, evaluation_failed=failed, score_failure=failed, score_failure_basis=basis)
        assert not (failed and row['precision_passed']), 'Failed evaluation cannot pass scoring precision'
        rows.append(row)
    assert data['expected_endpoints'] == len(rows) == 36
    methods = sorted({r['method'] for r in rows})
    assert len(methods) == 3
    method_summaries = []
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        assert len(subset) == 12
        assert {(r['start'], r['replicate']) for r in subset} == {(s, r) for s in range(6) for r in range(2)}
        finite = [r for r in subset if finite_number(r.get('logmeanlik'))]
        complete_fit = [r for r in finite if not r['fit_failed']]
        complete = [r for r in complete_fit if not r['evaluation_failed']]
        if package == 'PyDPF':
            original_summary = next(r for r in data['summary'] if r['method'] == method)
            assert original_summary['score_failures'] == sum(r['evaluation_failed'] for r in subset)
        method_summaries.append(dict(method=method, all_endpoint_records=12,
            finite_scored_endpoints=len(finite), complete_fit_finite_endpoints=len(complete_fit),
            completed_finite_fits=len(complete), complete_fit_scored_finite_endpoints=len(complete),
            fit_failures=sum(r['fit_failed'] for r in subset),
            evaluation_failures=sum(r['evaluation_failed'] for r in subset),
            finite_low_precision=sum(not r['precision_passed'] for r in finite),
            all_precision_not_passed=sum(not r['precision_passed'] for r in subset),
            precision_passed=sum(r['precision_passed'] for r in subset),
            score_floor_saturated=sum(r['score_floor_saturated'] for r in subset),
            median_all_finite_retained=statistics.median(r['logmeanlik'] for r in finite) if finite else None,
            median_complete_finite_fits=statistics.median(r['logmeanlik'] for r in complete) if complete else None))
    assert data['total_allocation_seconds'] > 0
    return dict(data, rows=rows, root_summary=method_summaries,
                root_costs=full_delivery_costs(package, data), source_file=str(folder / 'summary.json'),
                source_file_sha256=sha(folder / 'summary.json'), delivery_manifest=str(manifest_path),
                delivery_manifest_sha256=sha(manifest_path), release_ready=True)


def full_delivery_text(package, data):
    lines = [f'{package}：36/36条原始终点记录已完成数值审计、独立复核和图的视觉检查。',
             '方法 | 有限保留终点中位log likelihood | 有限评分/12 | 完整拟合且评分无失败/12 | 拟合失败 | 评分失败 | 底值饱和 | 评分精度合格']
    for r in data['root_summary']:
        median = r['median_all_finite_retained']
        text = f'{median:.4f}' if median is not None else '无有限评分'
        lines.append(f"{r['method']} | {text} | {r['finite_scored_endpoints']}/12 | "
                     f"{r['completed_finite_fits']}/12 | {r['fit_failures']} | {r['evaluation_failures']} | "
                     f"{r['score_floor_saturated']} | {r['precision_passed']}/12")
        if (r['fit_failures'] or r['evaluation_failures']) and r['median_complete_finite_fits'] is not None:
            lines.append(f"  完整拟合且评分无失败的有限子集{r['completed_finite_fits']}条的条件中位数：{r['median_complete_finite_fits']:.4f}。失败仍保留在全部记录中；不另排除低精度或底值点。")
    lines += ['方法 | 实测拟合秒 | 评分执行秒 | 评分编译秒 | 评分进程秒 | 方法作业分配秒（无法拆分时留空）']
    show = lambda v: '未单独测量' if v is None else f'{v:.3f}'
    for method, timing in data['root_costs']['by_method'].items():
        lines.append(' | '.join([method] + [show(timing[k]) for k in ('fitting_seconds', 'score_execution_seconds',
            'score_compile_seconds', 'score_process_seconds', 'allocation_seconds')]))
        lines.append(f"  拟合口径：{timing['fitting_scope']}。{timing['missing_timing_note']}")
    lines += [f"原始正式作业实际GPU占用合计{data['total_allocation_seconds']}秒（{data['total_allocation_seconds']/3600:.4f}GPU小时）。",
              '评分进程包含评分编译和执行，不能重复相加；批时间只计一次，Dynamax标量试验则累加12次真实调用。',
              'IFAD175+400与直接alpha1 Adam425的前缀、更新数不同，不能把结果差异单独归因于alpha。',
              f"完整报告、图及复现包：{Path(data['source_file']).parent}", '']
    prior = data['root_costs']['prior_package_and_gate_allocation_seconds']
    if prior is not None:
        lines.append(f'另有已记录的先前预检/效率/部署成本{prior} GPU秒，单列保留，不算入正式拟合作业占用。')
    return lines


def main():
    previous = read(OUT / 'current_results_summary.json')
    snapshots = [read(p) for p in WORK.glob('heartbeat_*_remote.json')]
    observed = max(snapshots, key=lambda d: d['observed_at'])
    now = datetime.now(timezone.utc)
    result = {k: previous[k] for k in ('old_cuthbert_J1024', 'old_cuthbert_costs')}
    result.update(observed_UTC=now.isoformat(), observed_Detroit=now.astimezone(ZoneInfo('America/Detroit')).isoformat(),
                  scheduler_observed_at=observed['observed_at'])
    pf = read(OUT / 'pfjax/evaluation_v2/audit.json')
    assert pf['release_ready'] and pf['all_original_parameters_unchanged']
    pf_folder = OUT / 'pfjax/evaluation_v2'
    for name, digest in read(pf_folder / 'delivery_manifest.json')['files'].items():
        assert sha(pf_folder / name) == digest
    pf_visual = read(pf_folder / 'visual_check.json')
    assert pf_visual['passed'] and pf_visual['image_sha256'] == sha(pf_folder / pf_visual['checked_file'])
    result['pfjax_completed_formal'] = pf
    cpus = {}
    for package in ('particles', 'pypfilt'):
        folder = OUT / package / 'formal_v1'
        final = folder / 'results.json'
        data = read(final if final.exists() else folder / 'summary.json')
        if final.exists():
            review = read(folder / 'root_delivery_audit.json')
            visual = read(folder / 'visual_check.json')
            assert review['passed'] and review['results_sha256'] == sha(final)
            assert review['reproduction_sha256'] == sha(folder / f'{package}_dacca_formal_reproduction.tar.gz')
            assert visual == review['visual_check'] and visual['passed']
            assert visual['image_sha256'] == sha(folder / 'figure.png')
        rows = [r for r in data['rows'] if r.get('score_finished')]
        verify_scores(rows)
        assert len({(r['start'], r['replicate']) for r in rows}) == len(rows)
        cpus[package] = {
            'completed': len(rows), 'expected': 12, 'complete': len(rows) == 12,
            'median_logmeanlik_all_rows': statistics.median(r['logmeanlik'] for r in rows),
            'best_logmeanlik_all_rows': max(r['logmeanlik'] for r in rows),
            'precision_passed': sum(r['precision_ok'] for r in rows),
            'failed_fits': sum(r.get('fit_failed', False) for r in rows),
            'method': data['method'], 'native_optimizer': False,
            'totals': data.get('totals', {
                'endpoint_recorded_fit_seconds': sum(r['fitting_seconds'] for r in rows),
                'endpoint_recorded_score_seconds': sum(r['score_seconds'] for r in rows),
            }),
            'rows': rows, 'source_file': str(final if final.exists() else folder / 'summary.json'),
            'source_file_sha256': sha(final if final.exists() else folder / 'summary.json'),
        }
    result['CPU_results'] = cpus
    dyn = []
    for path in sorted((WORK / 'dynamax/dacca/formal_fixed_v1/partial_report').glob('task*_audit.json')):
        if not re.fullmatch(r'task\d+_audit\.json', path.name):
            continue
        audit = read(path)
        if 'records' not in audit:
            continue
        assert audit['all_hashes_and_scores_verified'] and audit['Slurm_state'] == 'COMPLETED'
        for record in audit['records']:
            r = dict(record)
            # The first two historical audit schemas predate this explicit flag.
            precise = math.isfinite(r['logmeanlik']) and r['MCSE'] <= 1 and r['likelihood_ESS'] >= 10
            if 'score_precision_passed' in r:
                assert r['score_precision_passed'] == precise
            r['score_precision_passed'] = precise
            dyn.append(r)
    assert len({(r['method'], r['start_id'], r['replicate']) for r in dyn}) == len(dyn)
    result['Dynamax_partial'] = {'completed_audited_endpoints': len(dyn), 'expected': 36, 'rows': dyn,
                                 'failed_fits': sum(r.get('fit_failed', False) for r in dyn),
                                 'low_precision_scores': sum(not r['score_precision_passed'] for r in dyn),
                                 'scope': 'Only collected completed tasks; not an overall method comparison.'}
    paper, paper_pending_reason = load_paper_delivery()
    completed_paper_groups = sorted(Path(p).parent.name for p in observed.get('paper_group_complete', {}))
    if paper:
        result['paper_completed_audited_v2'] = paper
    else:
        result['paper_completed_fits_pending_scores'] = observed.get('paper_completed_fits', [])
        result['paper_completed_groups_pending_aggregate_audit'] = completed_paper_groups
        result['paper_delivery_pending_reason'] = paper_pending_reason
    result['tfp_status_snapshot'] = observed.get('tfp_status', {})
    tfp = load_tfp_IF2_audit()
    if tfp:
        result['TFP_IF2_completed_partial'] = tfp
    tfp_ifad = load_tfp_IFAD_audit()
    if tfp_ifad:
        result['TFP_IFAD_completed_partial'] = tfp_ifad
    pydpf = load_pydpf_IF2_audit()
    if pydpf:
        result['PyDPF_IF2_completed_partial'] = pydpf
    pydpf_ifad = load_pydpf_IFAD_audit()
    if pydpf_ifad:
        result['PyDPF_IFAD_completed_partial'] = pydpf_ifad
    delivered = {package: data for package, relative in {
        'PyDPF': 'pydpf/formal_v1_final', 'TFP': 'tfp/formal_v2_final',
        'Dynamax': 'dynamax/dacca_formal_final',
    }.items() if (data := load_remaining_full_delivery(package, relative)) is not None}
    result['remaining_packages_completed_deliveries'] = delivered
    result['current_status'] = {
        'PFJAX': '36/36 independent endpoints delivered',
        **{p: f"{d['completed']}/12 scored; {d['precision_passed']} precision passed" for p, d in cpus.items()},
        'Dynamax': f'{len(dyn)}/36 audited endpoints; remaining tasks running or queued',
        'paper': ('800/800 endpoint records audited and visually reviewed; 793 finite scores, '
                  '7 nonfinite fit failures, 2 floor-saturated scores retained, 9 finite low-precision scores'
                  if paper else f"{len(completed_paper_groups)}/3 groups have completion records; {paper_pending_reason}"),
        'PyDPF': ('IF2 and IFAD24/36 endpoint records independently audited; IFAD retains one failed fit; ADPF pending'
                  if pydpf_ifad else 'IF2 12/12 endpoints independently audited; whole comparison12/36, other methods incomplete'
                  if pydpf else 'See scheduler and stage records; final scored delivery not yet integrated here'),
        'TFP': ('IF2 and IFAD24/36 endpoint records independently audited; directADPF pending'
                if tfp_ifad else 'IF2 12/12 endpoints independently audited; whole comparison12/36, other methods incomplete'
                if tfp else 'Latest method status is retained separately; no completed-audit file inferred from the scheduler snapshot'),
    }
    for package in delivered:
        result['current_status'][package] = '36/36 original endpoint records independently audited and visually reviewed; complete reproduction delivered'
    result['scheduler_state'] = observed['squeue']
    result['pydpf_in_progress_unscored'] = observed.get('pydpf_stage_progress', {})
    result['notes'] = [
        'All new results use original Dacca T600,20 Gaussian Euler steps,J5000,23 free coordinates.',
        'CPU native PF plus external Nelder-Mead is distinct from native gradient estimation.',
        'CPU allocation time and GPU batch time are different resource measures, not a hardware speed ranking.',
        'All low-precision scores and failed fits are retained. Each median identifies its denominator and whether it is conditional on finite or complete fits.',
        'The original Cuthbert J1024,180-second experiment has different initialization and budget; do not pool it.',
    ]
    (OUT / 'current_results_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    lines = [f"Dacca 已完成结果与实际耗时\n更新于 {now.astimezone(ZoneInfo('America/Detroit')).strftime('%Y-%m-%d %H:%M %Z')}",
             f"调度状态观测时间：{observed['observed_at']}。拟合完成、独立评分完成与整轮交付分别记录。", '',
             'PFJAX：36/36 完整独立评分已交付；本地 IF2/IFAD/ADPF 扩展，原生 OT 梯度未通过预检。',
             '方法 | 最终 log likelihood 中位数 | 单 A40 整批12条拟合分钟（含IFAD前缀、不含编译评分）']
    for name, r in pf['summary'].items():
        lines.append(f"{name} | {r['median_logmeanlik']:.4f} | {r['batch_fit_seconds']/60:.4f}")
    lines += ['全部36点评分精度通过。原始作业与独立重评分合计2792秒GPU占用，拟合未重跑。',
              'IFAD175+400与直接ADPF425的初始化和迭代次数不同，差距不能单独归因于alpha。', '',
              'CPU方法：各包原生PF + 外部SciPy Nelder–Mead，6起点×2重复，每条1800秒拟合。',
              '包 | 完成/计划 | 最终log likelihood中位数（保留全部行） | 精度合格 | 实际累计拟合核时 | Slurm总核时']
    for name, d in cpus.items():
        t = d['totals']
        fit = t.get('charged_fitting_seconds', t.get('endpoint_recorded_fit_seconds')) / 3600
        allocation = t.get('Slurm_allocated_CPU_seconds')
        allocation_text = f'{allocation/3600:.4f}' if allocation is not None else '未汇总'
        lines.append(f"{name} | {d['completed']}/12 | {d['median_logmeanlik_all_rows']:.4f} | {d['precision_passed']}/{d['completed']} | {fit:.4f} | {allocation_text}")
    lines += ['低ESS行不删除；这些估计未达到全部预定评分门槛。核时是各任务占用之和，不是整轮墙钟时间。',
              'CPU与A40的时间不组成同硬件速度排名。以上是实际预算下的优化终点，不表示已收敛。', '',
              f'Dynamax：目前{len(dyn)}/36个已回收审计终点。官方SSM.fit_sgd接口配合本地Dacca PF/IF2/IFAD。',
              '起点/重复 | 方法 | 最终log likelihood | MCSE | likelihood ESS | 拟合秒 | 失败标记']
    for r in dyn:
        failed = r.get('fit_failed', False)
        failure = str(r.get('failure_reason', 'failed')) if failed else '无'
        lines.append(f"{r['start_id']}/{r['replicate']} | {r['method']} | {r['logmeanlik']:.4f} | {r['MCSE']:.4f} | {r['likelihood_ESS']:.2f} | {r['fit_seconds']:.3f} | {failure}")
        if failed:
            lines.append(f"  提前失败，实际接受更新={r.get('accepted_iterations')}；记录最后有效参数的评分，不能把短耗时视为完成全部迭代的加速。")
            if r.get('score_floor_saturated'):
                lines.append('  PF分数饱和于观测密度底值；MCSE=0/ESS=36不能说明拟合成功。')
        if not r['score_precision_passed']:
            lines.append(f"  独立评分R={r['R']}仍未达到预定精度门槛；原始重复和低精度标记保留，不删除或超出冻结上限追加。")
    lines += ['SGD内部编译计入实际拟合，不能把首步全当作编译扣除。' +
              ('完整交付结果另列于下方。' if 'Dynamax' in delivered else '部分起点不足以判断整轮优劣。'), '']
    if paper:
        lines += ['论文配置两包：800/800终点记录已完成独立审计、交付文件哈希核验和图的视觉检查。',
                  '完整600个月、每月20个Gaussian Euler步、J5000、23自由坐标、共同100个起点。',
                  'IF2=1950轮；IFAD0=60+410；IFAD0.97=175+400；IFAD1=60+425。IFAD0的Adam beta1=0，其余IFAD为0.9。',
                  '793条参数与评分均有限，7条非有限拟合失败，另有9条有限评分精度不足；2条观测密度底值饱和保留在有限值统计中。',
                  '包/方法 | 有限值中位log likelihood | 有限参数和评分/100 | 非有限拟合失败 | 底值饱和 | 有限低精度 | 整批100条拟合秒']
        for r in paper['summary']:
            lines.append(f"{r['backend']}/{r['method']} | {r['median_successful_finite']:.4f} | "
                         f"{r['successful_finite_n']}/{r['total_starts']} | {r['fit_failures']} | "
                         f"{r['score_floor_saturated_n']} | {r['finite_low_precision']} | {r['method_fit_seconds']:.3f}")
        lines += ['中位数仅针对明确数量的有限值，包含底值饱和与低精度行；有限不等于优化成功或收敛。7条非有限失败数量和索引保留在完整报告中。',
                  '各方法拟合时间包含完整IF2前缀；实际GPU占用去重计算共享60轮前缀。',
                  f"三作业实际GPU占用合计{paper['actual_GPU_allocation_hours']:.4f}小时；这是分配GPU时间，不是日历等待时间。", 
                  '最终PF评分未单独计时，不能把Slurm剩余时间称为纯PF用时。编译、启动、清理、I/O和哈希等开销另有记录。',
                  '同一方法内两包在同一作业的A40上顺序执行；不同方法可使用不同A40。固定迭代比较不等于同时间预算比较。',
                  '论文组的100起点结果与新增包6起点×2重复、CPU预算实验及旧J1024实验分别报告，不能直接合并排名。',
                  f"完整结果和复现包：{ROOT / 'outputs/cuthbert_dacca_paper/audited_v2'}", '']
    else:
        lines += [f"论文配置两包：已完成组={completed_paper_groups}；{paper_pending_reason}。以下仅列已结束的拟合记录（各100条整批）：",
                  '方法 | 拟合秒（含IF2前缀） | 有限参数终点 | 非有限终点索引']
        for r in observed.get('paper_completed_fits', []):
            failed = [i for i, finite in enumerate(r['finite']) if not finite]
            lines.append(f"{r['backend']}/{r['method']} | {r['total_fit_seconds']:.3f} | {sum(r['finite'])}/{len(r['finite'])} | {failed}")
        lines += ['非有限终点保留为拟合失败，不重跑、不从总数排除；此分支尚不展示完整最终likelihood比较。', '']
    if pydpf and 'PyDPF' not in delivered:
        t = pydpf['timing']
        lines += [('PyDPF：IF2与IFAD合计24/36条终点记录已审计；原生ADPF尚未完成。' if pydpf_ifad else
                   'PyDPF：IF2组12/12终点已完成独立审计，整体仅12/36；IFAD与原生ADPF尚未齐。'),
                  '使用原生PyDPF ParticleFilter和SystematicResampler的本地IF2扩展；6起点×2重复、J5000，全部跑满1950轮。',
                  f"最终log likelihood中位数{pydpf['median_logmeanlik']:.4f}；0拟合失败，12条评分精度均通过，432次原始PF评分已复算。",
                  f"整批12条拟合{t['batch_fitting_seconds']:.3f}秒，同步计算{t['synchronized_compute_seconds']:.3f}秒；Slurm占用{t['allocation_seconds']}秒（{t['GPU_hours']:.4f}GPU小时）。",
                  f"独立评分进程{t['score_process_wall_seconds']:.3f}秒，内含PF执行{t['score_execution_seconds']:.3f}秒和编译{t['score_compilation_seconds']:.3f}秒，嵌套计时不重复相加。",
                  '原运行未生成terminal_manifest；结束后生成并核验2003项回收文件SHA。整批时间只计一次，不乘12；单个方法完成不能判断三方法优劣。',
                  f"PyDPF部分报告：{OUT / 'pydpf/formal_IF2_audit_64994967/report_zh.txt'}", '']
    if pydpf_ifad and 'PyDPF' not in delivered:
        d = pydpf_ifad; t = d['timing']
        lines += ['PyDPF IFAD：12条均保留独立评分，其中11条完成175轮IF2+400次Adam，1条提前失败。',
                  f"全部12条保留终点中位数{d['median_all_retained_endpoints']:.4f}；11条完整拟合子集中位数{d['median_complete_fits']:.4f}。",
                  '失败row11（起点5/重复1）在175轮IF2后接受13次Adam，第14次尝试梯度非有限；不重跑。',
                  '失败行全部36个评分为−24867.998046875，等于观测密度底值饱和。MCSE=0、ESS=36虽通过机械精度条件，不代表成功或可信的良好拟合。',
                  f"整批拟合{t['batch_fitting_seconds']:.3f}秒，含前缀{t['warm_IF2_batch_seconds']:.3f}秒、梯度阶段{t['gradient_batch_seconds']:.3f}秒与失败尝试成本；Slurm占用{t['allocation_seconds']}秒（{t['GPU_hours']:.4f}GPU小时）。",
                  f"独立评分进程{t['score_process_wall_seconds']:.3f}秒，内含PF执行{t['score_execution_seconds']:.3f}秒与编译{t['score_compilation_seconds']:.3f}秒；计时嵌套不重复相加。",
                  '432次原始评分、12组参数和631项结束后回收SHA已复核；原运行没有terminal_manifest。失败后批量由12变11，批时间只计一次。',
                  f"PyDPF IFAD报告：{OUT / 'pydpf/formal_IFAD_audit_64994968/report_zh.txt'}", '']
    lines += ['PyDPF当前公开阶段进度：']
    for name, stage in observed.get('pydpf_stage_progress', {}).items():
        lines.append(f"{name}: 接受迭代 {stage.get('iteration')}/{stage.get('target_iterations')}，继续行数 {len(stage.get('active_row_ids', []))}，失败行 {stage.get('failed_rows', {})}")
    if tfp and 'TFP' not in delivered:
        t = tfp['timing']; s = tfp['score_summary']
        lines += ['', ('TFP：IF2与IFAD合计24/36条终点记录已审计；直接ADPF尚未完成。' if tfp_ifad else
                      'TFP：IF2组12/12终点已完成独立审计，整体仅12/36；IFAD和直接ADPF比较尚未完成。'),
                  '使用TFP原生SMC/systematic resampling配合本地IF2扩展，6起点×2重复，1950轮，J5000。',
                  f"12条最终log likelihood中位数{s['median_12_of_12']:.4f}；无拟合或评分失败，12条精度均通过。", 
                  f"MCSE范围{s['MCSE_min']:.4f}–{s['MCSE_max']:.4f}，likelihood ESS范围{s['ESS_min']:.2f}–{s['ESS_max']:.2f}；11条R36、1条按协议追加到R72。", 
                  f"1950次同步整批12条调用合计{t['IF2_1950_synchronized_calls_seconds']:.3f}秒；IF2阶段wall {t['IF2_stage_wall_seconds']:.3f}秒；Slurm占用{t['Slurm_allocation_seconds']}秒（{t['Slurm_allocation_GPU_hours']:.4f}GPU小时）。", 
                  '这些调用时间是整批12条，不是每条时间，也不包含所有作业开销；首个准备执行与内部XLA编译无法完全拆分。',
                  '原运行未生成terminal_manifest；50个已结束文件在回收前后及本地逐项哈希核验，不能称为原运行自带的终态清单。',
                  f"TFP部分审计：{tfp['source_file']}"]
    if tfp_ifad and 'TFP' not in delivered:
        d = tfp_ifad; t = d['timing']; s = d['score_summary']
        lines += ['', 'TFP IFAD：12/12完成175轮IF2+400次Adam，无拟合或评分失败，全部达到最终评分精度门槛。',
                  f"全部12条中位log likelihood为{s['median_12_of_12']:.4f}；范围{s['minimum']:.4f}至{s['maximum']:.4f}。",
                  '起点3/重复0虽完成全部迭代，likelihood仅−3821.2932；首次R36的ESS不足，依协议追加至R72后精度通过。保留这一较差结果，不称数值失败或重跑。其余11条R36，共468次原始PF评分。',
                  f"整批同步拟合调用{t['total_fitting_calls_seconds']:.3f}秒，含IF2前缀{t['IF2_prefix_175_synchronized_calls_seconds']:.3f}秒和Adam{t['Adam_400_synchronized_calls_seconds']:.3f}秒；Slurm占用{t['Slurm_allocation_seconds']}秒（{t['Slurm_allocation_GPU_hours']:.4f}GPU小时）。",
                  f"TF拟合进程{t['TF_fitting_process_seconds_including_import_control_prepare_IO']:.3f}秒；XLA探测进程{t['XLA_probe_process_seconds']:.3f}秒；独立评分进程{t['evaluation_process_wall_seconds']:.3f}秒，内含PF执行{t['evaluation_PF_execution_seconds']:.3f}秒、显式编译{t['evaluation_explicit_compile_seconds']:.3f}秒。嵌套计时不重复相加。",
                  'IF2前缀使用XLA；折扣梯度的可选XLA探测失败，按冻结规则使用普通TF图，保留原生TFP接口和Keras Adam。不能将IF2的XLA吞吐套用到梯度。',
                  '59项结束后回收SHA和全部评分已独立复核。完整运行的初始Adam checkpoint已被轮换删除；初始零状态依据冻结代码和先前回归，不能冒称留有完整初态实物。',
                  f"TFP IFAD报告：{WORK / 'tfp/formal_IFAD_audit_64995118/report_CN.txt'}"]
    for package, data in delivered.items():
        lines += ['', *full_delivery_text(package, data)]
    if 'TFP' not in delivered:
        lines += ['TFP最近公开阶段状态（仅据调度快照，不替代独立结果审计）：']
        for name, stage in observed.get('tfp_status', {}).items():
            lines.append(f"{name}: {json.dumps(stage, ensure_ascii=False, sort_keys=True)}")
        if not observed.get('tfp_status'):
            lines.append('当前快照未提供TFP阶段状态。')
    else:
        lines += ['TFP结束后的独立交付审计已取代此前运行中阶段快照；下方原调度快照仅按其观测时间保留。']
    lines += ['各方法完成状态以独立终点评分及交付审计为准。最近逐作业调度快照：', observed['squeue'], '',
              '此前Cuthbert J1024、180秒实验已完整交付，配置不同，不与本轮J5000合并排名。',
              '各包逐行结果、原始评分、精度、源hash、实际成本和复现文件见对应目录。']
    (OUT / 'current_results_CN.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'CPU': {k: {n: v[n] for n in ('completed','precision_passed','median_logmeanlik_all_rows')} for k,v in cpus.items()},
                      'Dynamax_audited': len(dyn), 'paper_complete_delivery_used': bool(paper),
                      'paper_delivery_pending_reason': paper_pending_reason,
                      'TFP_IF2_partial_used': bool(tfp), 'TFP_IFAD_partial_used': bool(tfp_ifad),
                      'PyDPF_IFAD_partial_used': bool(pydpf_ifad)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
