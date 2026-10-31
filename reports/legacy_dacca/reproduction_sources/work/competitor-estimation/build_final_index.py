"""Assemble the final original-Dacca index only after all package deliveries pass.

No remote work, fitting, PF evaluation, or changes to existing package deliveries.
The seven original reproduction archives remain separate and are indexed by SHA.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import statistics

from update_current_report import finite_number, load_paper_delivery, load_remaining_full_delivery, verify_scores

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/competitor_estimation'
DEST = OUT / 'final_comparison'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def summarize_records(rows):
    """All retained finite scores and the explicitly conditional complete/scored subset."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row['experiment_scope'], row['package'], row['method']), []).append(row)
    result = []
    for (scope, package, method), group in sorted(grouped.items()):
        finite = [r for r in group if finite_number(r['logmeanlik'])]
        complete_fit = [r for r in finite if not r['fit_failed']]
        complete = [r for r in complete_fit if not r['evaluation_failed']]
        result.append(dict(scope=scope, package=package, method=method, n=len(group), finite_n=len(finite),
            complete_fit_finite_n=len(complete_fit), complete_fit_scored_finite_n=len(complete),
            complete_finite_n=len(complete), fit_failures=sum(r['fit_failed'] for r in group),
            evaluation_failures=sum(r['evaluation_failed'] for r in group),
            low_precision=sum(not r['precision_passed'] for r in finite),
            all_precision_not_passed=sum(not r['precision_passed'] for r in group),
            floor_saturated=sum(r['score_floor_saturated'] for r in group),
            median_all_finite=statistics.median(r['logmeanlik'] for r in finite) if finite else None,
            median_complete_finite=statistics.median(r['logmeanlik'] for r in complete) if complete else None,
            conditional_median_definition='Finite score, completed fit, no evaluation failure; low-precision and floor-saturated points are not otherwise excluded'))
    return result


def main():
    # Refuse before creating output if any final review is outstanding.
    packages = {}
    for package, relative in {
        'PyDPF': 'pydpf/formal_v1_final', 'TFP': 'tfp/formal_v2_final',
        'Dynamax': 'dynamax/dacca_formal_final',
    }.items():
        data = load_remaining_full_delivery(package, relative)
        assert data is not None, f'{package} is not fully delivered'
        packages[package] = data
    paper, reason = load_paper_delivery()
    assert paper is not None, reason
    rows, sources, archives, costs = [], {}, {}, {}

    def record(package, method, start, replicate, ll, mcse, ess, R, failed, precision, floor,
               scope='six common starts x two repeats', evaluation_failed=False, raw_repetitions=None):
        actual_repetitions = R if raw_repetitions is None else raw_repetitions
        rows.append(dict(experiment_scope=scope, package=package, method=method, start=start,
            replicate=replicate, logmeanlik=ll if isinstance(ll, (int, float)) and math.isfinite(ll) else None,
            MCSE=mcse if isinstance(mcse, (int, float)) and math.isfinite(mcse) else None,
            likelihood_ESS=ess if isinstance(ess, (int, float)) and math.isfinite(ess) else None,
            R=R, raw_repetitions=actual_repetitions,
            evaluation_skipped_due_to_nonfinite_fit=bool(failed and actual_repetitions == 0),
            fit_failed=bool(failed),
            evaluation_failed=bool(evaluation_failed), score_failure=bool(evaluation_failed),
            precision_passed=bool(precision), score_floor_saturated=bool(floor)))

    def source(package, folder, summary, archive):
        p = folder / archive
        assert p.is_file(), p
        sources[package] = dict(summary=str(folder / summary), summary_sha256=sha(folder / summary),
                                directory=str(folder))
        for name in ('delivery_manifest.json', 'root_delivery_audit.json'):
            if (folder / name).is_file():
                sources[package][name.replace('.json', '_sha256')] = sha(folder / name)
        archives[package] = dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

    for package, data in packages.items():
        folder = Path(data['source_file']).parent
        manifest = read(folder / 'delivery_manifest.json')
        # Nested evidence/*.tar.gz are raw method archives, not the complete delivery.
        assert 'reproduction.tar.gz' in manifest['sha256']
        source(package, folder, 'summary.json', 'reproduction.tar.gz')
        costs[package] = data['root_costs']
        for r in data['rows']:
            record(package, r['method'], r['start'], r['replicate'], r['logmeanlik'], r['MCSE'],
                   r['likelihood_ESS'], r['R'], r['fit_failed'], r['precision_passed'], r['score_floor_saturated'],
                   evaluation_failed=r['evaluation_failed'])

    folder = OUT / 'pfjax/evaluation_v2'
    pf = read(folder / 'audit.json')
    assert pf['release_ready'] and pf['all_original_parameters_unchanged']
    for n, h in read(folder / 'delivery_manifest.json')['files'].items():
        assert sha(folder / n) == h
    visual = read(folder / 'visual_check.json')
    assert visual['passed'] and visual['image_sha256'] == sha(folder / visual['checked_file'])
    source('PFJAX', folder, 'audit.json', 'reproduction.tar.gz')
    assert pf['total_allocation_seconds'] == pf['original_job_allocation_seconds'] + pf['repair_allocation_seconds']
    costs['PFJAX'] = dict(resource='A40 GPU seconds', formal_allocation_seconds=pf['original_job_allocation_seconds'],
        repair_rescoring_allocation_seconds=pf['repair_allocation_seconds'],
        total_formal_and_repair_seconds=pf['total_allocation_seconds'],
        repair_score_compile_seconds=pf['repair_score_compile_seconds'],
        by_method={m:dict(fitting_seconds=t['batch_fit_seconds'], fitting_scope='Unique whole batch of12; includes full IFAD prefix',
            fit_compile_seconds=t['compile_seconds'], preparation_seconds=t['preparation_seconds'],
            score_execution_seconds=t['evaluation_seconds'], original_score_execution_seconds=t['original_evaluation_seconds'],
            score_compile_seconds=None, score_process_seconds=None, allocation_seconds=None, raw_timing=t,
            missing_timing_note='Repair compilation is recorded once for all methods, not allocated per method. Whole scoring-process time and per-method allocation are not isolated.')
            for m, t in pf['summary'].items()},
        timing_scope='Original fit and score job plus mandatory independent rescoring of all36 unchanged endpoints; old score costs retained. Method fit times are not multiplied by12.')
    for r in read(folder / 'endpoints.json'):
        assert finite_number(r['logmeanlik'])
        record('PFJAX', r['method'], r['start'], r['repeat'], r['logmeanlik'], r['MCSE'],
               r['likelihood_ESS'], r['repetitions'], r['failed'], r['precision_passed'], False,
               evaluation_failed=bool(r.get('evaluation_failed', r.get('score_failure', False))))

    for package in ('pypfilt', 'particles'):
        folder = OUT / package / 'formal_v1'
        data = read(folder / 'results.json'); review = read(folder / 'root_delivery_audit.json')
        assert data['complete'] and data['completed'] == 12 and review['passed']
        assert review['results_sha256'] == sha(folder / 'results.json')
        visual = read(folder / 'visual_check.json')
        assert visual == review['visual_check'] and visual['passed']
        assert visual['image_sha256'] == sha(folder / 'figure.png')
        name = f'{package}_dacca_formal_reproduction.tar.gz'
        assert review['reproduction_sha256'] == sha(folder / name)
        source(package, folder, 'results.json', name)
        verify_scores(data['rows'])
        assert data['totals']['Slurm_allocated_CPU_seconds'] == review['Slurm_CPU_seconds']
        costs[package] = dict(resource='CPU core seconds', formal_allocation_seconds=review['Slurm_CPU_seconds'],
            original_totals=data['totals'], failed_objective_PF_calls=data['failed_PF_calls'],
            by_method={'PF + external Nelder-Mead': dict(fitting_seconds=data['totals']['charged_fitting_seconds'],
                fitting_scope='12 scalar single-core CPU fits summed, including late rejected calls',
                fit_compile_seconds=None, preparation_seconds=sum(r['fit_preparation_seconds'] for r in data['rows']),
                score_execution_seconds=sum(r['score_execution_seconds'] for r in data['rows']),
                score_compile_seconds=sum(r['score_compile_seconds'] for r in data['rows']),
                score_process_seconds=data['totals']['scoring_wall_seconds'], allocation_seconds=review['Slurm_CPU_seconds'],
                missing_timing_note='No isolated fitting compile timer; reported scoring process wall includes its preparation, compilation and execution.')},
            timing_scope='12 scalar CPU fits,1800s each; final late calls and independent scoring charged. Not a GPU speed comparison.')
        for r in data['rows']:
            assert r['score_finished']
            record(package, 'PF + external Nelder-Mead', r['start'], r['replicate'], r['logmeanlik'],
                   r['mcse'], r['likelihood_ess'], r['score_repetitions'], r['fit_failed'], r['precision_ok'],
                   bool(r['score_raw']) and all(x == -24867.998046875 for x in r['score_raw']),
                   evaluation_failed=bool(r.get('evaluation_failed', r.get('score_failure', not r['score_finished']))))
    assert len(rows) == 168
    paper_folder = ROOT / 'outputs/cuthbert_dacca_paper/audited_v2'
    source('Paper two-package comparison', paper_folder, 'audit.json', 'reproduction.tar.gz')
    costs['Paper two-package comparison'] = dict(resource='A40 GPU seconds',
        formal_allocation_seconds=paper['actual_GPU_allocation_seconds'], timing_scope=paper['timing_limit'],
        physical_costs=paper['physical_costs'],
        by_method={f"{t['backend']}/{t['method']}":dict(fitting_seconds=t['method_fit_seconds'],
            fitting_scope='Whole batch100 with full IF2 prefix for each method; physical shared-prefix cost is deduplicated only in physical_costs',
            IF2_seconds=t['IF2_seconds'], gradient_seconds=t['gradient_seconds'], fit_compile_seconds=t['compile_seconds'],
            score_execution_seconds=None, score_compile_seconds=None, score_process_seconds=None, allocation_seconds=None,
            missing_timing_note='Final scoring was not separately timed. Job residual includes scoring, imports, cleanup, I/O and hashing; it is not a pure PF timer.')
            for t in paper['summary']})
    for r in read(paper_folder / 'audit.json')['scores']:
        record(r['backend'], r['method'], r['start'], None, r['log_likelihood'], r['MCSE'],
               r['likelihood_ESS'], r['R'], r['fit_failure'], r['precision_ok'], r['score_floor_saturated'],
               scope='paper100 common starts', evaluation_failed=r['score_failure'],
               raw_repetitions=r['R'] if r['fit_finite'] else 0)
    assert len(rows) == 968
    assert len({(r['experiment_scope'], r['package'], r['method'], r['start'], r['replicate']) for r in rows}) == 968
    summaries = summarize_records(rows)
    assert pf['native_OT_not_included']
    attribution = {
        'PFJAX': dict(native='PFJAX模型/接口作为接入基础',
            evaluated='IF2、IFAD和直接ADPF三主组均为本地扩展',
            limitation='原生OT重采样没有通过完整Dacca梯度门槛，不包含在本次三组结果中；不能把本地ADPF称为PFJAX原生OT结果。',
            evidence=sources['PFJAX']),
        'PyDPF': dict(native='直接alpha1使用原生StopGradientDPF与torch.optim.Adam',
            evaluated='IF2扰动/冷却和alpha=.97折扣接入为本地扩展，底层PF与重采样使用PyDPF',
            limitation='直接ADPF从原始起点开始，IFAD含175次IF2；差异不能单独归因于alpha。',
            evidence=sources['PyDPF']),
        'TFP': dict(native='直接alpha1使用原生particle_filter的unbiased_gradients=True与原生Keras Adam',
            evaluated='IF2参数扰动和alpha=.97梯度折扣为本地扩展；PF/SMC及systematic重采样接口来自TFP',
            limitation='每步重采样为明确配置，不是默认ESS自适应选项；梯度XLA失败后采用普通TF GPU图，未换成别包实现。',
            evidence=sources['TFP']),
        'Dynamax': dict(native='原生SSM.fit_sgd优化接口',
            evaluated='接入本地Dacca粒子滤波、IF2和IFAD扩展',
            limitation='不是Dynamax原生通用PF或EM比较；正式结果使用独立标量调用，非12-filter批量核。',
            evidence=sources['Dynamax']),
        'pypfilt': dict(native='原生粒子滤波', evaluated='外部SciPy Nelder-Mead参数搜索',
            limitation='这是CPU预算实验，不是原生梯度优化器或EM结果。', evidence=sources['pypfilt']),
        'particles': dict(native='原生粒子滤波', evaluated='外部SciPy Nelder-Mead参数搜索',
            limitation='这是CPU预算实验，不是原生梯度优化器或EM结果。', evidence=sources['particles']),
        'Paper/Cuthbert': dict(native='Cuthbert粒子滤波和systematic重采样',
            evaluated='本地IF2/DMOP/IFAD扩展；论文比较训练循环复用PyPOMP Adam',
            limitation='这些扩展并非Cuthbert开箱即有的参数估计设施，也不是其EM基线。',
            evidence=sources['Paper two-package comparison']),
        'Paper/PyPOMP': dict(native='冻结PyPOMP functional.mif、functional.train与Adam',
            evaluated='论文固定迭代和冷却/学习率设置下的IF2与IFAD',
            limitation='100起点论文实验与新增包6起点×2重复实验分别统计。',
            evidence=sources['Paper two-package comparison']),
    }
    result = dict(created_UTC=datetime.now(timezone.utc).isoformat(), all_requested_packages_delivered=True,
        paper_endpoint_records=800, added_package_endpoint_records=168, rows=rows, summaries=summaries,
        actual_final_raw_PF_repetitions=sum(r['raw_repetitions'] for r in rows),
        R_definition='Original recorded evaluation repetition field. Paper nonfinite fits retain planned R36 but produced no PF draws; raw_repetitions gives actual final evaluation draw count. Historical superseded scores are preserved in their original archives and are not counted here.',
        cost_by_experiment=costs, provenance=sources, reproduction_archives=archives,
        algorithm_attribution=attribution,
        scope_notes=['All use original600-month Gaussian Dacca and20 Euler steps per month,J5000,23 free coordinates.',
          'Paper100-start and added6-start x2-replicate experiments are distinct; oldJ1024 results are excluded.',
          'All-finite medians retain poor, failed-fit, floor-saturated and imprecise endpoints. Conditional medians require a completed fit and no evaluation failure, without additionally dropping low-precision or floor points.',
          'Fixed-iteration configurations do not establish equal-cost superiority or isolate alpha.',
          'CPU and GPU costs are not pooled into a speed ranking; batch times are not multiplied by starts.',
          'Separate per-package reproduction archives contain frozen sources, data, raw logs, raw scores, provenance and visual reviews.'])
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    for name, values in [('endpoints.csv', rows), ('method_summary.csv', summaries)]:
        with (DEST / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    lines = ['原始 Dacca 参数估计比较：最终交付索引', '',
             '论文两包800条终点记录和六个新增包168条终点记录均已交付。失败、低精度和观测底值饱和结果完整保留。',
             f"最终采用的原始PF评分共{result['actual_final_raw_PF_repetitions']}次。论文7条非有限拟合未执行终点评分；原记录R=36是计划重复数，表中另列raw_repetitions=0及跳过原因。历史被替换的评分另存原包，不混入此计数。",
             '以下按实验范围和包分别报告。共同模型不意味着初始化、迭代预算、算法接口和硬件相同，不能合并成统一速度或优化器排名。', '']
    lines += ['实际评估的算法与接入边界', '包 | 原生部分 | 本次评估/本地扩展']
    for package, a in attribution.items():
        lines.append(f"{package} | {a['native']} | {a['evaluated']}")
        lines.append(f"  {a['limitation']}")
    lines.append('')
    for scope in sorted({r['scope'] for r in summaries}):
        lines += [scope, '包/方法 | 有限保留中位likelihood | 有限/总数 | 完整拟合且评分无失败的有限数 | 拟合失败 | 评分失败 | 有限低精度 | 底值饱和']
        for r in summaries:
            if r['scope'] == scope:
                median = f"{r['median_all_finite']:.6f}" if r['median_all_finite'] is not None else '无'
                lines.append(f"{r['package']}/{r['method']} | {median} | {r['finite_n']}/{r['n']} | {r['complete_finite_n']} | {r['fit_failures']} | {r['evaluation_failures']} | {r['low_precision']}/{r['finite_n']} | {r['floor_saturated']}")
                if r['median_complete_finite'] is not None:
                    lines.append(f"  完整拟合且评分无失败的有限子集：{r['complete_finite_n']}/{r['n']}，条件中位数{r['median_complete_finite']:.6f}；此子集仍保留低精度和底值点。")
        lines.append('')
    show = lambda value: '未单独测量' if value is None else f'{value:.3f}'
    for resource in ('A40 GPU seconds', 'CPU core seconds'):
        label = 'GPU占用（GPU秒）' if resource == 'A40 GPU seconds' else 'CPU占用（核秒）'
        lines += [label, '实验/包 | 正式作业实际占用 | 必要重评分追加 | 正式与重评分合计']
        for package, cost in costs.items():
            if cost['resource'] != resource:
                continue
            repair = cost.get('repair_rescoring_allocation_seconds', 0)
            total = cost['formal_allocation_seconds'] + repair
            lines.append(f"{package} | {cost['formal_allocation_seconds']} | {repair} | {total}")
            if cost.get('prior_package_and_gate_allocation_seconds') is not None:
                lines.append(f"  此前单独记录的部署/预检/效率成本：{cost['prior_package_and_gate_allocation_seconds']} GPU秒，未混入本表正式作业。")
            lines += ['  方法 | 拟合秒 | 单独拟合编译秒 | 评分执行秒 | 评分编译秒 | 评分进程秒']
            for method, timing in cost['by_method'].items():
                lines.append('  ' + ' | '.join([method] + [show(timing.get(k)) for k in (
                    'fitting_seconds', 'fit_compile_seconds', 'score_execution_seconds',
                    'score_compile_seconds', 'score_process_seconds')]))
                lines.append(f"    {timing['fitting_scope']}。{timing['missing_timing_note']}")
            lines.append(f"  总计时范围：{cost['timing_scope']}")
        lines += ['计时字段有包含关系，不能把编译、执行、进程和分配时间全部相加；未单独测量不等于零。', '']
    lines += ['各包完整likelihood与实际耗时科学图、中文报告、原始评分和复现包：']
    for package, s in sources.items():
        lines.append(f"{package}: {s['directory']}\n  复现包SHA256: {archives[package]['sha256']}")
    lines += ['', '所有中位数均注明分母；条件中位数排除拟合失败和评分失败，但不额外排除低精度或底值点。低精度数量的分母是有限评分数；全部精度未通过数量另存CSV/JSON，含无法评分行。',
              '低精度和底值点保留不表示良好拟合；完成迭代不表示收敛。CPU/GPU实际成本分列，不合成速度排名。',
              '原生算法和本地扩展的具体边界、编译/拟合/评分嵌套计时、检查点可追溯性的限制见各包报告。',
              '本索引仅汇总已通过审核的交付，没有重新拟合或重新计算PF评分。']
    (DEST / 'report_CN.txt').write_text('\n'.join(lines) + '\n')
    names = ['results.json', 'endpoints.csv', 'method_summary.csv', 'report_CN.txt']
    (DEST / 'index_manifest.json').write_text(json.dumps(dict(release_ready=True, files={n:sha(DEST/n) for n in names},
        source_delivery_archives=archives), indent=2) + '\n')
    print(json.dumps(dict(output=str(DEST), endpoint_records=968, all_requested_packages_delivered=True)))


if __name__ == '__main__':
    main()
