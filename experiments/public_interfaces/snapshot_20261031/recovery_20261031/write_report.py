from pathlib import Path
import datetime,json,shutil
R=Path(__file__).resolve().parent;C=R.parent;W=C.parents[1]
OUT=W/'outputs/competitor_estimation/implementation_comparison_20261030';OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
g=read(R/'GPU_audit.json');p=read(R/'particles/audit.json')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report=f'''新公共接口 Dacca：作业检查和结果回收
更新：{now}

SSH 已恢复。本次仅检查、回收和重算已保存的评分统计，没有提交作业，没有重新拟合或运行粒子滤波，没有放宽测试容差。

一、正确性
Cuthbert/PFJAX 共用检查作业65097220 FAILED 1:0，实际1803 GPU秒。
Cuthbert：T600/J32 Dacca接口检查通过，但GPU上游测试在1200秒超时前已出现3个失败，未输出测试名称和完整总结；不能称GPU全套通过。
PFJAX：配置内72个GPU测试通过；Dacca公共接口与历史实现的梯度比较有1/23坐标超出原容差，违反坐标绝对差0.000608444，rtol=2e-5、atol=2e-4。没有测公共优化速率，也没有正式优化终点。
PyDPF检查65097403 FAILED 1:0，实际1411 GPU秒。扩展CUDA测试33通过、1失败：float32 multinomial原版与默认/alpha1路径的逐位相等检查，最大差2.384185791015625e-7。T600/J8 Dacca对照、Adam恢复/行失败检查与J5000调用计时通过；这些不替代失败的测试。
TFP检查65096288 FAILED 1:0，实际432 GPU秒。J5000原生PF与候选alpha1前向输出相同；alpha1梯度逐位相等检查失败，22/23坐标有差异，最大差1.4156103e-7。
上述差异的具体原因尚未证实，不能直接归为无害的舍入误差。下一步先做原版重复对照，再检查GPU确定性、编译图和计算顺序；保留原断言和失败结果。

这四个包的现行正式作业均为CANCELLED，Start=None、ElapsedRaw=0；没有开始拟合。它们的提交记录含失败检查的afterok依赖。sacct取消原因字段为None，未确定取消执行者。本次未取消任何作业。

二、原生PF运行时间
原Dacca T600、每月20 Euler步、J5000、单次B1，7次同步计时中位数：
Cuthbert：A40/gl1524，0.393526秒，编译3.120166秒另列。
PFJAX：A40/gl1524，0.220416秒，编译4.864631秒另列。
TFP：A40/gl1511，原生6.278312秒、候选alpha1前向6.213370秒；图构建和预热另列。
PyDPF：此前A40测得原生8.576096秒、候选8.565342秒，9次同种子600个增量哈希一致；本次复用原记录，没有重计时。
particles：CPU原生5.496709秒、候选5.526815秒；没有PF加速证据。
这些是PF调用时间，不是优化效果，也不把CPU与GPU拼成统一速度排名。原生PF计时完成不代表新增优化接口已通过全部正确性检查。

三、particles新接口优化结果
CPU拟合数组65097887全部24项COMPLETED 0:0；原始PyPOMP独立评分65097888 COMPLETED 0:0。
A：原生particles PF + 外部SciPy Nelder–Mead，预算1800秒。
B：新增公共if2_step前900秒 + 同一Nelder–Mead至总1800秒。IF2内部调用包内SMC；Nelder–Mead不是particles原生优化器。
两组各6个共同起点×2重复，全部24次拟合无失败。初值、终点、完整5000×23 swarm、RNG、迟到调用不参与参数选择以及全部成本已核验。
独立评分为原始PyPOMP PF J5000，R36→72→144→288；1692个原始值重新计算log(mean likelihood)、MCSE和likelihood ESS，评分keys逐个重算且无重复。
A全部12个估计分数的中位数：{p['per_method']['A']['median_all_scores_descriptive']:.6f}；9/12达到精度要求。
B全部12个估计分数的中位数：{p['per_method']['B']['median_all_scores_descriptive']:.6f}；12/12达到精度要求。
A的tasks0、2、12在R288时仍低于ESS要求，全部保留；没有追加评分或择优重跑。无密度底值饱和点。
12个配对估计分数均为B较高，B−A中位数{p['paired']['median_B_minus_A_descriptive']:.6f}。其中双方评分均合格的9对也均为B较高。这是本次起点和预算下的观察，不是普遍优势证明；包含低精度点的中位数仅作描述。
两组设置相同时间预算但更新数不同；B接受80–86次IF2。不能把这次结果说成原生优化器比较，也不能推广到尚无正式终点的四个GPU包。
实际CPU作业成本{p['actual_CPU_allocation_seconds']}核秒={p['actual_CPU_allocation_seconds']/3600:.6f}核时，含迟到调用；CPU预检155秒另列。独立评分A40作业96秒另列，不能算作CPU拟合加速。

四、保存位置和剩余工作
原始恢复记录：work/implementation-comparison-20261030/recovery_20261031。
已验证三项GPU检查源文件和回收文件哈希；JAX原有terminal manifest已验证。TFP/PyDPF的传输清单是在回收时创建，不能称其原运行写出的terminal manifest。
particles原始拟合tasks0–5沿用此前回收，tasks6–23及全部独立评分本次回收；各任务原有terminal manifest、357个源文件、传输前后及本地文件哈希全部核验。
本次提供审计JSON、逐终点CSV和中文报告。particles最终科学图和完整复现包尚待生成；四个GPU包需先解决上述检查问题，再运行新接口正式优化。TFP本地完整515对上游测试仍在进行。不能宣布整个比较完成。
'''
(OUT/'recovery_20261031_CN.txt').write_text(report)
(OUT/'status_CN.txt').write_text(report)
for source,name in [(R/'GPU_audit.json','GPU_audit_20261031.json'),(R/'particles/audit.json','particles_audit_20261031.json'),(R/'particles/endpoints.csv','particles_endpoints_20261031.csv')]:shutil.copyfile(source,OUT/name)
s=read(C/'status.json');snapshot=read(C/'scheduler_20261031T213441.json')
s.update(updated_utc=now,state='SSH_restored_GPU_validation_failed_formal_jobs_cancelled_particles_results_audited',
         last_scheduler_snapshot='scheduler_20261031T213441.json',last_scheduler_capture_utc=snapshot['captured_at_utc'],
         scheduler_ssh_returncode=0,last_successful_scheduler_snapshot='scheduler_20261031T213441.json',
         last_successful_scheduler_capture_utc=snapshot['captured_at_utc'],last_scheduler_attempt_snapshot='scheduler_20261031T213441.json',
         last_scheduler_attempt_utc=snapshot['captured_at_utc'],last_recovery_report='recovery_20261031/GPU_audit.json')
s['remote_connection'].update(state='restored',error=None,requires_user_action=None,control_socket_exists=True,
                              last_control_socket_check_utc=now)
for pkg in ['cuthbert','pfjax','pydpf','tfp']:
 s['packages'][pkg].update(state='GPU_validation_failed_formal_jobs_cancelled_before_start',
                          formal_endpoints=0,validation_failure=g['failures'][pkg],
                          scheduler_last_verified_utc=snapshot['captured_at_utc'])
s['packages']['particles'].update(state='24_fits_and_independent_scores_completed_and_audited_final_figures_archive_pending',
 scheduler_completed_tasks=list(range(24)),running_tasks=[],pending_tasks=[],audited_fit_endpoints=24,
 audited_fit_failures=0,independent_scored_endpoints=24,score_precision_passed=21,low_precision_tasks=[0,2,12],
 complete_audit='recovery_20261031/particles/audit.json',actual_fit_CPU_seconds=43647,actual_score_GPU_seconds=96,
 completed_tasks_fit_success_audit='passed all24; independent score audit passed; low precision retained',
 scheduler_last_verified_utc=snapshot['captured_at_utc'])
s['completed_evidence']['recovered_GPU_checks']='recovery_20261031/GPU_audit.json'
s['completed_evidence']['native_PF_timing_new_GPU_records']=g['native_PF_timing']
s['known_limitations']=[x for x in s['known_limitations'] if x!='no new formal optimization result yet']+[
 'GPU formal optimization has zero endpoints because correctness gates failed and dependent jobs were cancelled before start',
 'particles24 endpoints audited;3 low precision scores retained;final figures and reproduction archive pending']
s['monitoring']['latest_decision']='Recovered new complete particles results and real GPU validation failures; all retained'
(C/'status.json').write_text(json.dumps(s,indent=2)+'\n')
handoff=f'''\n{now} 覆盖此前排队和SSH失效状态：连接已恢复。scheduler_20261031T213441.json确认jax65097220 FAILED1803秒、TFP65096288 FAILED432秒、PyDPF65097403 FAILED1411秒；四GPU包所有现行formal均CANCELLED、Start=None、elapsed0，无新接口优化终点。三组检查全部已稳定回收并审计，见recovery_20261031/GPU_audit.json，不重复回收。CuthbertGPU全套1200秒超时且已有3个F；PFJAX72GPU测试过但Dacca梯度1坐标超原容差；PyDPF33/34过、一项float32 multinomial逐位相等失败；TFPalpha1梯度22/23坐标逐位相等失败。三者浮点差原因仍需原版重复对照和计算路径诊断，不放宽容差，不直接重交正式拟合。旧源和失败不可覆盖。\nparticles65097887全部24拟合与65097888评分完成；此前tasks0–5沿用，本次仅回收6–23、评分和25日志。全部源/终态/传输/初值/swarm/RNG/late选择/实际成本及1692raw评分重算通过。最终审计recovery_20261031/particles/audit.json、CSV；A median−4169.070668（9/12精度通过），B median−3784.099127（12/12通过）；A低精度tasks0/2/12达R288保留。12配对B均较高，双方合格9对也均正，仅限本次预算起点。实际43647CPU秒+预检155CPU秒，独立A40评分96秒。拟合/评分不重跑不重新回收；剩余作图、视觉检查和复现包。\n回收报告outputs/competitor_estimation/implementation_comparison_20261030/recovery_20261031_CN.txt。新原生PF时间Cuthbert0.393526、PFJAX0.220416、TFP6.278312秒（T600/J5000/B1/A40，7次同步中位数，编译/预热另列）。原生PF时间不能替代优化效果。本次无新提交；定期任务继续，后续先做失败诊断和particles交付，不能反复把已取消作业当排队。\n'''
with (C/'HANDOFF.txt').open('a') as f:f.write(handoff)
print(OUT/'recovery_20261031_CN.txt')
