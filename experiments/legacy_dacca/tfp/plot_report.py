"""Scientific figures and Chinese report from a gated 36-endpoint audit. No simulated data."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'outputs/competitor_estimation/tfp/formal_v2_final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 gate=json.loads((OUT/'delivery_manifest.json').read_text());assert gate['numeric_audit_passed'] and not gate['release_ready']
 assert sha(OUT/'summary.json')==gate['summary_sha256'];s=json.loads((OUT/'summary.json').read_text());assert len(s['rows'])==s['observed_endpoints']==s['expected_endpoints']==36
 methods=['IF2','IFAD','ADPF'];names={'IF2':'IF2\n1950 passes','IFAD':'IFAD (α = .97)\n175 IF2 +\n400 Adam','ADPF':'Native ADPF\n425 Adam\noriginal starts'}
 colors=['#3B6793','#CE8B2B','#229883','#8A62AD','#D16358','#6D8C40'];rows=s['rows']
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'pdf.fonttype':42,'ps.fonttype':42})
 finite=[r['logmeanlik'] for r in rows if r['logmeanlik'] is not None];close=[r['logmeanlik'] for r in rows if r['method'] in ['IF2','IFAD'] and r['logmeanlik'] is not None]
 zoom=bool(len(finite)>1 and len(close)>1 and np.ptp(finite)>1.8*max(np.ptp(close),1))
 fig,axes=plt.subplots(1,2 if zoom else 1,figsize=(12.7 if zoom else 8.6,6.6),squeeze=False)
 def draw(ax,chosen,title):
  for x,m in enumerate(chosen):
   rr=[r for r in rows if r['method']==m];good=[r['logmeanlik'] for r in rr if r['logmeanlik'] is not None and not r['fit_failed'] and not r['evaluation_failed']]
   if good:
    bp=ax.boxplot([good],positions=[x],widths=.48,showfliers=False,patch_artist=True,medianprops={'color':'#252525','linewidth':2},boxprops={'facecolor':'#E8ECF0','edgecolor':'#75818C'},whiskerprops={'color':'#75818C'},capprops={'color':'#75818C'})
   for r in rr:
    if r['logmeanlik'] is None:continue
    offset=(r['start']-2.5)*.042+(-.012 if r['replicate']==0 else .012);bad=r['fit_failed'] or r['evaluation_failed'];marker='x' if bad else ('o' if r['replicate']==0 else '^')
    ax.errorbar(x+offset,r['logmeanlik'],yerr=1.96*r['MCSE'] if r['MCSE'] is not None else None,fmt='none',ecolor=colors[r['start']],elinewidth=.7,alpha=.6,capsize=0,zorder=3)
    ax.scatter(x+offset,r['logmeanlik'],s=34,marker=marker,color=colors[r['start']],edgecolors='none' if marker!='x' else None,zorder=4)
    if not r['precision_passed']:ax.scatter(x+offset,r['logmeanlik'],s=80,marker='s',facecolors='none',edgecolors='#a31414',linewidths=1.2,zorder=5)
   if len(good)<12:ax.text(x,.98,f'{len(good)}/12 successful',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9)
  ax.set_xticks(range(len(chosen)),[names[m] for m in chosen]);ax.set_title(title,pad=12);ax.set_ylabel('Final log likelihood (higher is better)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.11)
 draw(axes[0,0],methods,'All 36 endpoints; full likelihood range')
 if zoom:draw(axes[0,1],['IF2','IFAD'],'IF2 and IFAD: the same endpoints, expanded scale')
 handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markeredgecolor='none',label=f'Start {i+1}',markersize=6) for i,c in enumerate(colors)]
 handles.extend([Line2D([0],[0],marker='o',color='#555555',ls='none',label='Repeat 1',markersize=5),Line2D([0],[0],marker='^',color='#555555',ls='none',label='Repeat 2',markersize=5)])
 if any(r['fit_failed'] or r['evaluation_failed'] for r in rows):handles.append(Line2D([0],[0],marker='x',color='#a31414',ls='none',label='Failed fit or score'))
 if any(not r['precision_passed'] for r in rows):handles.append(Line2D([0],[0],marker='s',mfc='none',color='#a31414',ls='none',label='Low scoring precision'))
 fig.legend(handles=handles,loc='lower center',ncol=4 if len(handles)>8 else 8,frameon=False,bbox_to_anchor=(.5,.045),fontsize=9)
 fig.suptitle('Dacca: TFP parameter estimates',fontsize=16,y=.98)
 fig.text(.5,.92,'600 months · 20 Gaussian Euler steps/month · J = 5000 · 6 starts × 2 repeats',ha='center',fontsize=11)
 fig.text(.5,.012,'Independent original PyPOMP PF scores. Bars show ±1.96 MCSE; boxes use successful finite fits. All poor endpoints are retained.',ha='center',fontsize=9)
 fig.subplots_adjust(top=.82,bottom=.26,left=.085,right=.98,wspace=.26)
 for ext in ['png','pdf']:fig.savefig(OUT/f'likelihood.{ext}',dpi=190,bbox_inches='tight')
 plt.close(fig)
 # The stacked bar sums exactly to actual Slurm allocation for each method.
 times=s['timing_by_method'];x=np.arange(3);prefix=np.array([times[m]['IF2_synchronized_calls_seconds'] for m in methods]);grad=np.array([times[m]['gradient_synchronized_calls_seconds'] for m in methods]);score=np.array([times[m]['evaluation_process_wall_seconds'] for m in methods]);total=np.array([times[m]['Slurm_allocation_seconds'] for m in methods]);other=total-prefix-grad-score;assert np.all(other>=0)
 fig,ax=plt.subplots(figsize=(10.4,6.1));bottom=np.zeros(3)
 for arr,label,color in [(prefix,'IF2 updates','#577D9F'),(grad,'Adam updates','#348E83'),(score,'Final PF scoring process','#B383B6'),(other,'Other allocation time','#C8CDD3')]:
  ax.bar(x,arr/60,bottom=bottom/60,width=.57,label=label,color=color);bottom+=arr
 assert np.allclose(bottom,total)
 for i,m in enumerate(methods):ax.text(i,total[i]/60+2,f'{total[i]/60:.2f} min total\n{(prefix[i]+grad[i])/60:.2f} min fitting',ha='center',va='bottom',fontsize=10)
 ax.set_ylim(0,total.max()/60*1.2);ax.set_xticks(x,[names[m] for m in methods]);ax.set_ylabel('Actual GPU allocation (minutes)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 ax.legend(loc='upper left',frameon=False,fontsize=10);fig.suptitle('Dacca: measured GPU time in TFP',fontsize=16,y=.98)
 fig.text(.5,.92,'One A40 per method; 12 fits per batch; different fixed iteration counts',ha='center',fontsize=11)
 fig.text(.5,.025,'Other time includes XLA probes, graph preparation, imports, the control PF and I/O. Scoring includes its process initialization.',ha='center',fontsize=9)
 fig.subplots_adjust(top=.84,bottom=.23,left=.1,right=.98)
 for ext in ['png','pdf']:fig.savefig(OUT/f'runtime.{ext}',dpi=190,bbox_inches='tight')
 plt.close(fig)
 lines=['TFP 原始 Dacca：三方法正式比较','', '本候选报告已汇总36个真实终点并完成数值审核；发布状态以delivery_manifest.json为准，独立复核和图像检查分别记录。', '', '固定配置：全部600个月、每月20个Gaussian Euler步、23自由坐标、J=5000；六个共同论文起点，各两次独立重复。', 'IF2为1950次更新；IFAD(alpha=.97)为175次IF2后400次Adam；原生ADPF从原始起点直接425次Adam。三组采用不同固定迭代配置，实际计算时间不同，不能当作等时间比赛。', '', '最终likelihood（独立原始PyPOMP PF；值越高越好）']
 for m in methods:
  q=s['summary_by_method'][m];ti=times[m];v=q['median_successful_finite_logmeanlik'];vstr=f'{v:.6f}' if v is not None else '无有限成功结果'
  lines.append(f"{m}: 成功有限终点{q['finite_success_count']}/12，中位数{vstr}；范围{q['minimum_successful_finite_logmeanlik']}至{q['maximum_successful_finite_logmeanlik']}。拟合失败{q['fit_failures']}、评分失败{q['evaluation_failures']}、精度未达标/失败{q['imprecise_or_failed_scores']}、观测floor饱和{q['score_floor_saturated']}。")
  lines.append(f"  同步拟合调用{ti['fitting_synchronized_calls_seconds']:.6f}秒（IF2 {ti['IF2_synchronized_calls_seconds']:.6f}，Adam {ti['gradient_synchronized_calls_seconds']:.6f}）；Slurm占用{ti['Slurm_allocation_seconds']}秒/{ti['Slurm_allocation_GPU_hours']:.6f}GPU小时。")
 lines.extend(['',f"总计{s['total_raw_PF_draws']}个独立原始PF重复，实际评分入口key跨三组无重复。原始PF J=5000，起始R36；ESS<10或MCSE>1按冻结规则追加至R72/144/288。全部差点、数值失败、低精度评分与扩展前的原始重复均保留。",'IFAD的case6（第四个起点第一次重复）最终log likelihood=-3821.29323342，完成全部400次有限更新，作为较差结果保留，并未记为数值失败。','', '实现与编译选择','原生ADPF使用TFP particle_filter的unbiased_gradients=True和原生Keras Adam。每步systematic重采样是明确配置，不是上游默认ESS<.5J。每成员独立随机key布局由本地薄层实现。','IF2的参数扰动与冷却由本地扩展驱动，SMC与重采样使用TFP；IFAD的alpha=.97梯度折扣也是本地扩展，不能称作TFP开箱即有IFAD。','IF2核通过实测门槛后启用XLA；原生/折扣梯度XLA的可选探测均因反向图Transpose维度不是编译时常量而失败。依冻结协议改用普通TensorFlow GPU图，仍调用原生TFP与Keras；错误及其成本完整保留。native=false/discounted=false仅指XLA选择。','', '时间范围与可比性','每次同步调用包含整批12个filter、完整600个月与J5000。拟合调用时间不含单独预热/图构建、控制PF、评分或I/O。TF拟合进程包含导入、控制PF、准备、拟合和I/O；Slurm实际占用进一步包含XLA探测、评分与进程开销。各范围相互包含，不能直接相加。','IFAD前缀在该arm真实执行一次；独立IF2组另行运行1950次，没有跨方法物理复用前缀。方法比较中的IFAD时间包含全部175次前缀成本。','全部使用A40，但IF2/IFAD位于gl1517，ADPF位于gl1504；并非三个方法都在同一块物理GPU。本图也不提供跨包加速比。',f"三作业实际累计占用{s['total_allocation_seconds']}秒（{s['total_allocation_seconds']/3600:.6f}GPU小时）。",'', '完整性与局限','三作业均有真实sacct COMPLETED/0:0与12终点COMPLETE证据；28个冻结源/输入文件SHA、全部回收文件SHA、内部checkpoint/参数SHA、原始评分与MCSE/ESS已核验。数据与论文起点文件一致，全部600个月保留。','原运行器没有写terminal_manifest.json，因此提供的是作业结束后远程读取前后与本地一致的回收SHA，不声称存在原始terminal manifest。','正式初始Adam快照已被checkpoint轮换清理；新优化器零状态与预热后恢复由冻结源码及此前小规格回归支持，最后优化器状态由真实正式checkpoint核验。','6起点×2重复的固定迭代结果不等于普遍优化器优越性。终点likelihood、计算时间和失败情况必须一起阅读。','', '图与数据','likelihood.png/pdf：完整36点范围；若有右侧放大图，只对同一批IF2/IFAD终点放大，原图仍保留所有ADPF和差点。','runtime.png/pdf：每个作业的真实Slurm占用，分解为IF2、Adam、评分进程与其余时间。','endpoints.csv：全部36终点，包括失败、精度、floor标记。','summary.json：逐方法统计和完整时间口径；delivery_manifest.json：数值、视觉、独立复核及发布门槛。'])
 (OUT/'report_CN.txt').write_text('\n'.join(lines)+'\n')
 gate['figure_sha256']={n:sha(OUT/n) for n in ['likelihood.png','likelihood.pdf','runtime.png','runtime.pdf']};gate['report_sha256']=sha(OUT/'report_CN.txt');gate['visual_review_passed']=False;gate['root_review_passed']=False;gate['release_ready']=False
 (OUT/'delivery_manifest.json').write_text(json.dumps(gate,indent=2)+'\n');print('Saved real-data figures and report; independent/visual review still required.')
if __name__=='__main__':main()
