# PyDPF Dacca 比较：中文辅助说明

36个拟合全部完成400次accepted Adam更新，无拟合失败、拒绝提案或整批执行异常；来源、参数/Adam、B/C配对、随机流、全部1548个独立PF值和费用总审核通过。独立评分为34个非floor合格点、1个A低ESS点、1个B密度下限点。完成400步不代表终点一定好。

A/B/C全12点median LL为−3806.946514 / −3746.804253 / −3744.445235。C−B在12对中全部为正，median 2.240650；case7的巨大差值20993.210336来自B落在floor。去掉该floor配对作另外说明，11个非floor合格配对仍全部为正，median 2.031365、范围1.109046–4.688133。全部12对的统计始终保留；这只是六起点各两次重复，不是普遍优势或显著性认证。

B−A在11/12对为正，median28.643451，但case7 B比A低20599.161966。C−A在12/12对为正，median61.913803，包含A低精度点。A不含IF2，总工作量较少；BC同IF2终点、空Adam、学习率与400训练seed及generator-after hash，不能把A/B当等计算预算比较。

A case9(start4rep1)到R288仍ESS1.186695、MCSE.917676、LL−4243.528972，因此保留为低精度。B case7(start3rep1)虽完整400步、accepted参数和Adam有限，36个独立PF值全等floor−24867.998047，MCSE0ESS36不是好拟合或有意义高精度，也不能改称优化器失败。C case7为−3874.787711，MCSE.270386、ESS10.115754合格但仍差于C其他点。其余全部R36；没有非有限raw。

原DaccaT600、每月20GaussianEuler、J5000、23坐标（18过程观测+5归一化IVP，有冗余）。A原点alpha1Adam400；B公共IF2175+alpha1Adam400；C同IF2点alpha.97Adam400。使用原ParticleFilter/SIS.forward和StopGradientDPF，public IF2及discount参数为本地新增，外部torch.optim.Adam；并非原包自带IFAD。生产float32/systematic/checkpoint/default非确定设置不改。

A/B/C整批gradient kernel秒24031.036/20411.694/20170.608；BC加IF2后的逻辑kernel秒40525.585/40284.499。物理warm20216+A24512+B20861+C20719=86308，加原启动失败181，共86489≤108000原30GPUh；验证2522另计，合计89011。批时间不能乘12或当每条终点耗时；共享prefix物理一次，BC逻辑各含完整prefix；费用是SlurmElapsedRaw资源和非墙钟。

总审核核对812个计划Torchseed和1627个JAX评分key分别唯一，不把Torch/JAX当同一空间或枚举内部draw。所有最终5000x23 IF2 swarm、176mean-history和175LL、生成器bytes及初末Adam保留；并未保存每步全部swarm和所有成功提案。旧普通PF/nativeDPF差558.884766未解释；原DPF/public48forward和RNG完全相同。用户据此接受实证比较，旧FAILED仍保留，无数学梯度认证。181秒旧startup失败发生在正式IF2前，未重跑完成的IF2。

三图已视觉核验；IQR仅显示隐藏A7/9、B7、C7，不删raw或统计。全范围图和配对图均保留floor；另外11对放大图清楚标明只隐藏该floor配对。英文报告附全部36终点表。完整复现包包含所有实际源/原始记录/失败/可读diff/环境/审计和绘图程序，解压复算不调用PF或优化器。详见[英文报告](report.md)。
