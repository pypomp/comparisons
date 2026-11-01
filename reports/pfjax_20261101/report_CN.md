# PFJAX Dacca A/B/C：中文辅助说明

已完成并审核全部12个case、36个终点和1548个原始独立PF评分。六个共同起点各两次重复，C的独立log likelihood在12对中都高于B，配对差中位数3.762021（范围0.407534–20.147663）。B在11对中高于A，case1比A差。这只是本次起点、设置和预算下的观察，不能推出普遍优势。

| Method | Complete fits | Nonfloor scores meeting precision | Median log mean likelihood, all 12 | Median fitting kernels, complete fits (s) |
|---|---:|---:|---:|---:|
| A: Adam, α=1 | 11/12 | 11/12 | -3793.495698 | 529.630 (n=11) |
| B: IF2 + Adam, α=1 | 12/12 | 12/12 | -3748.631853 | 569.591 (n=12) |
| C: IF2 + Adam, α=0.97 | 12/12 | 12/12 | -3745.089672 | 569.732 (n=12) |

表内likelihood中位数保留全部12点。A有1次真实失败（case8，116次有效更新后失败），其评分落到密度下限−24867.998047；MCSE0、ESS36不能解释成好结果。35个非floor终点评分满足预定精度，case6 A到R288后通过。A完整11次子集的likelihood中位数为−3772.480315；失败计算154.550秒保留，不算成功加速。B/C均完整175次IF2加400次Adam；A只有Adam，因此A/B总工作量不同。

失败提案完整保留：loss有限，但23梯度、23提案参数、46Adam矩和23/28自然参数为NaN。已确认出问题的字段，具体底层算子未定位；不重跑或挑最好loss。B/C相同IF2终点、零Adam、学习率和训练随机流已核验。

实际调用新增公开`iterated_filter_step`与`particle_filter_mop`，外部Optax更新；复用原包完整PF时间循环。alpha1导数修正也是新增代码，不是PFJAX原生ADPF。IF2保留每case最终完整5000×23 swarm和175pass acceptance/LL记录，不包含每pass完整粒子群。23坐标含5个归一化初值坐标，有冗余。

正式Slurm共20481GPU秒，加293+194+354检查共21322秒，在原42480秒预算内；并行资源秒相加不是墙钟等待时间。共享IF2物理只算一次，B/C方法逻辑耗时各含完整前缀；核函数时间不含编译、Python/IO或独立评分。原生PF时间单独报告，不重新计时。

[英文报告](report.md)给出全部36点、调用细节、失败与验证限制、三个PNG/PDF图和复现路径。全范围图保留所有点。可选1.5IQR图只隐藏A6/8、B1/7、C7，原始数据及所有统计不删除。总审核核验1627评分keys与6936枚举训练keys无冲突，不宣称穷举模型内部所有随机draw。

旧GPU检查失败原样保留。同执行方式原版/公共likelihood控制324/324相同，而原版本身forward-only和value_and_grad也存在差异，底层原因未确定。按用户接受likelihood一致后的授权完成实证比较；这不认证数学梯度，不把旧失败改称通过。完整源码、补丁、失败、状态、评分和审计在`pfjax_review_20261101.tar.gz`，只作本地作者审阅，不发布。
