各包实际调用与我们编写的代码
2026-10-29；针对已完成的原始 Dacca 正式实验

这份说明替代 v1 中过于概括的“我们添加的部分”。它描述实际执行的代码，不表示这些包一般情况下只能这样使用，也不表示我们的新增函数已被上游接纳。本次只修改说明，没有修改实验源、参数、评分或图。

一、表格中应分别写出的内容

1. Dacca 模型：初始状态、每月20步Euler模拟、观测密度、参数变换。PyPOMP一侧调用包内的models.dhaka；其他包使用我们或本项目已有的Dacca代码，按各包接口提供模型函数。
2. IF2：参数存在哪里，何时扰动，谁执行逐月滤波，谁执行外层重复，下一轮接收什么。
3. 梯度：直接调用包原有的梯度功能，还是我们另外加入了权重导数修正；修正加在哪里。
4. 参数更新：谁计算梯度，谁执行Adam，谁设置学习率和保存优化器状态。

“我们实现DMOP梯度”不能代替第3项的具体说明。实际代码是构造带有指定导数的PF计算，再由JAX、PyTorch或TensorFlow自动求导；不是我们手写了整个Dacca似然的解析梯度。
下文alpha表示梯度折扣系数，不是Dacca模型中同名的固定参数。
范围：下表Cuthbert/PyPOMP描述work/cuthbert-dacca-paper的J5000论文实验。早期work/cuthbert-dacca的J1024、180秒GPU实验另有已完成的A/B/C：A/B调用原有stop_gradient_decorator和PF，使用Optax Adam；C使用新增折扣梯度。不能将下表的PyPOMP Adam或新增alpha=1装饰器描述套到早期实验。

二、可用于表格的逐包说明

PyPOMP
  IF2：我们的native_mif函数设置粒子数、迭代数、扰动尺度和冷却参数，调用pypomp.functional.mif。IF2逐月滤波、参数扰动和轮次循环由该包函数执行。
  梯度及更新：我们的native_train函数向pypomp.functional.train传入alpha、学习率和pypomp.Adam。梯度计算和Adam更新循环由该包函数执行。
  两阶段衔接：run.fitting取IF2的最后参数，作为train的输入；创建新的Adam状态。我们设置共同初值、方法顺序、随机数、批量执行、编译与运行计时、结果保存和独立评分。
  模型：调用pypomp.models.dhaka(dt=None,nstep=20)，并按冻结数据和观测长度配置输入。
  代码：work/cuthbert-dacca-paper/kernels.py:71,79；settings.py:35；run.py:230,244。

Cuthbert
  IF2：我们写cuthbert_mif。每个粒子保存{state,parameters}；每轮开始只扰动初值参数，每月传播前扰动其他参数，扰动尺度随轮次和月份减小。我们提供初始化、传播和观测函数，交给原有particle_filter.build_filter。
  逐月滤波：我们写scan_final中的jax.lax.scan，每月调用包内filter_prepare和filter_combine。filter_combine调用原有systematic.resampling、我们的传播/观测函数，并更新权重与累计likelihood。状态和参数使用相同祖先索引。
  外层IF2：我们重复上述整段滤波；每轮末尾再次调用原有systematic.resampling，取得等权的完整参数粒子集合，传入下一轮。参数均值单独保存，在IF2结束后用于梯度阶段；不是每轮把整个参数集合替换成均值。
  梯度：我们新增dmop.log_likelihood和discounted_stop_gradient_decorator。后者先调用原重采样函数，再给返回的log权重加alpha*(selected-stop_gradient(selected))；selected是被选中祖先的log权重。该项的前向值为零，导数保留并乘以alpha。我们在cuthbert_train中调用jax.grad计算负log-likelihood/600的梯度。
  参数更新：我们写更新循环并调用pypomp.Adam.step，再按坐标学习率更新参数。正式Dacca运行没有调用通用iterated_filtering.fit、dmop.fit或ifad.fit。
  原文件边界：原particle_filter.py和systematic重采样代码未改；新增了上述函数及循环。原stop_gradient_decorator仍存在；正式alpha=1分支使用的是新增装饰器并令alpha=1。
  代码：work/cuthbert-dacca-paper/kernels.py:18,51；source/cuthbert/smc/_estimation.py:139；source/cuthbert/smc/dmop.py:71；source/cuthbertlib/resampling/autodiff.py:22。

PFJAX
  IF2：我们写bridge.if2中的两层jax.lax.scan。外层控制IF2轮次及冷却；内层逐月扰动各粒子的参数、调用我们写的Dacca.pf_step进行状态传播和观测密度计算、调用包内resample_multinomial同时重采样状态和参数，并累计log-likelihood。完整参数粒子集合传入下一轮。
  关键边界：这条IF2路径没有调用pfjax.particle_filter，逐月滤波循环由我们编写。
  梯度：这条路径调用完整的pfjax.particle_filter。我们新增CorrectedDacca，在粒子中增加correction字段，并新增corrected_resampler：调用原生multinomial重采样，取选中祖先的log权重，构造alpha*(selected-stop_gradient(selected))，再减去其log-mean-exp；下一月的观测log权重加上该correction。归一化项用于匹配PFJAX的likelihood累积方式。
  参数更新：我们调用jax.value_and_grad计算负log-likelihood/600的梯度；调用Optax的scale_by_adam更新矩估计；我们写循环、应用坐标学习率、保存Adam状态并记录失败。alpha=1和alpha=.97都走上述新增修正函数；没有把原生OT算法用于这两组正式结果。
  原文件边界：除了新增模型和函数，实际使用的particle_filter.py还有已记录的随机key数组索引性能修改，见v1/pfjax_particle_filter.diff；不能笼统写“原PF源码完全未改”。
  代码：work/competitor-estimation/pfjax/formal_v1/bridge.py:36,46,75,96；run.py:42；common.py:70。

PyDPF
  IF2：我们写IF2Session和Prior.sample、Dyn.sample、Obs.score。粒子是8维状态与23维参数拼接的向量。Prior处理初值参数扰动及第一个月；Dyn每月扰动其他参数，再用该粒子的参数模拟Dacca；Obs计算观测log密度。
  逐月滤波：上述函数交给包内FilteringModel和ParticleFilter，整段滤波循环、重采样和likelihood因子的计算由包执行。我们取最后粒子并调用SystematicResampler取得等权参数集合；IF2Session保存完整集合、更新冷却参数并开始下一轮。
  直接ADPF：调用包内StopGradientDPF。我们提供Dacca模型并调用torch.autograd.grad和torch.optim.Adam；这条路径没有使用我们写的DiscountedResampler。
  IFAD梯度：使用ParticleFilter配合我们新增的DiscountedResampler。该类调用原SystematicResampler，再读取其cache中的used_weight和sampled_indices，把alpha*(selected-selected.detach())加到返回的权重上。
  参数更新：我们写AdamSession中的目标函数、自动求导调用、坐标学习率、逐次更新、失败处理和状态保存；Adam的一阶、二阶矩更新由torch.optim.Adam执行。IFAD的前段IF2和后段Adam由我们的fit.py衔接。
  代码：work/competitor-estimation/pydpf/formal_v1/sessions.py:15,60；dacca_model.py:134,154；fit.py:189,207,212。

TensorFlow Probability（TFP）
  IF2：我们写keyed.if2_step，把状态x与参数p放入WeightedParticles。每轮开始扰动初值参数；我们写propose回调，在每月扰动其他参数、执行20步Euler模拟，并把观测log密度加到传入的log权重上。
  逐月滤波：propose交给包内sequential_monte_carlo。包负责SMC循环、权重归一化、重采样和log-likelihood累积；观测密度及未归一化权重的构造发生在我们的propose中。我们写batch_engine.IF2和外层运行循环，保存完整参数粒子集合并控制轮次及冷却。
  直接ADPF：调用包内particle_filter(unbiased_gradients=True)，采用其已有的stop-gradient修正；我们提供Dacca模型和独立重复的随机数，并通过GradientTape求导。
  IFAD梯度：我们写discounted_weights。在TensorFlow建立计算图时，临时替换内部weighted_resampling.resample的Python引用，把target_log_weights替换为stop_gradient(w)+alpha*(w-stop_gradient(w))，再调用原函数，完成后恢复引用。前向权重不变，导数乘以alpha。这不是TFP原有的alpha选项，也不是改写完整particle_filter。
  参数更新：我们写batch_engine.Adam，给23个坐标分别创建Keras Adam并设置学习率，调用apply_gradients，保存状态和失败记录；TFP本身没有在此执行参数优化。
  代码：work/competitor-estimation/tfp/formal_v2/source/keyed.py:22,43,60,69；source/model.py:49；source/batch_engine.py:7,36；run.py:94。

Dynamax
  IF2：我们写engine中的逐轮、逐月循环、参数扰动和冷却，以及bridge.systematic重采样。没有调用Dynamax的粒子滤波函数。
  梯度：我们写local_pf中的重采样、Dacca传播、观测权重、alpha导数修正和likelihood累积。DaccaSGD继承包内SSM，并覆写marginal_log_prob，让它返回我们写的local_pf结果。
  参数更新：调用原有SSM.fit_sgd，传入Optax Adam及我们写的ScalarCheckpointOptimizer，后者保存实际参数、优化器状态、随机数和失败记录。这里使用Dynamax的是训练接口，Dacca PF和IF2均由我们实现。
  代码：work/competitor-estimation/dynamax/dacca/formal_fixed_v1/engine.py:43,71,142,168；bridge.py:16,23,48,61,76。

particles
  模型：提供DaccaSSM的状态模拟和观测密度，并将其交给原有state_space_models.Bootstrap。
  滤波：我们写RecordedSMC继承particles.SMC，仅在调用原compute_summaries后记录时间和loglt；run、重采样和logLt累计使用包内实现。
  参数更新：我们写目标函数，把23维待估参数变换到模型参数，调用上述PF并返回负logLt；SciPy的minimize(method='Nelder-Mead')选择下一组参数。预算、每次PF的随机数、可接受候选的选择和记录由我们控制。
  本次没有实现IF2或IFAD，也没有调用particles的PMMH、SMC2或模型专用EM。
  代码：work/competitor-estimation/particles/formal_v1/support.py:50,84；fit.py:79,113；adapter/bench_particles.py:64。

pypfilt
  模型：我们写DaccaModel的init、update，以及DeathsObs.log_llhd；按包要求提供状态初始化、Euler模拟和观测log密度。每次PF中参数固定，不作为粒子的一部分进行估计，正则化关闭。
  滤波：调用原有pypfilt.fit；逐时更新和重采样由包执行。我们的log_llhd在返回观测log密度给包之前，另外计算logsumexp(观测log密度+更新前归一化权重的log)，把各月结果相加，得到该次PF的log-likelihood估计。这是我们的累计代码，并非读取包原有的marginal-likelihood返回字段。
  参数更新：我们写目标函数及参数变换，把上述负log-likelihood交给SciPy Nelder-Mead，由SciPy选择下一组参数。
  本次没有实现IF2或IFAD；也没有检验pypfilt原生的参数粒子估计、post_regularise或adaptive_fit在Dacca上的表现。
  代码：work/competitor-estimation/pypfilt/formal_v1/support.py:66；fit.py:79,113；adapter/pypfilt_dacca.py:29,73。

三、训练和最终评分

上述调用用于训练。最终终点的likelihood、MCSE和likelihood ESS统一来自独立的原始PyPOMP PF评分。
它们不表示训练过程中也使用PyPOMP PF。这里的Cuthbert论文实验梯度训练确实使用pypomp.Adam，这是优化器调用，与最终评分调用是两回事。
PFJAX最终评分采用evaluation_v2；不使用原formal_v1中被替换的评分。

四、文件证据

代码路径均相对 /Users/mac/Documents/Codex/2026-10-22/cloud-gl。
此前的来源核验、Cuthbert原文件比较和PFJAX修改记录保存在：
outputs/competitor_estimation/package_call_boundaries_v1/source_audit.json
outputs/competitor_estimation/package_call_boundaries_v1/cuthbert_autodiff.diff
outputs/competitor_estimation/package_call_boundaries_v1/pfjax_particle_filter.diff
本版重新阅读上述主要函数；source_recheck.json记录与此前冻结SHA的复核结果。该复核只确认文件来源，不能代替数值正确性实验。
