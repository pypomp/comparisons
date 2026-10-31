# 各包是否使用原生 PF：源码复核

本次重新检查了候选代码、对应原代码、完整补丁及实验驱动。结论是：**五包新接口均在原包 PF 上扩展，但复用的层次不同。** Cuthbert 复用逐步更新内核；另四包复用完整滤波循环。没有发现另一个像已排除的 Dynamax 那样，主要借用优化器而自行实现整个 Dacca PF 的新接口。

两处必须单独说明：旧 PFJAX 的 IF2 使用了本地滤波循环；当前 Cuthbert 的时间循环仍是新增代码。不能把这些写成“完整调用原生 PF”。

## 五包当前实现

| 包 | 当前调用链 | 原包负责什么 | 本地新增或修改什么 |
|---|---|---|---|
| Cuthbert v3 | 新 `iterated_filtering.fit` / `dmop.log_likelihood` → 原 `particle_filter.build_filter` → 新 `scan_final` → 原 `filter_prepare` / `filter_combine` | 每一步重采样、状态传播、观测权重与 likelihood 更新；α=1 调用原 `stop_gradient_decorator` | 参数与状态一起存储、参数扰动、冷却、IF2 外层循环、末端参数集合；只保留终态的 `jax.lax.scan` 时间循环；α<1 梯度折扣 |
| PFJAX v2 | 新 `iterated_filter_step` / `particle_filter_mop` → 原 `particle_filter` | 完整 PF 时间循环、原 `filter_step`、原多项重采样 | `initial_particles` 入口；参数化模型和参数扰动、IF2 外层循环；重采样梯度修正，包括 α=1 |
| PyDPF v1 | 新 `IteratedFiltering` → 原 `ParticleFilter` → 原 `SIS.forward`；梯度阶段调用扩展后的 `StopGradientDPF` | 完整时间循环、重采样、权重与 likelihood；原 `StopGradientDPF` 已有 α=1 梯度处理 | 参数化模型包装、参数扰动、冷却与 IF2 重复；给 `StopGradientDPF` / `StopGradientResampler` 增加 `alpha` |
| TFP v1 | 新 `iterated_filter_step` → 原 `sequential_monte_carlo`；梯度阶段走扩展后的原 `particle_filter` → SMC | 完整滤波循环、`SequentialMonteCarlo.one_step`、原重采样与已有的无偏梯度处理 | 状态与参数共同传播、扰动和 IF2 重复；传递 `gradient_discount` 并修改权重导数；按起点分配随机流 |
| particles v1 | 新 `if2_step` → 原 `SMC.run`，模型包装调用原 `Bootstrap.M0/M/logG` | 完整时间循环、重采样、权重归一化和 `logLt` 累积 | 参数随状态一起传递、扰动、冷却与 IF2 外层循环；实验外接 SciPy Nelder–Mead |

这些新增函数位于本地候选包中，尚不表示上游自带或已经接受。Dacca 的状态模拟和观测密度由模型回调提供；编写模型回调与自行编写 PF 是不同的修改。

## 需要避免的误解

**Cuthbert。** 原 `cuthbert/smc/particle_filter.py`、多项和系统重采样文件逐字未改，原 `stop_gradient_decorator` 函数也逐字未改。但是新接口没有直接调用顶层 `cuthbert.filter`；`scan_final` 自行遍历时间并调用原更新函数。因此适当描述是“基于 Cuthbert 原 PF 更新内核扩展 IF2 和梯度折扣”。正式 A/B/C 驱动使用新 `iterated_filtering.fit`、`dmop.log_likelihood` 和外部 Optax，未调用完整 `ifad.fit`。

**PFJAX。** 旧 `formal_v1/bridge.py:if2` 中的 `month` 及 `jax.lax.scan(month, ...)` 是我们写的，只复用原重采样。它不满足完整复用 PF 循环的标准，旧结果仅能按该实际实现解释。新接口确实改为完整调用原 `particle_filter`，原嵌套 `filter_step` 逐字未改；原文件增加了初始粒子入口。新驱动只从旧 bridge 取 Dacca 模型类，没有调用旧 `bridge.if2` 或 `bridge.objective`。此外，新旧梯度阶段的 α=1 修正均来自我们，不能写成“PFJAX 原生 ADPF”或原生 OT 方法。

**PyDPF。** `SIS` 与 `ParticleFilter` 两个类逐字未改。实际新驱动通过 `api_adapter.warm_if2` 调用新增 `IteratedFiltering`，通过 `api_adapter.factors` 调用 `ParticleFilter` / `StopGradientDPF`。历史模型文件中留有旧实验函数，不等于新驱动在使用它们。参数优化使用外部 `torch.optim.Adam`。

**TFP。** IF2 调用底层完整 SMC 循环，不能写成调用顶层 `particle_filter`；梯度阶段才调用后者。按起点分配随机流的重采样包装仍调用原 `tfp.experimental.mcmc.resample_systematic`，没有自行实现祖先抽样算法。`gradient_discount` 修改原梯度处理分支，原重采样文件未改。参数优化使用外部 Keras Adam。

**particles。** 原 `core.py`、`state_space_models.py`、`resampling.py` 逐字未改。新增 IF2 包装调用原 `SMC` 和 `Bootstrap`。Nelder–Mead 来自 SciPy，不能称作 particles 自带的 MLE 算法，也没有可微 PF 比较。

## 另外两个包

**PyPOMP** 是参考实现。已将本仓库 `functional/pfilter.py`、`mif.py`、`train.py`、`mop.py` 与固定提交 `17f8798e7d944977335c81b8525353fa90e14be9` 比较，四个文件完全一致。旧论文实验使用包内 IF2 / 训练函数，正式终点评分使用其冻结的原 PF。

**pypfilt** 的旧实验实际调用 `pypfilt.fit`，后者调用原 `pfilter.run`。我们写 Dacca 模型回调和 likelihood 增量记录，外接 SciPy Nelder–Mead；没有自行改写 PF 时间循环，也没有提供本包新 IF2 或 IFAD 接口。为便于检查，已从原复现包中只读取出 `fit`、PF 和重采样源码并记录成员 SHA。

## 核验依据与限度

- [调用位置](source_locations.md)链接到实际源码与行号，涵盖新增接口和实验驱动。
- [机器可读审计](audit.json)包含 15 项原代码逐字比较、25 处调用位置以及相关文件 SHA256。
- [补充源码来源](evidence_sources.json)记录旧 PFJAX、TFP 驱动、particles 驱动、pypfilt 原始归档成员和 PyPOMP 提交。
- 运行 `python3 scripts/audit_native_pf.py` 可重复源码核对；运行 `python3 scripts/verify.py` 可核验完整补丁重建候选代码。

这是源码与调用关系审计，没有重新运行 GPU 测试、拟合或评分。它不解决已有的 GPU 验证失败，也不证明新接口优化效果。正确性、原生 PF 耗时、优化效果继续分开报告。各包初始扰动、首区间和冷却细节也不完全相同，不能称跨包完全同一算法。
