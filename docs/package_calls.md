# 各包实际调用和新增代码

下表描述 `packages/` 中的当前公共接口候选版本。它与旧实验驱动不同。旧实验的逐项调用见 [旧实验调用核对](../reports/package_calls_legacy/table_CN.txt)；旧结果所用的完整源代码在相应复现包中。

Dynamax 已按用户决定从主比较中移除，不再进行后续实现或实验。旧调用表仍含其历史记录，不代表当前范围。

已重新逐包检查候选代码、原代码和实际实验驱动，详见 [2026-10-31 源码复核](../reports/native_pf_audit_20261031/report_CN.md)与[可点击的调用位置](../reports/native_pf_audit_20261031/source_locations.md)。旧 PFJAX 的 IF2 用本地滤波循环，新版才改为完整调用原 PF；Cuthbert 当前仍使用新增时间循环调用原更新内核。

| 包 | 原包已有、实际复用的函数 | 我们新增或修改的代码 | 优化如何执行 |
|---|---|---|---|
| Cuthbert v3 | `particle_filter.build_filter` 产生的原 PF 更新函数；原重采样函数；α=1 时原 `stop_gradient_decorator` | 状态与参数共同存储、扰动、冷却和 IF2 外层循环；新 `scan_final` 自行遍历时间，逐步调用原 `filter_prepare` / `filter_combine`，不直接调用顶层 `cuthbert.filter`；新 `dmop.log_likelihood` 和 α<1 梯度折扣 | 正式驱动调用新 IF2 和 `dmop.log_likelihood`，外接 Optax `scale_by_adam`；没有调用完整 `ifad.fit` |
| PFJAX v2 | 完整 `particle_filter` 的原时间循环与 `resample_multinomial`；通过 `pf_init` / `pf_step` 调用我们提供的 Dacca 模型 | 给原 PF 增加 `initial_particles` 入口；`iterated_filter` 包装状态与参数、扰动、冷却并重复原 PF；`particle_filter_mop` 包装重采样及模型，加入梯度修正，包括 α=1 | 实验驱动外接 Optax Adam。α=1 修正也是新增代码，不能称原生 OT 方法 |
| PyDPF v1 | 原 `ParticleFilter`、`SIS.forward`、`FilteringModel`、重采样及 likelihood 计算；原 `StopGradientDPF` 的梯度修正 | 新 `IteratedFiltering` 用模型包装让状态和参数共同重采样，保留完整参数集合并加扰动、冷却；给 `StopGradientDPF` / `StopGradientResampler` 增加 `alpha` | 外部 `torch.optim.Adam` 循环；未把该循环称为原包已有的 MLE 接口 |
| TFP v1 | 原 `sequential_monte_carlo`、`SequentialMonteCarlo.one_step`、`WeightedParticles` 和重采样；原可微 `particle_filter` | 新 `iterated_filter` 包装参数化模型，在原 SMC 时间循环外重复滤波；沿 `particle_filter` → SMC 传递新增 `gradient_discount`，在权重导数中应用折扣 | 外部 Keras Adam 循环；没有加入虚构的月度转移密度 |
| particles v1 | 原 `SMC.run` 的时间循环、`Bootstrap` 的模型调用、重采样和 `logLt` | 新 `iterated_filtering` 和 `_PerturbedBootstrap` 让粒子携带状态与参数，加入扰动、冷却、整组参数传递；原有 Python 文件保持不变 | 新实验 A 是原 PF + 外部 SciPy Nelder–Mead；B 先调用新 IF2 再用同一个外部优化器 |
| PyPOMP | 旧论文实验调用 `functional.mif`、`functional.train` 及其 Adam；终点评分调用固定原始 PF | 模型、参数设置和实验编排；本仓库不宣称新增 PyPOMP 公共接口 | 包内优化与独立 PF 评分分别记录 |
| pypfilt | 旧实验通过 `pypfilt.fit` 执行原生 PF | Dacca 模型回调、likelihood 增量累计、外部优化和记录 | 外部 SciPy Nelder–Mead；本次没有原生 IF2、IFAD 或 MLE 比较 |

“新增公共接口”指本地候选代码提供的函数，不表示原版本已经自带，也不表示上游已接受。每个包目录包含完整补丁、函数位置和原有代码保持不变的核验记录。

最终优化评分统一来自冻结的原始 PyPOMP PF，不来自各包的训练 loss，也不来自带参数扰动的 IF2 likelihood。各包的小型 API 示例可能用本包 PF 作检查；这与正式实验的独立终点评分是两件事。
