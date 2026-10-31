# comparisons 本地仓库

按照 Ed 在会议中提出的安排，把软件包修改、验证、实验和报告放在同一个可审阅的仓库中。当前只在本地整理，没有创建远程仓库、发送邮件、push 或提交 PR。

先看 [各包实际调用和新增代码](docs/package_calls.md)，再看 [当前证据与未完成工作](reports/status.md)。英文入口是 [README.md](README.md)。

| 内容 | 位置 |
|---|---|
| Cuthbert、PFJAX、PyDPF、TFP、particles 的候选代码、对应原代码、完整补丁、测试和验证记录 | [`packages/`](packages/) |
| PyPOMP 参考代码，Dynamax、pypfilt 的旧实验代码及说明 | [`packages/`](packages/) |
| 已完成的旧 Dacca 比较，包括论文两包的 800 个终点及其他六包 | [`experiments/legacy_dacca/`](experiments/legacy_dacca/)、[总报告](reports/legacy_dacca/report_CN.txt) |
| 新公共接口的协议、驱动、作业记录、失败日志、particles 的 24 个终点 | [`experiments/public_interfaces/`](experiments/public_interfaces/) |
| 已完成结果的总图、按包分图及隐藏 1.5×IQR 离群点的版本 | [`reports/legacy_dacca/`](reports/legacy_dacca/) |
| 完整原始记录和复现压缩包 | [`artifacts/`](artifacts/)，文件实际在本地仓库内，未纳入 Git |
| 每个文件的来源及 SHA256 | [`manifests/`](manifests/) |

旧实验保留当时实际运行的代码。五包新公共接口单独组织，不能用旧实验的优化结果证明新接口已经验证成功。原包自带函数、我们新增的函数、外部优化器也分别写明。

正确性、原生 PF 耗时、优化效果分别报告。记录中的 Cuthbert GPU 检查仍待完成，新 A/B/C 尚未开始；其余三个 GPU 包有待解决的验证失败。新 particles 的拟合和评分已审计，但最终图和复现交付尚未完成。准确时间与证据见 [状态说明](reports/status.md)。

在仓库内运行 `python3 scripts/verify.py --archives` 可以核验文件、原始压缩包和五份补丁。此命令不重新拟合或评分。原有结果、失败、低精度评分和实际费用均保留。压缩包另有 [SHA256 清单](manifests/archives.json)，单独复制 Git 仓库时需一并复制这些本地文件。
