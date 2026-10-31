# 在本机浏览 comparisons

浏览界面和主要审阅文档以英文为主，本页是中文辅助说明。

在终端运行：

```bash
cd /Users/mac/Documents/Codex/2026-10-22/cloud-gl/comparisons
python3 scripts/browse.py
```

浏览器打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。页面提供目录、文件搜索、Markdown 阅读、带行号的源码、红绿补丁、图像预览和原文件下载。左侧每个候选包都有“Diff”入口。

建议先看“Package calls and additions”和“Native PF source audit”，再打开对应包的 `review/full.patch`。`upstream/` 是对应原代码，`candidate/` 是含新增接口的候选代码，`validation/` 是验证记录。实验结果在 `experiments/` 和 `reports/`；旧实验与新接口分开。

这是本地只读浏览器，只监听 `127.0.0.1`，没有 GitHub 远程仓库或发布功能。停止服务可在启动它的终端按 Ctrl+C；之后重新运行上面的命令即可。复制 Git 仓库到别的电脑时，另需复制 `artifacts/` 中的原始压缩包，清单见 `manifests/archives.json`。

浏览器不运行实验，也不更新调度状态。目录索引在启动时生成；新增文件后重新启动即可显示。
