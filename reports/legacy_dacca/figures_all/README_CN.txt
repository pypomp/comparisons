原始 Dacca 全包总览

主图 dacca_overview 将100起点论文实验、6起点乘2重复新增GPU实验、CPU1800秒实验分别展示，附拟合耗时、接口归属与实际资源占用。

主图A/B为放大范围；每个范围外有限评分均以三角保留。全范围补图 dacca_all_endpoints 展示961个有限评分，各面板纵轴范围分别设置。7个非有限拟合无评分，仍计入968条记录和9次失败。低精度和底值饱和不作成功拟合解释。

拟合批时间只计一次。Dynamax耗时是12次标量拟合之和且包含原生SGD内部编译。CPU/GPU分开；不能将配置、批处理、原生接口及编译口径不同的耗时当作统一库性能排名。

数据源：../final_comparison/results.json；原交付不修改，不重新拟合或运行PF。数值审计与源SHA见numeric_audit.json。独立评分误差、失败和原生/扩展边界完整记录在原始报告及7份复现包。

复现命令（项目根目录）：
MPLCONFIGDIR=/private/tmp/dacca-matplotlib-cache PYTHONPATH=work/competitor-estimation/dynamax/plot-deps work/panel-cute-cpu/bin/python work/competitor-estimation/plot_overview.py

方法名称：package 表示软件包提供的代码，custom 表示本次添加的代码。IF2、IFAD分别标注实现来源。直接alpha=1方法称为ADPF + Adam，并注明package PF或custom PF；它从原始起点运行425步Adam，没有IF2初始化，不能称为论文IFAD1。
PyDPF和TFP使用包提供的可微PF；PFJAX的正式三方法均为本地扩展；Dynamax仅优化接口SSM.fit_sgd来自原包，Dacca PF由本地实现。particles和pypfilt为Package PF + SciPy Nelder-Mead。具体记录名称到图示名称的对应关系见method_names.json。
