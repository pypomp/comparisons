Dacca：隐藏箱线图离群点后的图

每个软件包×方法分别用全部有限likelihood估计计算Q1、Q3（线性插值），隐藏小于Q1−1.5×IQR或大于Q3+1.5×IQR的点。共69个点隐藏，892个点显示；完整961个有限评分、968次优化、9次失败均保留。全部4个数值下限点被同一规则隐藏。

箱体、须、中位数、失败/精度计数和计算成本仍使用全部原始结果，不根据绘图筛选重算。隐藏仅改变散点及误差条显示和坐标范围。按分组的阈值与隐藏记录见outliers_hidden.json。原来包含全部点的两张图保持在../overview_v1。

dacca_overview为总览，dacca_by_package按包展示。两图均明确标注隐藏规则与数量，不称为全部结果图。总图F表直接列出包内函数、我们添加的部分及外部优化器；分包图在每个面板下标注实际调用方式。added here表示本次添加的代码。PFJAX调用包内particle_filter，梯度修正由我们添加；Dynamax调用SSM.fit_sgd，Dacca PF由我们实现。原生PF不代表整套参数估计方法来自原包。四个新增GPU包的直接alpha=1方法没有IF2初始化，与论文IFAD1不同。具体对应见method_names.json。

数据：../final_comparison/results.json；未重新拟合或运行PF。
复现：
MPLCONFIGDIR=/private/tmp/dacca-matplotlib-cache PYTHONPATH=work/competitor-estimation/dynamax/plot-deps work/panel-cute-cpu/bin/python work/competitor-estimation/plot_overview.py --hide-iqr-outliers
