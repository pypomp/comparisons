particles 新IF2的Dacca比较（CPU预检通过，冻结候选待父审核，尚未提交）

A：原生PF + 外部SciPy Nelder-Mead。B：新IF2前900秒 + 相同NM，合计1800秒。
6共同起点×2独立重复×2组=24次拟合，J5000，600个月，每月20个Gaussian Euler步，23个自由坐标。
新增接口是真实if2_step(vectorized=True)，内部调用原SMC/Bootstrap；没有改写PF。
NM不是particles原生优化器。新vectorized适配器复用冻结Dacca.step/interval/dmeas/PX0/PX/PY；仅广播参数、初值和按区间计算系数。
IF2完整swarm与NumPy RNG每pass保存。迟到pass/目标调用全部计费但参数不接受。失败保留并不重试。
12 CPUh为设定拟合预算；24个45min任务的最大allocation为18 CPUh。预检成本另列。
全部24终点结束后，单独A40作业最多30min统一原PyPOMP评分，GPU评分时间不算CPU优化加速。
独立评分固定A0/B1方法码，score root94981000，R36→72→144→288；低精度与失败全保留。
训练root24981000；同模型/起点/总时间，不声称相同迭代数或所有包的原生优化器比较。
