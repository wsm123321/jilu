# 第二步冻结协议：二维无噪声少样本可辨识性 Pilot

## 1. 唯一研究问题

> 在局部图表和输出尺度已经明确规定的 Oracle 表示层，规定图表中的曲率量级、各向异性谱和固定探针排序签名，能否从少量、偏置但无噪声的二维目标样本中稳定估计？

本步不研究源—目标迁移、GP、BO、MAB、区域提名或高维完整二次估计。

## 2. 术语与表示冻结

生成模型为

\[
x=a+Bu,\qquad y=m+s\,g(u),\qquad
 g(u)=\frac12u^\top Ku,
\]

其中 \(u\in[-1,1]^2\)、\(s>0\)、\(K\succ0\)。规范曲率只采用

\[
K=\frac1sB^\top H_xB.
\]

本步称 \(\|K\|_F/\sqrt 2\) 为“相对于规定局部图表的曲率量级”，不得称为数学光滑性或 GP 长度尺度。

### Oracle 表示层

生成器给出真实 \(B,s,m\)，样本先精确映射为 \(u=B^{-1}(x-a)\)，输出先归一化为 \(z=(y-m)/s\)，再估计 \(K\)。本步主实验只识别估计器和采样设计的影响。

### Operational 表示层

实际规则得到 \(\widehat B,\widehat s\)。由于其误差会与曲率估计误差混合，本步不把它纳入主 Pilot；后续必须作为独立实验，比较 Oracle \(B,s\)、已知区域边界、样本协方差图表和冻结输出尺度规则。

## 3. 景观网格

曲率矩阵为

\[
K=Q_\theta\operatorname{diag}(1,r)Q_\theta^\top,
\]

其中

\[
r\in\{1,4,16\},\qquad
\theta\in\{0,\pi/8,\pi/4\}.
\]

各向同性 \(r=1\) 下方向没有物理区别，但仍保留三个标签用于检查实现是否错误地产生方向效应。共 9 个冻结景观条件。

## 4. 采样设计

每个设计均在规范域 \([-1,1]^2\) 中生成，不主动包含真实最优点 \(u=0\)。

1. `sobol`：先生成长度为 \(2^{\lceil\log_2n\rceil}\) 的 scrambled Sobol net，再取前 \(n\) 点；非二次幂样本量称为 Sobol prefix，不宣称保持完整 net 的标准 balance property；
2. `uniform`：二维独立均匀随机；
3. `trajectory`：先生成两个全域点，随后围绕当前已观测最好点、以逐步收缩半径生成点，模拟偏置、相关和局部聚集的轨迹，不声称代表完整 BO。

样本量

\[
n\in\{3,5,6,8,12,20\}.
\]

每个条件使用 50 个成对 seed；同一 `seed × landscape × n` 在不同估计条件中保持一致。

## 5. 二次估计器

设计矩阵每行为

\[
X(u)=
\begin{bmatrix}
1&u_1&u_2&\frac12u_1^2&u_1u_2&\frac12u_2^2
\end{bmatrix}.
\]

通过 SVD 最小二乘估计

\[
z=\beta_0+\beta_1u_1+\beta_2u_2
+\frac12H_{11}u_1^2+H_{12}u_1u_2+\frac12H_{22}u_2^2.
\]

输出对象必须包含：截距、梯度、Hessian、秩、奇异值、设计矩阵条件数、是否可辨识和 `abstain_reason`。

### Abstain 规则

- `n < 6`：`insufficient_samples`；
- SVD 数值秩小于 6：`rank_deficient`；
- 输入非有限或维度错误：抛出明确异常，不生成伪估计；
- 满秩时返回估计和条件数。本 Pilot 不预先冻结条件数阈值，避免事后用阈值制造成功；先报告误差—条件数关系。

主分析不将非正定估计结果投影为 SPD。SPD 投影只能作为以后独立估计器对照。

## 6. 独立恢复验证

为避免解析构造与解析恢复循环验证，必须增加：

1. 独立函数 `recover_canonical_curvature(H_x, B, s)`，直接接收外部矩阵；
2. 从函数值中心有限差分估计物理 Hessian，再恢复 \(K\)；
3. 随机 SPD 曲率、随机正交方向、随机条件数和随机可逆图表测试。

## 7. 指标

每条 run 记录：

- `identifiable` 与 `abstain_reason`；
- 设计矩阵秩、最小/最大奇异值和条件数；
- 曲率矩阵相对 Frobenius 误差；
- 曲率量级相对误差；
- 归一化谱 \(L_1\) 误差；
- \(\log\kappa\) 绝对误差；
- 固定探针成对次序准确率；
- 近 tie 比例。

排序签名由拟合模型在冻结探针上的预测得到，不额外消耗本 Pilot 的目标评估。它是“模型预测的固定探针排序签名”，不是直接评估探针的在线属性。成对差小于冻结容差 \(10^{-10}\max(1,\operatorname{range}(z))\) 时记为 tie 并从确定性比较中 Abstain。

每个条件汇总：

1. 可估计率；
2. 条件于可估计的中位和 90% 分位误差；
3. 最差误差；
4. 条件数与误差关系；
5. 同一景观跨采样器差异与不同景观间差异。

## 8. Pilot 暂定验收线

这些阈值用于解释 Pilot，不在运行后修改：

- \(n<6\) 时完整二次模型 100% `Abstain`；
- 无噪声、满秩且条件良好的设计下误差接近数值精度；
- \(n\le8\) 的满秩率至少 80%；
- 归一化谱中位误差 < 0.1、90% 分位误差 < 0.25；
- Sobol 与均匀随机的定性结论一致；
- 轨迹型采样若显著恶化条件数或误差，应如实归类为采样器敏感性；
- 不以成功子集误差替代全体可估计率。

属性结论只能分为：`Early-transfer candidate`、`Mid-stage candidate`、`Sampler-dependent state` 或 `Not operationally identifiable`。本步由于只完成 Oracle 表示层，最多授予“Oracle-layer candidate”，不能直接授予实际迁移资格。

## 9. 输出与身份

- 原始长表：`results/identifiability_pilot.csv`；
- 条件汇总：`results/identifiability_summary.csv`；
- 四张图：可估计率—样本数、谱误差—样本数、误差—条件数、采样器方差对比；
- 配置与运行元数据：`results/identifiability_manifest.json`；
- 结论：`SECOND_STEP_RESULT.md`。

运行脚本不得静默覆盖：只有显式 `--overwrite` 才可重建冻结路径。结果必须记录代码提交、Python/NumPy/SciPy 版本、seed 数和完整网格。
