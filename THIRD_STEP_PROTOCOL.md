# 第三步冻结协议：噪声、采样几何与模型失配下的统计可辨识性

## 1. Gate 3 主问题

> 在 Oracle 局部图表与输出尺度已知时，观测噪声、采样几何和局部中心非驻点如何共同决定曲率量级与各向异性谱的统计可辨识性？目标数据自身能否可靠判断何时应拒绝输出属性？

本阶段不研究 GP、BO、源—目标迁移、Operational \(\widehat B,\widehat s\) 或高维完整二次模型。二次模型预测的固定探针排序只作为派生诊断，不作为第三个独立属性。

## 2. 正交景观网格与共同随机数

规范曲率冻结为

\[
K(q,r,\theta)=q\sqrt2\,Q_\theta
\frac{\operatorname{diag}(1,r)}{\sqrt{1+r^2}}Q_\theta^\top,
\]

从而

\[
\|K\|_F/\sqrt2=q,\qquad \kappa(K)=r.
\]

第三步 A 使用：

- \(q\in\{0.5,1,2\}\)；
- \(r\in\{1,4,16\}\)；
- \(\theta\in\{0,\pi/8,\pi/4\}\)；
- 梯度状态 \(b=0\) 或 \(b=q[0.6,-0.4]^\top\)；
- \(n\in\{6,8,12,20\}\)；
- 相对噪声 \(\eta\in\{0,0.01,0.05,0.10\}\)，\(\sigma_z=\eta q\)。

共同随机数规则：

1. Sobol prefix 与 Uniform 的同一 `split × design × n × seed` 在所有 \(q,r,\theta,b,\eta\) 下使用相同采样点；
2. 同一 `split × design × n × seed` 使用同一标准正态噪声前缀，再乘 \(\eta q\)；
3. 不同 \(n\) 必须使用同一最大样本序列的前缀，保证嵌套比较；
4. Trajectory 使用相同初始点、随机增量与潜在噪声流，但允许因观测值不同自然分叉；轨迹生成不得窥视未来噪声；
5. 开发 seed 为 0—49，保留 seed 为 10000—10099，二者禁止混用。

## 3. 第三步 A：精确二次模型下的噪声压力

景观为

\[
g(u)=c+b^\top u+\frac12u^\top Ku,
\qquad z=g(u)+\epsilon,
\qquad \epsilon\sim\mathcal N(0,\eta^2q^2).
\]

### 3.1 估计器

继续使用不作 SPD 投影的 SVD 完整二次 OLS。除现有输出外，增加：

- 残差、SSE、残差自由度；
- Oracle 噪声协方差；
- Residual 噪声协方差；
- Hessian 系数协方差；
- delta-method 曲率量级标准误；
- 参数化 bootstrap 的量级、谱和最小特征值区间。

冻结 bootstrap 次数为 200；每条 run 的 bootstrap RNG 由完整条件的稳定哈希派生，不与数据生成 RNG 共用。主网格计算成本过高时，bootstrap 仅用于开发阈值子集和保留集门控所需记录，并在 manifest 中明确范围。

### 3.2 符号与类型指标

必须记录：

- 最小特征值；
- `is_spd`；
- 正、负、零特征值个数；
- `inertia_mismatch`；
- 负特征值比例；
- 真实盆地被误判为鞍点/非正定的比例。

归一化绝对谱只反映相对幅度，不能替代符号指标。

### 3.3 排序派生诊断

固定探针对只使用二次模型预测，不额外计入目标评估。报告：

\[
\text{pair coverage}=\frac{\text{未弃权点对数}}{\text{全部点对数}},
\]

\[
\text{selective accuracy}=P(\text{顺序正确}\mid\text{未弃权}).
\]

不同不确定性阈值形成风险—覆盖率曲线。不得只报告跳过大量点对后的条件准确率。

## 4. 选择性准入规则

可靠事件冻结为

\[
\mathcal R=\{E_q\le0.2,\ E_{\mathrm{spectrum}}\le0.15,\ \widehat K\succ0\}.
\]

比较：

1. **Rank-Only**：满秩即接受；
2. **Condition-Gate**：满秩且 \(\kappa(X)\le\tau_\kappa\)；
3. **Oracle-Uncertainty-Gate**：使用真实 \(\sigma_z\) 的相对标准误/区间宽度；
4. **Residual-Uncertainty-Gate**：使用 SSE/(n-6)；当 \(n=6\) 时必须 Abstain，因为残差自由度为 0；
5. **Combined-Gate**：条件数和 Operational 不确定性同时通过。

开发集只用于冻结候选阈值。Condition 候选阈值取开发数据条件数的预设网格 \(\{10,20,50,100,200,500,1000,\infty\}\)；相对标准误阈值取 \(\{0.05,0.10,0.20,0.30,0.50,\infty\}\)。选择规则预先定义为：在开发集 selective risk 不超过 10% 的候选中选择 coverage 最大者；若无候选满足，则该 gate 输出“无安全阈值”。平局选择更严格阈值。阈值冻结后原样用于 100 个全新保留 seed。

每个 gate 在保留集报告：

\[
\text{coverage}=P(\text{接受}),
\qquad
\text{selective risk}=P(\neg\mathcal R\mid\text{接受}).
\]

同时报告分 \(n,\eta,design,b\) 的风险，避免总体平均掩盖危险角落。

## 5. 第三步 B：无噪声非二次模型失配

景观为

\[
g(u)=c+b^\top u+\frac12u^\top K_0u+\beta\sum_{j=1}^2u_j^4,
\]

其中 \(\beta/q\in\{0,0.05,0.20,0.50\}\)，区域尺度 \(\rho\in\{0.25,0.5,1.0\}\)，采样点为 \(u=\rho v\)。本实验无观测噪声，隔离模型失配。

必须同时定义：

- 中心本征曲率 \(K_0=\nabla^2g(0)\)；
- 给定设计分布与区域尺度下的 dense 最优二次投影 \(K^\star_{\mathcal D,\rho}\)。

对 Sobol/Uniform，用冻结的独立 dense 设计近似其分布投影；对 trajectory，投影目标必须绑定其经验采样机制/路径定义，不得假装存在采样器无关的唯一投影。

分解

\[
\widehat K-K_0=(\widehat K-K^\star_{\mathcal D,\rho})+(K^\star_{\mathcal D,\rho}-K_0),
\]

报告有限样本估计误差与模型/区域定义偏差。主图为 `region scale × quartic strength -> projection bias`，并按采样器分层。

## 6. 第三步 C：有限组合压力

只运行代表性组合，不做全笛卡尔积：

- 低噪声 + 弱非二次；
- 低噪声 + 强非二次；
- 高噪声 + 弱非二次；
- trajectory + 非驻点 + 中等区域尺度。

使用 A 中冻结的 gate，不重新调参，只检查风险机制是否叠加。

## 7. 输出与覆盖保护

结果写入独立 `results_step3/`，不得与第二步结果拼接。正式输出包括开发原始表、冻结阈值、保留原始表、风险—覆盖率表、非二次原始表、组合压力表、图、manifest 和 `THIRD_STEP_RESULT.md`。

所有脚本默认拒绝覆盖；只有显式 `--overwrite` 可重建同一结果身份。manifest 记录生成代码提交、配置网格、种子集合、依赖版本、行数和结果哈希。失败运行不得混入正式表。

## 8. Gate 3 判定

- \(n\le8\)、低噪声下高覆盖且低风险：`Early-stage candidate`；
- 仅 \(n\ge12\) 稳定：`Mid-stage candidate`；
- 满秩但频繁非正定或过度自信：`Not safely identifiable`；
- 对采样器或区域半径高度敏感：`Sampler/scale-dependent descriptor`；
- 谱稳定而量级不稳定：只保留谱；
- 所有 gate 在 \(n\le8\) 无法兼顾风险与覆盖：禁止早期迁移；
- 非二次下收敛到采样器相关投影：属性存储必须携带区域尺度和采样定义。
