# 第四步冻结协议：多尺度语义鉴别与偏差门控

## 1. 唯一研究问题

> 在 Oracle \(B,s\) 已知时，能否用严格计费的 8 个内层验证点区分中心本征曲率与尺度相关区域投影，并在全新 seed 和未见函数家族上控制错误接受中心曲率的风险？

本阶段不研究 GP、BO、源—目标迁移或 Operational \(\widehat B,\widehat s\)。

## 2. 三分类输出

- `Two-scale-stable-SPD`（字段兼容名可暂用 `Intrinsic-stable`）：联合统计置信通过、在预设两个尺度与探针设计上稳定、SPD 概率通过；该标签不声称两个尺度足以证明一般数学意义的本征性；
- `Scale-dependent`：外层和内层各自统计可用，但尺度漂移显著；
- `Unidentifiable`：任一尺度统计证据不足或类型不确定。

只有第一类获得中心曲率的条件资格。第二类只能保存为带 \(\rho,\mu\) 的区域投影。

## 3. 研究对象与真实标签

中心曲率为 \(K_0=\nabla^2g(0)\)。尺度 \(\rho\)、测度 \(\mu\) 下的区域投影为 \(K^\star_{\rho,\mu}\)。真实语义偏差

\[
B^\star_{\rho,\mu}=\|K^\star_{\rho,\mu}-K_0\|_F/\|K_0\|_F
\]

只用于标签和机理分析，不进入在线门控。冻结语义标签：\(B^\star\le0.2\) 为 intrinsic-compatible，\(B^\star>0.2\) 为 scale-dependent truth。

## 4. 严格等预算设计

外层历史：\(n_o\in\{8,12\}\)、\(\rho_o=1\)。内层尺度：\(\rho_i\in\{0.5,0.25\}\)。

原提议把 8 点全部放在同一半径圆上，但此时 \(u_1^2+u_2^2=\rho_i^2\) 与截距列线性相关，完整二次模型秩亏。为保持 8 次严格预算，冻结为两壳对称设计：

\[
\pm \rho_i e_1,\ \pm \rho_i e_2,
\quad
\pm \frac{\rho_i}{2}\frac{e_1+e_2}{\sqrt2},\
\pm \frac{\rho_i}{2}\frac{e_1-e_2}{\sqrt2}.
\]

它仍由四对反向点组成，但两个半径使截距与曲率 trace 可分离。这里 \(\rho_i\) 表示内层支持域的外半径，而不是所有点的共同半径。比较：

1. `Residual-Only`：只用 \(n_o\)，作为无额外预算参照；
2. `Outer-More`：在外层增加 8 点；
3. `Random-Inner`：增加 8 个随机内层点；
4. `Structured-Two-Scale`：增加 8 个冻结对称探针；
5. `Oracle-Scale`：相同总预算但使用真实 \(B^\star\)，只作上界。

除 Residual-Only 外总预算均为 \(n_o+8\in\{16,20\}\)。所有点均计为目标评估。

## 5. 多尺度统计

外层与内层独立拟合，禁止共享观测，因此

\[
\operatorname{Cov}(\widehat h_o-\widehat h_i)=\widehat\Sigma_o+\widehat\Sigma_i.
\]

归一化漂移：

\[
D=\frac{\|\widehat K_o-\widehat K_i\|_F}{0.5(\|\widehat K_o\|_F+\|\widehat K_i\|_F)+10^{-12}}.
\]

协方差标准化漂移：

\[
T=(\widehat h_o-\widehat h_i)^\top(\widehat\Sigma_o+\widehat\Sigma_i)^\dagger(\widehat h_o-\widehat h_i).
\]

`vech` 约定固定为 \((K_{11},K_{12},K_{22})\)。

## 6. 联合属性不确定性

对每个可用尺度，从高斯参数近似

\[
h^{(b)}\sim\mathcal N(\widehat h,\widehat\Sigma_h)
\]

进行 200 次冻结 RNG 参数化抽样，记录：

- 曲率量级相对区间宽度；
- 带符号归一化谱的分位区间宽度；
- \(P(K\succ0)\)；
- 最小特征值 5% 分位数。

n=8 的残差自由度仅 2，仍允许估计但必须反映其不稳定；任何尺度 df≤0 时 Residual 联合不确定性不可用并输出 `Unidentifiable`。开发阶段可同时保存 Oracle-noise 联合不确定性作机理上界，但 operational 分类只使用 residual 版本。

## 7. 景观与真正自适应 trajectory

开发家族：exact quadratic、axis quartic、cross quartic。保留未见家族：rotated quartic、asymmetric cubic、oscillatory。

所有景观使用 Gate 3 正交 \(K(q,r,\theta)\)，主量级 \(q=1\)，\(r\in\{1,4,16\}\)，\(\theta\in\{0,\pi/4\}\)，非驻点状态开启/关闭，\(\eta\in\{0.01,0.05\}\)。

Trajectory 的每一步 incumbent 和下一步位置必须由**实际带噪声、带当前失配项的观测值**决定，不能先在二次景观生成路径后重放。

## 8. 开发与保留身份

- 开发：50 seeds，函数族 quadratic/axis/cross；
- 保留：100 个全新 seeds，同一开发函数族，用于 seed 外推；
- 未见家族：100 个独立 seeds，rotated/cubic/oscillatory，用于模型失配类型外推。

同一 seed 内条件共享基础随机点、噪声和候选创新；统计 resampling 单位是完整 seed cluster。

## 9. 阈值冻结

候选网格：

- Residual relative magnitude SE：\(\{0.05,0.10,0.20,0.30\}\)；
- 谱形区间宽度：\(\{0.10,0.20,0.30,0.50\}\)；
- \(P(SPD)\)：\(\{0.80,0.90,0.95\}\)；
- \(D\)：\(\{0.05,0.10,0.20,0.30,0.50\}\)；
- 主门控只调 \(D\)；\(T\) 固定为诊断并报告其数值与协方差秩，不参与阈值搜索，避免同时调两个漂移族扩大研究者自由度。

每个候选 gate 用 2000 次 seed-level cluster bootstrap 估计 selective risk 的 95% 上界。选择规则：

\[
\max \text{coverage}\quad\text{s.t.}\quad UCB_{95\%}(\text{risk})\le0.10.
\]

若无候选满足则冻结“无安全阈值”。平局依次选择更低风险、更严格阈值和更低探针成本。保留集不得重调。

## 10. 主要指标

- `Intrinsic-stable` 接受覆盖率与接受后中心曲率失败风险；
- seed-cluster-bootstrap 95% 风险 UCB；
- 真实 \(B^\star>0.2\) 时 `Scale-dependent` 检出率；
- 真二次/弱失配条件的 intrinsic 覆盖率；
- 三分类混淆矩阵；
- 严格等预算下安全覆盖率—探针成本；
- Outer-More、Random-Inner 与 Structured-Two-Scale 的成对 seed 差异。

主验收：保留 seed 和未见函数家族的 accepted failure risk 95% UCB≤0.10；外层 n=12 加 8 点时弱失配覆盖率≥40%；强失配错误接受率≤10%；当 \(B^\star>0.2\) 时 Scale-dependent 检出率≥80%。

## 11. 输出纪律

结果单独写入 `results_step4/`，不得与 Gate 3 拼接。长表确定性 gzip、分片、SHA-256 校验；默认拒绝覆盖。正式结果必须由已提交代码生成，并记录开发阈值、seed 集、函数族、计费点数、行数和代码提交。
