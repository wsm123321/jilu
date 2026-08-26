# 第四步结果：两尺度语义证书未通过

> **审计状态：已被 Gate 4-R 取代。** 本文件永久保留为原证据版本，但其中未见函数族中心 Hessian、正式双尺度门控和真值标签存在已确认偏差，不能作为论文最终数字。最终修订结论见 `GATE4R_RESULT.md` 与 `results_step4r/`。

## 1. 总结

Gate 4 已完成开发阈值冻结、全新 seed 保留验证、未见函数族验证、真实失配观测驱动 trajectory 和严格计费等预算对照。

最终结论是：

> **两尺度联合门控没有通过主安全标准，也没有达到最低实用覆盖率。当前中心 Hessian 的 few-shot 条件迁移路线应停止，不进入 Operational \(\widehat B,\widehat s\) 或源—目标迁移。**

开发集上 gate 可以把 seed-cluster-bootstrap 风险 UCB 压到 10% 以下，但覆盖率只有 2.40%；在全新 seed 上风险 UCB 升至 11.62%，未见函数族上升至 33.77%。尺度相关检出率约 18%，远低于 80% 目标。

## 2. 协议修正

### 2.1 Gate 3 偏差补记

Gate 3 的 uncertainty gate 已改称 Magnitude-SE Gate。第三步未实现完整 bootstrap，只用量级 delta-method SE 代理量级、谱和 SPD 的联合可靠事件；风险区间按记录而非 seed 聚类；非二次 trajectory 也未完全由实际失配观测驱动。这些限制已写入 `THIRD_STEP_RESULT.md`。

### 2.2 单壳对称探针不可辨识

原提议的 8 点全在同一半径圆上，使截距与 \(u_1^2+u_2^2\) 共线，完整二次模型秩亏。正式实现改为两壳设计：

- 外壳：\(\pm\rho e_1,\pm\rho e_2\)；
- 内壳：四个对角方向，半径 \(\rho/2\)。

仍严格使用 8 次目标评估，并通过完整秩测试。

### 2.3 正类口径收紧

两个尺度最多支持“在预设半径与探针设计上稳定的 SPD 曲率”，不能证明一般意义的本征性。正式标签为 `Two-scale-stable-SPD`、`Scale-dependent` 和 `Unidentifiable`。后者是弃权，不是可以免费计为正确的真实类别。

## 3. 实验身份

- 开发函数族：quadratic、axis quartic、cross quartic；
- 未见函数族：rotated quartic、asymmetric cubic、oscillatory；
- 外层样本：8 或 12；
- 内层探针：8；
- 总预算：16 或 20；
- 内层外半径：0.25 或 0.5；
- 采样器：Sobol prefix、Uniform、真实失配观测驱动 trajectory；
- 相对噪声：0.01、0.05；
- 开发 seed：50；
- 保留 seed：100；
- 未见家族 seed：100；
- 每个 Hessian 联合不确定性：200 次参数化抽样；
- 风险阈值：2000 次 seed-cluster bootstrap UCB。

开发阶段只用 Structured-Two-Scale 冻结 gate；等预算方法只在保留阶段比较，不能影响阈值。

## 4. 冻结 gate

开发集在 228 个具备足够 seed cluster 的候选中，只有 12 个满足风险 UCB≤0.10。最大覆盖候选为：

| 参数 | 冻结值 |
|---|---:|
| relative magnitude SE | 0.10 |
| signed-spectrum interval width | 0.50 |
| P(SPD) | 0.80 |
| normalized two-scale drift D | 0.10 |
| 开发覆盖率 | 2.40% |
| 开发风险 | 6.62% |
| 开发风险 UCB95 | 8.89% |
| 被选 seed cluster | 49/50 |

虽然开发风险满足约束，但覆盖率已经远低于 40% 实用性目标，属于“安全但几乎不可用”的候选。

## 5. 保留集主结果

### 5.1 全新 seed，同开发函数族

Structured-Two-Scale：

- 语义证书覆盖率：2.01%；
- 接受后风险：9.50%；
- seed-cluster-bootstrap 风险 UCB95：11.62%；
- Scale-dependent 检出率：18.40%；
- Unidentifiable：80.13%。

风险点估计接近 10%，但 UCB 超过主安全线，因此未通过确认性标准。更重要的是覆盖率和尺度检出率远未达到实用门槛。

### 5.2 未见函数族

Structured-Two-Scale：

- 覆盖率：1.38%；
- 接受后风险：28.51%；
- 风险 UCB95：33.77%；
- Scale-dependent 检出率：18.33%；
- Unidentifiable：81.09%。

未见模型失配类型下门控明显失效，不能声称泛化。

## 6. 函数族边界

| 函数族 | Two-scale stable 覆盖率 | 接受后风险 | Scale-dependent 检出率 |
|---|---:|---:|---:|
| Exact quadratic | 3.65% | 0% | 无 scale-dependent truth |
| Axis quartic | 1.48% | 18.44% | 19.13% |
| Cross quartic | 1.98% | 8.64% | 17.28% |
| Cubic（未见） | 2.22% | 20.31% | 15.66% |
| Rotated quartic（未见） | 1.71% | 34.05% | 21.14% |
| Oscillatory（未见） | 0.06% | 100% | 15.98% |

振荡函数构成两尺度 aliasing 反例：两个预设尺度可以表现得相容，但不代表中心曲率语义正确。即使 gate 极少接受，接受者仍全部错误。

## 7. 等预算对照

| 方法 | 平均计费评估 | 语义证书覆盖率 | 中心曲率可靠率 | 说明 |
|---|---:|---:|---:|---|
| Residual-Only | 10 | 不提供 | 41.40% | 无第二尺度证据 |
| Outer-More | 18 | 不提供 | 56.05% | 多 8 点提高拟合，但不能验证尺度语义 |
| Random-Inner | 18 | 0.79% | 13.50% | 随机内层条件较差 |
| Structured-Two-Scale | 18 | 2.01% | 19.92% | 语义证书略多，但远低于实用门槛 |

未见函数族下 Structured-Two-Scale 的中心曲率可靠率仅 14.49%。结构化探针优于随机内层的证书覆盖，但没有达到安全实用标准；Outer-More 提高一般估计质量，却不具备尺度语义证书，不能被当作相同类型 gate。

## 8. Gate 4 验收

| 冻结标准 | 结果 |
|---|---|
| 保留风险 UCB≤0.10 | 失败：0.116 |
| 未见函数族风险 UCB≤0.10 | 失败：0.338 |
| 弱失配/正确二次覆盖率≥40% | 失败：最高仅约3.7% |
| 强失配错误接受≤10% | 总体及多个家族失败 |
| Scale-dependent 检出率≥80% | 失败：约18% |
| Structured 明显优于 Outer-More 的安全覆盖 | 无法成立；Outer-More无语义证书，Structured覆盖过低 |

## 9. 科学决策

Gate 4 的结论不是继续调两个尺度或增加门控复杂度。现有证据表明：

1. 8 个内层点只有 2 个残差自由度，联合属性不确定性极不稳定；
2. 内层半径缩小时曲率信号按 \(\rho^2\) 衰减，噪声相对影响快速增加；
3. 两尺度一致性不能排除振荡 aliasing；
4. 失配家族变化会破坏开发 gate 的风险控制；
5. 即使安全门槛可在开发集满足，覆盖率也低到缺乏昂贵优化实用性。

因此：

> **停止中心 Hessian 的 few-shot 条件迁移路线。暂不进入 Operational \(\widehat B,\widehat s\)，因为 Oracle 表示层尚未获得安全实用资格。**

后续若继续局部知识研究，应转向低自由度、直接决策相关的证据，例如：

- 多尺度有限差分增长比；
- 局部排序/改进方向的直接序贯证据；
- 带明确 \(\rho,\mu\) 的区域投影，而非中心 Hessian；
- source-blind 局部化和区域覆盖对照。

## 10. 核心图

- `results_step4/figures/figure1_semantic_risk_coverage.png`：保留/未见语义证书风险—覆盖率；
- `results_step4/figures/figure2_family_generalization.png`：函数族覆盖率与尺度检出率；
- `results_step4/figures/figure3_cost_coverage.png`：严格计费评估数—安全证书覆盖率。
