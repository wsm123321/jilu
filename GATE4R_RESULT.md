# Gate 4-R 修订结果

## 1. 修订结论

Gate 4 原证据版本因中心 Hessian 标注、双尺度门控实现和真值语义偏差，不能作为最终确认结果；原 `results_step4/` 已永久保留。Gate 4-R 在独立 `results_step4r/` 中完整重跑。

修订后结论为：

> **在开发函数族的全新 seed 上，修订后的双尺度 gate 可以满足风险 UCB≤10%，但证书覆盖率仅 0.30%；在未见函数族上风险 UCB 为 18.70%，仍不安全。中心 Hessian few-shot 迁移路线继续停止。**

同时，Oracle-Scale 上界在保留/未见函数族上分别有 55.18%/40.47% 的真实语义覆盖，说明尺度语义信息本身并非不存在；失败主要来自当前统计检测器的极低效率和失配泛化不足。

## 2. 修正内容

1. 新增 `center_hessian()`：
   - quadratic、axis/cross/rotated quartic 的附加中心 Hessian 为 0；
   - cubic 增加 \(1.38\gamma\operatorname{diag}(1,0)\)；
   - oscillatory 增加 \(\omega^2\cos\phi\,\gamma\operatorname{diag}(1,0)\)。
2. 所有家族由函数值中心有限差分独立验证解析 Hessian。
3. CSV 分别保存 outer/inner relative SE、spectrum width、SPD probability；正式 gate 要求两尺度同时通过。
4. 真值拆为 outer-to-center、inner-to-center 和 outer-inner drift。
5. `center_semantically_valid` 要求 outer 和 inner 投影都距修正中心 Hessian不超过 0.2。
6. 增加 Oracle-Scale 上界。
7. trajectory 外层真值明确记录为 `path-specific empirical projection`。
8. 新增 `.gitattributes`，文本冻结 LF，gzip/parts/PNG 冻结 binary；CI 重构证据并验证 manifest。

## 3. 修订开发阈值

修订后的正式 gate 比原 Gate 4 更严格，因为 outer 和 inner 必须同时通过。冻结阈值为：

| 参数 | Gate 4-R |
|---|---:|
| outer/inner relative SE | ≤0.10 |
| outer/inner spectrum width | ≤0.50 |
| outer/inner P(SPD) | ≥0.80 |
| estimated drift | ≤0.05 |
| 开发覆盖率 | 0.262% |
| 开发风险 | 3.03% |
| seed-cluster bootstrap UCB95 | 7.55% |
| 选中 seed cluster | 25/50 |

Gate 4 原版本开发覆盖率为 2.40%；双尺度正确实现后下降到 0.262%，说明原实现只检查 inner uncertainty 确实高估了准入能力。

## 4. 修订保留结果

### 4.1 同开发家族、100 个全新 seed

Structured-Two-Scale：

- 证书覆盖率：0.302%；
- 接受后风险：5.26%；
- seed-cluster-bootstrap 风险 UCB95：8.73%；
- Scale-dependent 检出率：16.47%；
- Unidentifiable：83.93%。

安全上界通过 10% 标准，但覆盖率远低于 40% 实用目标，尺度相关检出率也远低于 80%。这属于“确认安全但几乎完全弃权”，不具备昂贵优化实用资格。

### 4.2 未见函数族、100 个独立 seed

Structured-Two-Scale：

- 覆盖率：0.260%；
- 接受后风险：11.45%；
- 风险 UCB95：18.70%；
- Scale-dependent 检出率：10.28%；
- Unidentifiable：87.75%。

未见失配类型下主安全标准失败。

## 5. 修订后的函数族结论

| 函数族 | 覆盖率 | 接受后风险 | 风险 UCB95 | Scale-dependent 检出率 |
|---|---:|---:|---:|---:|
| Exact quadratic | 0.500% | 0% | 0% | 无 scale-dependent truth |
| Axis quartic | 0.231% | 8.00% | 16.22% | 18.24% |
| Cross quartic | 0.306% | 6.06% | 11.29% | 13.22% |
| Cubic（修正中心 Hessian） | 0.507% | 0% | 0% | 5.72% |
| Rotated quartic | 0.264% | 24.56% | 39.62% | 15.95% |
| Oscillatory（修正中心 Hessian） | 0.0069% | 100% | 100% | 5.29% |

修正中心 Hessian 后：

- cubic 不再表现出原报告中的虚假中心错误，但证书覆盖仅约 0.51%；
- oscillatory 的真实中心 Hessian 已修正，它仍是强失败场景：唯一极少数接受记录错误；
- 未见总体风险数字已经重算，原 28.51%/33.77% 不再作为最终数字。

## 6. 三种真值的区别

每条修订记录分别保存：

\[
B_o=d(K_o^\star,K_0),\quad B_i=d(K_i^\star,K_0),\quad D^\star_{o,i}=d(K_o^\star,K_i^\star).
\]

开发函数族中：

- axis quartic 的 outer-to-center 中位偏差约 0.343，inner 约 0.057；
- cross quartic 的 outer-to-center 约 0.133，inner 约 0.0009。

这说明“两个尺度相互接近”与“两个尺度都接近中心 Hessian”不是同一标签。Gate 4-R 的 center truth 使用前两项，不再用 outer-inner drift 替代。

## 7. Oracle-Scale 上界

| Split | Oracle 真实语义覆盖率 | 风险 | Scale-dependent 检出率 |
|---|---:|---:|---:|
| Holdout | 55.18% | 0% | 100% |
| Unseen | 40.47% | 0% | 100% |

Oracle 上界说明：

- 多尺度语义标签本身有非平凡覆盖；
- 当前 Structured gate 的 0.3% 覆盖并非因为所有场景都语义不合格；
- 主要瓶颈是 8 点 inner 的统计置信、双尺度共同通过概率和未见失配检测能力。

## 8. 等预算比较

| 方法 | 平均预算 | 修订证书覆盖率（holdout） | 说明 |
|---|---:|---:|---|
| Residual-Only | 10 | 不提供 | 无尺度证据 |
| Outer-More | 18 | 不提供 | 提高单尺度估计但无语义证书 |
| Random-Inner | 18 | 0.133% | 风险 UCB 15.28%，失败 |
| Structured-Two-Scale | 18 | 0.302% | 风险 UCB 8.73%，但覆盖不可用 |
| Oracle-Scale | 18 | 55.18% | 真实尺度标签上界，不是在线方法 |

Structured 优于 Random-Inner，但与 Oracle 上界相差约两个数量级。

## 9. 工程修订

- 新增 `.gitattributes`，文本统一 `eol=lf`；
- Step 3/4 历史 bundle manifest 已按 LF 规范重新计算；
- CI 现在会重构 Gate 3/4 gzip 并验证所有分片和整体 SHA-256；
- Gate 4-R 使用独立结果目录、独立 manifests 和独立压缩分片，不覆盖旧证据。

## 10. 最终决策

Gate 4-R 修订没有改变停止决策，但使证据口径准确：

1. 已见函数族上可达到风险控制，但覆盖仅 0.3%；
2. 未见函数族上风险控制失败；
3. 尺度相关检出率仅 10%—16%；
4. Oracle 证明语义信息存在，但当前 few-shot 检测器无法有效利用；
5. 不进入 Operational \(\widehat B,\widehat s\) 或中心 Hessian 迁移。

Gate 5 可以在 Gate 4-R 归档后另立新协议，研究带 \(\rho,\mu\) 的区域投影是否解释局部决策相似性，但不得把 Gate 4 原数字继续作为最终确认结果。
