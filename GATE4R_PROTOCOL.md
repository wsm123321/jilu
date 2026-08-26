# Gate 4-R 封闭修订协议

## 1. 身份与范围

`results_step4/` 永久保留为 Gate 4 原证据版本，不删除、不改写、不与修订结果拼接。Gate 4-R 使用独立 `results_step4r/`，完整重跑 development → freeze → holdout → unseen。Gate 5 在本修订完成前不启动。

## 2. 修订项

1. 所有函数族通过独立 `center_hessian()` 定义原点真实 Hessian，并由函数值有限差分验证；
2. Two-scale 正式 gate 同时要求 outer 与 inner 的 relative SE、signed-spectrum width 和 SPD probability 通过；
3. 每条记录保存：
   - `outer_projection_to_center`；
   - `inner_projection_to_center`；
   - `outer_inner_projection_drift`；
4. 中心可靠事件相对于修正后的真实中心 Hessian计算；
5. 增加 `Oracle-Scale`，使用真实 outer/inner-to-center 偏差，在相同总预算下给出检测上界；
6. trajectory 外层真值明确为该 seed 的 20 点 `path-specific empirical projection`，不外推为采样机制总体投影；
7. 正式字段分别保存 outer/inner 的联合不确定性，不再以内层代替双尺度门控；
8. Git 文本统一 LF，gzip/part 为 binary；CI 重构分片并验证 manifest。

## 3. 真值与标签

修正中心 Hessian为 \(K_0\)。定义：

\[
B_o=\|K_o^\star-K_0\|_F/\|K_0\|_F,
\quad
B_i=\|K_i^\star-K_0\|_F/\|K_0\|_F,
\]

\[
D^\star_{o,i}=\|K_o^\star-K_i^\star\|_F/[0.5(\|K_o^\star\|_F+\|K_i^\star\|_F)+10^{-12}].
\]

`center_semantically_valid = (B_o<=0.2 and B_i<=0.2)`；`scale_dependent_truth = (B_o>0.2 or B_i>0.2)`。外内漂移单独报告，不再替代 center truth。

## 4. 正式 gate

对 outer 和 inner 分别要求：

- relative magnitude SE ≤ \(\tau_{SE}\)；
- signed-spectrum interval width ≤ \(\tau_{spec}\)；
- SPD probability ≥ \(\tau_{SPD}\)。

然后要求估计尺度漂移 \(D\le\tau_D\)。只有全部通过才输出 `Two-scale-stable-SPD`。两尺度统计可用但漂移失败输出 `Scale-dependent`；其余 `Unidentifiable`。

阈值网格、seed cluster bootstrap、开发/保留 seed、函数族、预算和采样协议保持 Gate 4 不变，避免修订时扩展研究自由度。

## 5. Oracle-Scale

`Oracle-Scale` 使用真实 \(B_o,B_i,D^\star_{o,i}\) 判断中心语义是否有效，但仍使用相同的外层和 8 个内层计费预算。它不进入阈值冻结，仅回答多尺度真值信息本身的上界。

## 6. 确认标准

所有 Gate 4 数字作废为最终确认结果，但保留为审计历史。Gate 4-R 重新报告开发阈值、保留风险 UCB、未见函数族风险、尺度相关检出、函数族分层、等预算比较和 Oracle 上界。无论新数字是否改善，当前中心 Hessian 路线的停止决策不因修订而反向调参。
