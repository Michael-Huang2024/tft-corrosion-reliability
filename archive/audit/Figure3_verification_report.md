# Figure 3 图像核查报告

**核查日期：** 2026-06-29  
**Figure 目录：** `outputs/revision/figures`  
**候选文件：** `parameter_candidate_cover_groups.png`  
**手稿 Figure 3 说明：**

> Figure 3. Cover-depth-stratified comparison between simulator-derived and TFT-predicted corrosion initiation probability, P_f(t).

---

## 1. 执行摘要

| 项目 | 结论 |
|---|---|
| 候选文件是否为 Figure 3 | **否** |
| 推荐使用的文件 | `outputs/figures/Fig3_pf_by_cover_depth.png`（**当前不存在，需生成**） |
| 是否存在更高分辨率版本 | **否** |
| 说明文字是否需要修改 | **否**（应更换图像，而非修改 caption） |

**结论：** `parameter_candidate_cover_groups.png` 是 Phase 1 参数候选筛选的诊断图，不是手稿 Figure 3 所需的「模拟器 vs TFT」保护层深度分层对比图。请勿将该文件插入 Figure 3 说明之前。

---

## 2. 候选文件核查

### 2.1 文件信息

| 属性 | 值 |
|---|---|
| 路径 | `outputs/revision/figures/parameter_candidate_cover_groups.png` |
| 分辨率 | 2100 × 1200 px |
| DPI | 300 |
| 生成脚本 | `scripts/13_screen_parameter_candidates.py` |
| 关联文档 | `outputs/revision/phase_1_4_status.md` |

### 2.2 图像内容

- **标题：** Candidate cover-depth groups
- **Y 轴：** Reference cumulative P_f(t)
- **X 轴：** Year（仅在 20、40、60 年三个时间点有数据点）
- **图例：** 9 条曲线，按参数候选 A/B/C 与保护层分组（40–60 mm、60–80 mm、80–110 mm）组合
- **数据来源：** 物理模拟器参考值（参数候选筛选阶段），**不含 TFT 预测**

### 2.3 与 Figure 3 说明的对比

| 核查项 | 手稿 Figure 3 要求 | 候选文件实际情况 | 是否匹配 |
|---|---|---|---|
| 内容类型 | 模拟器 vs TFT 对比 | 仅模拟器参考曲线 | **否** |
| 分层维度 | 按保护层深度 | 按保护层深度 + 参数候选 A/B/C | **否** |
| 曲线类型 | 实线（真值）+ 虚线（TFT） | 仅一组参考曲线 | **否** |
| 时间范围 | 完整 P_f(t) 轨迹 | 仅 20/40/60 年三个点 | **否** |
| 用途 | 手稿 Figure 3 | Phase 1 参数筛选诊断 | **否** |

---

## 3. 正确的 Figure 3 规格

### 3.1 规范来源

正确 Figure 3 由 `scripts/05_make_figures.py` 中的 `make_fig3()` 生成：

```python
# scripts/05_make_figures.py — make_fig3()
curve = group.groupby("t_year", as_index=False).agg(
    Pf_true=("onset_flag", "mean"),
    Pf_pred=("p_onset1_pred", "mean"),
)
# 实线: Pf_true (simulator-derived)
# 虚线: Pf_pred (TFT-predicted)
```

### 3.2 预期输出

| 文件 | 路径 | 状态 |
|---|---|---|
| PNG | `outputs/figures/Fig3_pf_by_cover_depth.png` | **不存在** |
| PDF | `outputs/figures/Fig3_pf_by_cover_depth.pdf` | **不存在** |

### 3.3 图像规格

- **分层方式：** 10 mm 保护层深度区间（40–50、50–60、…、100–110 mm）
- **曲线：** 每个区间两条线——实线（simulator-derived / true）与虚线（TFT-predicted）
- **时间轴：** 完整评估时间范围内的连续 P_f(t) 曲线
- **Y 轴标签：** Corrosion initiation probability, P_f(t)
- **图标题（脚本内）：** Population-level P_f(t) stratified by concrete cover depth

### 3.4 生成前提

运行 `scripts/05_make_figures.py` 前需存在：

- `outputs/predictions/onset_flag_pred_point.parquet`
- `outputs/predictions/series_static.csv`（含 `cover_mm` 列）

当前 `outputs/predictions/` 目录为空，故 Figure 3 尚未生成。

---

## 4. 高分辨率版本核查

在 `outputs/revision/figures` 目录中：

- 与 cover / Fig3 相关的文件仅有 `parameter_candidate_cover_groups.png`
- **无** PDF、SVG、EPS 矢量版本
- **无** `_600dpi` 高分辨率变体（对比：`mc_dropout_uncertainty_band_revised_600dpi.png` 存在 600 DPI 版本）
- 全仓库检索 **无** `Fig3_pf_by_cover_depth.*` 文件

| 文件 | 尺寸 | DPI |
|---|---|---|
| `parameter_candidate_cover_groups.png` | 2100 × 1200 | 300 |
| `Fig3_pf_by_cover_depth.png` | — | **未生成** |

---

## 5. Caption 核查

**当前 caption 无需修改。**

| Caption 表述 | 对应图像元素 |
|---|---|
| simulator-derived | `onset_flag` 均值，实线，图例标注 `(true)` |
| TFT-predicted | `p_onset1_pred` 均值，虚线，图例标注 `(TFT)` |
| cover-depth-stratified | 按 10 mm 保护层深度区间分组 |
| P_f(t) | 腐蚀起始累积概率 |

可选微调（非必须）：若需与脚本标题完全一致，可在 caption 中加入 “population-level”，但现有 caption 已足够准确。

---

## 6. 建议操作

1. **不要** 将 `parameter_candidate_cover_groups.png` 插入 Figure 3 位置。
2. 完成 TFT 推理后，运行 `scripts/05_make_figures.py` 生成正确的 Figure 3。
3. 将生成的 `outputs/figures/Fig3_pf_by_cover_depth.png`（或 PDF）用于手稿。
4. 如需更高分辨率，可在生成后导出 600 DPI 版本，或直接使用 PDF 矢量格式。

---

## 7. 附录：revision/figures 目录中与 Figure 3 易混淆的文件

| 文件 | 实际用途 | 是否为 Figure 3 |
|---|---|---|
| `parameter_candidate_cover_groups.png` | 参数候选筛选：保护层分组参考 P_f(t) | **否** |
| `parameter_candidate_pf_curves.png` | 参数候选筛选：总体参考 P_f(t) | **否** |
| `final_population_trajectories_by_model.png` | 多模型（LR/MLP/GRU/TFT）总体轨迹对比 | **否** |

---

*报告由 Figure 3 图像核查任务自动生成。*
