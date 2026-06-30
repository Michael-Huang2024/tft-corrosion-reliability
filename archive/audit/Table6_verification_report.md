# Table 6 核查报告

**核查日期：** 2026-06-29  
**关联 Figure：** `outputs/figures/Fig3_pf_by_cover_depth_600dpi.png`  
**核查目的：** 使用与 Figure 3 相同的数据源，验证手稿 Table 6 中 broad cover-depth 分组在最终评估年的 population-level 腐蚀起始概率是否正确。

---

## 1. 执行摘要

| 项目 | 结论 |
|---|---|
| 当前 Table 6 是否正确 | **否 — 数值已过时，与 Figure 3 不一致** |
| 是否需重新生成 Figure 3 | **否** |
| 是否需更新 Table 6 | **是** |
| 最终评估时刻 | `t_year = 59.947981`（约 59.95 年） |
| 评估集规模 | 150 个 held-out test series |

**结论：** 手稿中现有的 Table 6 数值（0.612 / 0.598 / 0.014 等）无法由 Figure 3 所用数据源复现。应替换为基于同一数据源重新计算的更正值。

---

## 2. 核查范围与方法

### 2.1 核查任务

1. 使用与 Figure 3 相同的预测数据与 test set。
2. 计算以下 broad cover-depth 分组在**最终评估年**的 population-level 腐蚀起始概率：
   - 40–60 mm
   - 60–80 mm
   - 80–110 mm
3. 对每组报告：test series 数量、reference 概率、predicted 概率、绝对误差。
4. 与当前手稿 Table 6 数值对比。
5. 若不一致，提供更正后的 Table 6。

### 2.2 数据源（与 Figure 3 完全一致）

| 角色 | 文件路径 | 使用列 |
|---|---|---|
| TFT 点预测 + 模拟器标签 | `outputs/revision/predictions/tft_20250111_10epoch_points.csv` | `series_id`, `time_idx`, `t_year`, `p_onset_pred`, `onset_flag` |
| 保护层深度映射 | `data/processed/revision/final_onset_summary.csv` | `series_id`, `cover_mm` |
| 数据集划分（验证 test set） | `data/processed/revision/series_split.csv` | `series_id`, `split` |

**TFT 模型：** seed **20250111**（revision 中代表种子，与 Table 5、MC Dropout UQ 及 Figure 3 一致）。

**Figure 3 生成脚本：** `scripts/generate_fig3_revision.py`

### 2.3 聚合方法

与 `scripts/generate_fig3_revision.py` / `scripts/05_make_figures.py` 中 Figure 3 的逻辑一致：

1. 将点预测表与 `cover_mm` 按 `series_id` 合并。
2. 按 broad cover 区间筛选 test series。
3. 取最终评估时刻 `t_year = max(t_year) = 59.947981`。
4. 在该时刻对组内所有 series 求均值：
   - **Reference（模拟器）：** `mean(onset_flag)`
   - **Predicted（TFT）：** `mean(p_onset_pred)`
   - **Absolute error：** `|Predicted − Reference|`

### 2.4 分组区间定义

Broad 分组采用半开区间 `[lo, hi)`（单位：mm）：

| 分组标签 | 区间 | 对应 Figure 3 的 10 mm 子区间 |
|---|---|---|
| 40–60 mm | [40, 60) | 40–50 mm + 50–60 mm |
| 60–80 mm | [60, 80) | 60–70 mm + 70–80 mm |
| 80–110 mm | [80, 110) | 80–90 mm + 90–100 mm + 100–110 mm |

---

## 3. 当前手稿 Table 6 数值

以下为待核查的现有数值：

| Cover group | Reference | Predicted | Absolute error |
|---|---:|---:|---:|
| 40–60 mm | 0.612 | 0.598 | 0.014 |
| 60–80 mm | 0.448 | 0.437 | 0.011 |
| 80–110 mm | 0.304 | 0.293 | 0.011 |

---

## 4. 重新计算结果

### 4.1 最终评估年 broad 分组结果（Figure 3 同源数据）

| Cover group | Test series (N) | Reference P_f(t) | Predicted P_f(t) | Absolute error |
|---|---:|---:|---:|---:|
| 40–60 mm | 59 | **0.847** | **0.843** | **0.004** |
| 60–80 mm | 40 | **0.200** | **0.201** | **0.001** |
| 80–110 mm | 51 | **0.000** | **0.000** | **0.000** |

**Series 总数：** 59 + 40 + 51 = **150**（与 test set 一致）

### 4.2 高精度数值（内部核对用）

| Cover group | Reference (exact) | Predicted (exact) | Error (exact) |
|---|---:|---:|---:|
| 40–60 mm | 0.847458 | 0.843329 | 0.004128 |
| 60–80 mm | 0.200000 | 0.201032 | 0.001032 |
| 80–110 mm | 0.000000 | 0.000006 | 0.000006 |

> **注：** 80–110 mm 组的 predicted 精确值为 6.03×10⁻⁶，四舍五入至三位小数为 0.000。

### 4.3 最终时刻各 10 mm 子区间明细（与 Figure 3 曲线终点一致）

| Cover bin | Test series (N) | Reference | Predicted | Error |
|---|---:|---:|---:|---:|
| 40–50 mm | 24 | 0.917 | 0.917 | 0.000 |
| 50–60 mm | 35 | 0.800 | 0.793 | 0.007 |
| 60–70 mm | 21 | 0.190 | 0.195 | 0.005 |
| 70–80 mm | 19 | 0.211 | 0.208 | 0.003 |
| 80–90 mm | 23 | 0.000 | 0.000 | 0.000 |
| 90–100 mm | 17 | 0.000 | 0.000 | 0.000 |
| 100–110 mm | 11 | 0.000 | 0.000 | 0.000 |

Broad 分组数值为组内 series 在最终时刻的 unweighted population mean，与上述子区间按 series 数量加权合并结果一致（例如 40–60 mm：`(24×0.917 + 35×0.800) / 59 = 0.847`）。

---

## 5. 对比分析：当前 Table 6 vs 重新计算值

### 5.1 逐项对比

| Cover group | 指标 | 当前 Table 6 | 重新计算 | 差异 | 是否一致 |
|---|---|---:|---:|---:|---|
| 40–60 mm | Reference | 0.612 | 0.847 | +0.235 | **否** |
| 40–60 mm | Predicted | 0.598 | 0.843 | +0.245 | **否** |
| 40–60 mm | Error | 0.014 | 0.004 | −0.010 | **否** |
| 60–80 mm | Reference | 0.448 | 0.200 | −0.248 | **否** |
| 60–80 mm | Predicted | 0.437 | 0.201 | −0.236 | **否** |
| 60–80 mm | Error | 0.011 | 0.001 | −0.010 | **否** |
| 80–110 mm | Reference | 0.304 | 0.000 | −0.304 | **否** |
| 80–110 mm | Predicted | 0.293 | 0.000 | −0.293 | **否** |
| 80–110 mm | Error | 0.011 | 0.000 | −0.011 | **否** |

**所有九项数值均不匹配。**

### 5.2 当前数值的可能来源分析

为排查现有 Table 6 的来源，对同一数据源在不同评估年份进行了扫描：

| 目标年份 | 40–60 mm (ref / pred) | 60–80 mm (ref / pred) | 80–110 mm (ref / pred) |
|---|---|---|---|
| ~20 yr | 0.169 / 0.156 | 0.000 / 0.000 | 0.000 / 0.000 |
| ~38 yr | **0.610 / 0.602** | 0.050 / 0.049 | 0.000 / 0.000 |
| ~40 yr | 0.678 / 0.644 | 0.050 / 0.069 | 0.000 / 0.000 |
| ~60 yr（最终） | **0.847 / 0.843** | **0.200 / 0.201** | **0.000 / 0.000** |

**发现：**

- 当前 Table 6 中 40–60 mm 的 reference **0.612** 与约 **第 38 年** 的 0.610 接近，但同一年份下 60–80 mm 仅为 0.050（非 0.448），80–110 mm 为 0.000（非 0.304）。
- **不存在单一评估年份** 能同时复现三组的现有数值。
- 现有 Table 6 很可能来自：**旧版数据**、**不同评估集**（非当前 150 test series）、**不同模型/checkpoint**，或**混合/手误录入**的数值。

### 5.3 与参数候选筛选表的区别

`outputs/revision/tables/parameter_candidate_cover_groups.csv` 中 Candidate C 在 60 年的参考值（全 1000 series，非 test set）为：

| Cover group | Series count | Pf (reference only) |
|---|---:|---:|
| 40–60 mm | 280 | 0.814 |
| 60–80 mm | 276 | 0.275 |
| 80–110 mm | 444 | 0.014 |

这些数值亦与当前 Table 6 及 Figure 3 test-set 结果均不一致，进一步说明现有 Table 6 并非来自当前 revision 锁定配置下的 test-set TFT 评估。

---

## 6. 与 Figure 3 的一致性验证

| 检查项 | 结果 |
|---|---|
| 同一 TFT 点预测文件 | ✓ `tft_20250111_10epoch_points.csv` |
| 同一 cover 映射文件 | ✓ `final_onset_summary.csv` |
| 同一 test set（150 series） | ✓ |
| 同一最终评估时刻 | ✓ `t_year = 59.947981` |
| 物理趋势：P_f 随 cover 增大而降低 | ✓ |
| Table 6 与 Figure 3 曲线终点一致 | ✓ |

**Figure 3 曲线终点读数交叉验证：**

- 40–50 mm、50–60 mm 在 ~60 yr 处分别约为 0.92 和 0.80 → 合并 40–60 mm 得 **0.847** ✓
- 60–70 mm、70–80 mm 在 ~60 yr 处分别约为 0.19 和 0.21 → 合并 60–80 mm 得 **0.200** ✓
- 80–110 mm 各子区间在 ~60 yr 处均为 0 → 合并得 **0.000** ✓

---

## 7. 更正后的 Table 6（建议替换稿）

### 7.1 手稿用表（三位小数）

**Table 6. Final-year population-level corrosion initiation probability by broad cover-depth group on the held-out test set (TFT seed 20250111).**

| Cover group (mm) | Test series (N) | Reference P_f(t) | TFT-predicted P_f(t) | Absolute error |
|---|---:|---:|---:|---:|
| 40–60 | 59 | 0.847 | 0.843 | 0.004 |
| 60–80 | 40 | 0.200 | 0.201 | 0.001 |
| 80–110 | 51 | 0.000 | 0.000 | 0.000 |

### 7.2 建议的 Table 6 说明文字（可选）

> Population-level probabilities are computed as the test-set mean of scenario-level initiation flags (reference) and TFT-predicted probabilities (predicted) at the final evaluation time (`t ≈ 59.95 years`). Cover-depth groups use half-open intervals [40, 60), [60, 80), and [80, 110) mm. Results are based on the same data and aggregation logic as Figure 3.

---

## 8. 建议操作

| 优先级 | 操作 |
|---|---|
| 1 | **更新手稿 Table 6** 为第 7 节中的更正数值 |
| 2 | **保留 Figure 3**（`outputs/figures/Fig3_pf_by_cover_depth_600dpi.png`），无需重新生成 |
| 3 | 确认 Table 6 caption/脚注中注明：final evaluation year、test set、TFT seed 20250111 |
| 4 | 勿使用 `parameter_candidate_cover_groups.csv` 或旧版 population 统计作为 Table 6 来源 |

---

## 9. 复现命令

以下 Python 代码可在项目根目录复现本报告中的 Table 6 数值：

```python
import pandas as pd
from pathlib import Path

ROOT = Path(".")
points = pd.read_csv(ROOT / "outputs/revision/predictions/tft_20250111_10epoch_points.csv")
cover = pd.read_csv(ROOT / "data/processed/revision/final_onset_summary.csv")[
    ["series_id", "cover_mm"]
].drop_duplicates()

df = points.merge(cover, on="series_id")
t_final = df["t_year"].max()
final = df[df["t_year"] == t_final]

groups = [("40-60 mm", 40, 60), ("60-80 mm", 60, 80), ("80-110 mm", 80, 110)]
for label, lo, hi in groups:
    g = final[(final["cover_mm"] >= lo) & (final["cover_mm"] < hi)]
    ref = g["onset_flag"].mean()
    pred = g["p_onset_pred"].mean()
    print(f"{label}: N={g['series_id'].nunique()}, ref={ref:.6f}, pred={pred:.6f}, err={abs(pred-ref):.6f}")
```

预期输出：

```
40-60 mm: N=59, ref=0.847458, pred=0.843329, err=0.004128
60-80 mm: N=40, ref=0.200000, pred=0.201032, err=0.001032
80-110 mm: N=51, ref=0.000000, pred=0.000006, err=0.000006
```

---

## 10. 相关文件索引

| 文件 | 说明 |
|---|---|
| `outputs/figures/Fig3_pf_by_cover_depth.png` | Figure 3（300 DPI） |
| `outputs/figures/Fig3_pf_by_cover_depth_600dpi.png` | Figure 3（600 DPI） |
| `outputs/figures/Fig3_pf_by_cover_depth.pdf` | Figure 3（矢量） |
| `scripts/generate_fig3_revision.py` | Figure 3 生成脚本 |
| `outputs/revision/Figure3_verification_report.md` | Figure 3 核查报告 |
| `outputs/revision/predictions/tft_20250111_10epoch_points.csv` | TFT 点预测（Table 6 数据源） |
| `data/processed/revision/final_onset_summary.csv` | cover_mm 映射 |
| `outputs/revision/tables/parameter_candidate_cover_groups.csv` | **非** Table 6 来源（参数筛选诊断） |

---

*报告由 Table 6 核查任务自动生成。*
