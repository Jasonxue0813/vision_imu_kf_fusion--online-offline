# splg_fusion 视觉匹配与协方差估计机制说明

本文档主要说明 `splg_fusion` 工程中，是如何从底层的特征匹配结果中提取定位坐标，并科学地估算出这一定位结果的**置信度（观测协方差）**的。这套机制对于卡尔曼滤波融合至关重要，它决定了滤波器该给予当前帧多大的权重。

## 1. 视觉匹配与双几何校验 (Dual Geometry Check)

当无人机拍摄的实时图像（通常经过去畸变和基于航向的旋转对齐）和裁剪出的卫星子图送入 `sp_lg`（SuperPoint + LightGlue）匹配模块后，系统不仅要求解出相机位姿，还需评估几何一致性。

在底层匹配代码 (`src/vision/sp_lg.cpp`) 中，针对匹配好的 2D-2D 特征点对，系统会并行执行两种几何模型的求解：

1. **PnP 几何 (Perspective-n-Point)**：
   结合相机内参以及通过高程图 (DEM) 赋予卫星特征点的 3D 高度，使用 RANSAC 求解完整的 6-DoF 相机位姿（包含 XYZ 及俯仰横滚航向）。
2. **单应几何 (Homography)**：
   假设地表是绝对平坦的，使用 RANSAC 求解 2D 像素到 2D 像素的投影变换矩阵。

### 1.1 计算像素级残差协方差
有了这两种模型后，系统会分别把匹配内点重新投影回图像上，并计算出三类偏差：
- `sigma_h`：单应矩阵预测的像素坐标与实际特征点坐标之间的残差协方差。
- `sigma_p`：PnP 预测坐标与实际坐标之间的残差协方差。
- `sigma_d`：**PnP 预测与单应预测彼此之间的分歧协方差（Disagreement）**。

随后，这三者被加权组合成一个综合的像素级误差协方差矩阵 `sigma_px`：
```cpp
sigma_px = 0.25 * sigma_h + 0.25 * sigma_p + 0.50 * sigma_d + 1e-4 * I
```
*注：`sigma_d` 权重最高，因为如果 PnP 和单应模型的预测相互打架，通常意味着地形起伏剧烈或者匹配存在严重错位，置信度应大幅下降。*

### 1.2 投影到物理地面 (Jacobian Propagation)
有了像素误差 `sigma_px` 后，系统会通过求解相机成像模型到地面坐标系的**局部雅可比矩阵 (Jacobian)**，将像素平面的误差传播到真实的物理地面（米）。
这会产生三个基础几何方差：
- `dual_geom_var_x_m2` (东西方向基础方差)
- `dual_geom_var_y_m2` (南北方向基础方差)
- `dual_geom_cov_xy_m2` (协方差)

## 2. 质量特征与经验趋势修正

拿到上述的物理基础方差后，系统并未直接将其喂给卡尔曼滤波器。因为在实际工程中，基础几何方差往往偏于乐观。

在融合接收端 (`include/splg_fusion/fusion/fusion_types.h` 的 `computeVisualMeasurementCovariance` 中)，系统提取了底层匹配输出的一系列**质量特征**：
- `localization_reproj_error` (重投影误差)
- `sp_lg_pnp_inlier_rate` (PnP 内点率)
- `sp_lg_point_spread_img_ratio` (特征点在图像上的分布散度，越分散越好)
- `sp_lg_obliqueness_deg` (视角倾斜度)
- `sp_lg_terrain_relief_m` (地形起伏程度)

### 2.1 XY 平面方差的倍率惩罚
系统内部硬编码了一套预先在特定数据集上拟合好的多元经验模型。它会将上述质量特征转换成一个**惩罚倍率 (Multiplier)**。
- 如果内点率极高、特征点均匀铺满画面、重投影误差极小，倍率接近 1.0。
- 如果特征点扎堆在角落、内点率低，倍率可能被放大到 4.0。
最终的 XY 方差 = `dual_geom_var_x_m2` × `惩罚倍率`。

### 2.2 Z 轴 (高度) 的对数模型预测
由于单目相机对深度的可观性天然极差，系统放弃了从几何推导高度方差，而是直接用一个对数回归模型，利用重投影误差、内点数量、地形起伏等特征，强行估算出一个 `var_z_m2`。

### 2.3 保底与门控 (Gating)
最后，为了防止滤波器过于信任某一次“看起来完美”的视觉观测，或者由于预测模型失效给出负数，系统会强制应用配置项中的最小下限：
- `kf_visual_min_xy_sigma_m` (默认 3.0 米)
- `kf_visual_min_z_sigma_m` (默认 4.0 米)

至此，最终的 3x3 视觉观测协方差矩阵 `R_visual` 构造完成，交由连续滤波器 (Continuous Filter) 计算卡尔曼增益。

## 3. CSV 数据输出清单

当视觉匹配完成后，`splg_fusion` 会将详细的匹配统计与不确定度信息输出至 **`sp_lg_result.csv`** 文件中。该文件主要包含以下几类核心数据，非常适合用于后期的数据分析和匹配器性能评估：

1. **特征点与匹配数量**
   - `sp_lg_map_feature_points`: 卫星地图底图提取出的 SuperPoint 特征点数量。
   - `sp_lg_aerial_feature_points`: 无人机航拍图提取出的特征点数量。
   - `sp_lg_lightglue_match_pairs`: LightGlue 初步匹配成功的特征点对数。

2. **几何内点与误差**
   - `sp_lg_homography_inlier_count` / `sp_lg_homography_inlier_rate`: 单应矩阵 RANSAC 后的内点数与内点率。
   - `localization_inlier_points` / `sp_lg_pnp_inlier_rate`: PnP RANSAC 后的内点数与内点率。
   - `localization_reproj_error`: PnP 最终的重投影误差 (像素)。

3. **双几何残差与分布特征**
   - `sp_lg_dual_geom_h_rmse_px` / `sp_lg_dual_geom_pnp_rmse_px` / `sp_lg_dual_geom_disagree_rmse_px`: 双几何在图像上的残差均方根。
   - `sp_lg_point_spread_img_ratio`: 特征点在图像上的覆盖散度。
   - `sp_lg_depth_spread_m` / `sp_lg_terrain_relief_m`: 深度跨度与地形起伏度。

4. **物理层面不确定度 (米)**
   - `sp_lg_dual_geom_var_x_m2` / `sp_lg_dual_geom_var_y_m2`: 雅可比传播到地面的基础物理方差。
   - `pred_var_diag_m2`: 结合经验模型修正后的最终预测方差。

*注：如果脱离融合框架，单独运行 `run_sp_lg_match.py` 脚本，输出的 CSV 文件通常命名为 `sp_lg_match_map2d.csv`，且会在终端打印包含 `homography_reproj_rmse_ground_m` 等指标的 summary。*

---

## 4. 匹配器替换指南 (如换用 SuperGlue 等)

如果你打算将目前的 `SuperPoint + LightGlue` 替换为其他匹配网络（例如 `SuperGlue`），请特别注意以下几点：

### 3.1 接口与统计信息的对齐
新匹配器不能只输出“我在哪（经纬高）”，它**必须**能够像目前的 `sp_lg` 一样，并行执行 PnP 和单应验证，并输出：
- PnP 内点率、重投影误差
- 图像上的点散度 (Point spread)
- 双几何残差推导出的基础物理方差 (`dual_geom_var_x_m2`)
如果没有这些指标，上述严密的协方差推导机制将无数据可用。

### 3.2 经验回归模型的失效
`fusion_types.h` 中用于推算惩罚倍率和 Z 轴方差的多元模型公式（那些带有长串小数系数的线性/对数方程），是**针对当前特定的 sp+lg 组合和特定的测试数据集拟合出来的**。
不同的匹配网络，其内点分布、重投影误差的统计特性完全不同。直接套用旧系数，会导致算出的协方差要么过大（导致视觉被忽略），要么过小（导致轨迹拉飞）。

### 3.3 替换建议
- **短期/调试期**：
  建议在 `computeVisualMeasurementCovariance` 函数中，直接屏蔽掉 `computeVisualAxisSigmaPrediction` 相关的经验推算逻辑。直接退化为：使用一个固定的常数协方差（如 XY=3.0, Z=4.0），先验证新匹配网络的基础定位成功率和融合联通性。
- **长期/生产期**：
  需要使用新网络跑一批带有 RTK/GNSS 真值的数据集。记录下新网络输出的各类特征（内点率、散度等）以及实际的物理定位误差，然后利用 Python 的机器学习库（如 `statsmodels`）重新拟合出一套新的多元回归系数，替换回 `fusion_types.h` 中。