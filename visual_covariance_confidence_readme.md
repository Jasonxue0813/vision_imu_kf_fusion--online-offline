# 视觉协方差与置信度说明

## 1. 这份文档讲什么

这份文档专门解释当前 `splg_fusion` 工程里，视觉定位结果的“置信度”是怎么被评估出来的，以及它最后是如何变成融合里使用的视觉量测协方差矩阵的。

这里说的“视觉协方差”，本质上是在回答下面这个问题：

- 对于某一帧视觉定位结果，系统认为它在 `x / y / z` 三个方向上大概有多不确定？
- 这个不确定度应该在融合时给多大权重？

对应实现主要分布在：

- [fusion_types.h](file:///home/jasonxue/splg_lo_origin/splg_fusion/include/splg_fusion/fusion/fusion_types.h)
- [sp_lg.cpp](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/vision/sp_lg.cpp)
- [fusion_io_worker.h](file:///home/jasonxue/splg_lo_origin/splg_fusion/include/splg_fusion/io/fusion_io_worker.h)
- [online_ypr_crop_node_fusion.cpp](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/fusion/online_ypr_crop_node_fusion.cpp)

## 2. 总体思路

当前实现不是简单地把“视觉成功”当成一个固定精度的观测，而是走了下面这条链路：

1. `sp_lg` 先从匹配和几何关系里提取一批质量指标。
2. 这些指标先被整理成像素层和地面层的误差统计量。
3. 再通过经验趋势模型，把当前帧的质量指标映射成 `x / y / z` 三个方向的预测方差。
4. 融合模块把这些预测方差和最小下限结合，构成视觉量测协方差矩阵 `R_visual`。
5. 如果开启了自适应协方差，还会根据 NIS 再乘一个缩放因子。

所以这套逻辑的核心特点是：

- 不是固定协方差。
- 也不是只看单一指标。
- 而是把“匹配质量 + 几何稳定性 + 地形条件 + PnP 质量”综合成一个观测不确定度。

## 3. 视觉置信度的原始来源

视觉置信度的原始信息来自 `sp_lg` 单帧定位阶段。

在 [sp_lg.cpp](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/vision/sp_lg.cpp) 里，会生成很多与当前帧质量有关的指标，比较重要的有：

- `localization_used_points`
  - 真正参与定位求解的点数。

- `localization_inlier_points`
  - PnP 内点数。

- `sp_lg_pnp_inlier_rate`
  - PnP 内点率。

- `sp_lg_homography_inlier_rate`
  - 单应估计内点率。

- `localization_reproj_error`
  - PnP 重投影误差。

- `sp_lg_point_spread_img_ratio`
  - 匹配点在图像上的分布是否足够分散。

- `sp_lg_point_spread_map_xy_m2`
  - 匹配点在地图平面上的覆盖范围。

- `sp_lg_depth_spread_m`
  - 3D 点深度分布范围。

- `sp_lg_obliqueness_deg`
  - 视角斜视程度。

- `sp_lg_terrain_relief_m`
  - 地形起伏程度。

这些量本质上都在反映一件事：

- 当前这帧视觉定位是不是“几何条件好、匹配稳定、结果可信”。

## 4. 双几何一致性统计是怎么来的

当前实现里，一个很关键的中间量叫 `dual geometry` 统计，也就是：

- 单应几何和 PnP 几何之间的一致性。

在 [sp_lg.cpp:L1269-L1339](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/vision/sp_lg.cpp#L1269-L1339) 中，系统会先统计三类残差：

- `sigma_h`
  - 单应预测和实际图像点之间的残差协方差。

- `sigma_p`
  - PnP 预测和实际图像点之间的残差协方差。

- `sigma_d`
  - PnP 预测和单应预测彼此之间的分歧协方差。

然后把它们组合成像素平面上的综合误差协方差：

```text
sigma_px = 0.25 * sigma_h + 0.25 * sigma_p + 0.50 * sigma_d + 1e-4 * I
```

这里权重的直观含义是：

- 单应残差有参考价值。
- PnP 残差也有参考价值。
- 但“两种几何结果互相打架”通常更危险，所以给了更高权重。

之后系统再通过局部雅可比，把像素误差传播到地面平面：

- 得到 `var_x_m2`
- 得到 `var_y_m2`
- 得到 `cov_xy_m2`

这些量就是视觉在地面坐标系里的基础方差信息。

对应输出字段在：

- [sp_lg.cpp:L1413-L1432](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/vision/sp_lg.cpp#L1413-L1432)
- [fusion_io_worker.h:L584-L599](file:///home/jasonxue/splg_lo_origin/splg_fusion/include/splg_fusion/io/fusion_io_worker.h#L584-L599)

## 5. 从质量指标到预测方差

基础 `dual geometry` 方差还不是最终用于融合的视觉协方差。

在 [fusion_types.h:L492-L557](file:///home/jasonxue/splg_lo_origin/splg_fusion/include/splg_fusion/fusion/fusion_types.h#L492-L557) 中，`computeVisualAxisSigmaPrediction()` 会进一步做一层“质量趋势修正”。

### 5.1 `x` 方向

`x` 方向的预测方差，是以 `sp_lg_dual_geom_var_x_m2` 为底，再乘一个质量倍率：

- `localization_reproj_error`
- `sp_lg_pnp_inlier_rate`
- `sp_lg_homography_inlier_rate`
- `sp_lg_point_spread_img_ratio`
- `sp_lg_depth_spread_m`
- `sp_lg_obliqueness_deg`

这些指标通过 `computeClippedMultiplierFromTrend()` 被转成一个倍率：

- 越差的质量指标，倍率越大。
- 越好的质量指标，倍率越小。
- 最后倍率会被裁剪在 `[0.8, 4.0]` 内。

直观上就是：

- 这帧虽然基础几何方差是某个值，
- 但如果重投影误差偏大、内点率偏低、点分布不好，
- 就把这个方差继续放大。

### 5.2 `y` 方向

`y` 方向和 `x` 方向类似，也是对 `sp_lg_dual_geom_var_y_m2` 再乘一个质量倍率。

用到的指标略有不同：

- `localization_reproj_error`
- `sp_lg_pnp_inlier_rate`
- `sp_lg_homography_inlier_rate`
- `sp_lg_dual_geom_h_rmse_px`
- `sp_lg_obliqueness_deg`

倍率裁剪范围比 `x` 更窄：

- `[0.4, 2.5]`

这说明当前实现认为：

- `y` 方向的经验修正不需要像 `x` 方向那样放得那么猛。

### 5.3 `z` 方向

`z` 方向没有直接沿用 `dual_geom_var_z`，而是单独用一个对数线性模型预测：

- `localization_reproj_error`
- `sp_lg_terrain_relief_m`
- `localization_inlier_points`
- `sp_lg_pnp_inlier_rate`

原因也比较好理解：

- 高度方向往往比平面方向更难稳定。
- 它更依赖地形起伏、视角和 PnP 约束条件。
- 所以单独拟合一个经验模型更合理。

## 6. 最终视觉量测协方差矩阵怎么构造

真正给融合用的视觉量测协方差，是在 [fusion_types.h:L569-L596](file:///home/jasonxue/splg_lo_origin/splg_fusion/include/splg_fusion/fusion/fusion_types.h#L569-L596) 的 `computeVisualMeasurementCovariance()` 里构造的。

它的规则可以概括成三步。

### 6.1 先拿预测方差

优先使用：

- `pred.pred_x_var_m2`
- `pred.pred_y_var_m2`
- `pred.pred_z_var_m2`

也就是上一节从质量指标预测出来的方差。

### 6.2 如果预测不出来，就退回备用值

对 `x / y`：

- 先退回 `sp_lg_dual_geom_var_x_m2`
- 或 `sp_lg_dual_geom_var_y_m2`

对 `z`：

- 如果没有预测值，就根据 `sp_lg_scale_h_m_per_px` 估一个备用高度方差
- 形式大概是 `(4 * scale_h_m_per_px)^2`

这相当于说：

- 如果没有更好的高度质量预测，
- 就按当前地面尺度给一个偏保守的高度不确定度。

### 6.3 再加最小下限

配置里还有两个下限：

- `kf_visual_min_xy_sigma_m`
- `kf_visual_min_z_sigma_m`

在当前配置 [fusion_config.yaml:L108-L118](file:///home/jasonxue/splg_lo_origin/splg_fusion/config/fusion_config.yaml#L108-L118) 中，默认是：

- `kf_visual_min_xy_sigma_m: 3.0`
- `kf_visual_min_z_sigma_m: 4.0`

这意味着：

- 就算某一帧视觉质量看起来特别好，
- 融合里也不会把它当成“几乎零误差”的完美观测。

这样做的目的很明确：

- 防止某些偶然高质量帧把滤波器拉得过猛。

### 6.4 最终形式

最后得到的 `R_visual` 是一个 `3x3` 矩阵：

```text
R_visual =
[ var_x   cov_xy   0 ]
[ cov_xy  var_y    0 ]
[ 0       0       var_z ]
```

其中：

- `var_x`
- `var_y`
- `var_z`

来自上面的选择逻辑，`cov_xy` 则优先使用 `sp_lg_dual_geom_cov_xy_m2`。

## 7. 这套“置信度”到底怎么参与融合

在融合更新时，视觉量测协方差首先被构造成：

- `R_visual = computeVisualMeasurementCovariance(rec, opt_)`

见 [online_ypr_crop_node_fusion.cpp:L360-L363](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/fusion/online_ypr_crop_node_fusion.cpp#L360-L363)

然后它会直接进入标准量测更新中的创新协方差：

```text
S = P_pred + R_visual
```

其中：

- `P_pred` 是传播后的状态协方差。
- `R_visual` 是当前视觉观测的不确定度。

视觉协方差越大，就意味着：

- 系统越不信这一帧视觉。
- 卡尔曼增益会更小。
- 视觉对状态的拉回作用更弱。

视觉协方差越小，就意味着：

- 系统越信这一帧视觉。
- 卡尔曼增益更大。
- 这帧视觉更能主导更新结果。

所以“视觉协方差”其实就是：

- 这帧视觉在融合里话语权的数值化表达。

## 8. NIS 门控和自适应协方差

当前实现里，视觉观测不是无条件接受的，还会再走两步。

### 8.1 先计算 NIS

在 [online_ypr_crop_node_fusion.cpp:L381-L389](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/fusion/online_ypr_crop_node_fusion.cpp#L381-L389) 中：

```text
innovation = z_enu - predicted_enu
S = predicted_cov + R
NIS = innovation^T * S^-1 * innovation
```

直观理解：

- 如果当前视觉观测和预测值差得不算离谱，
- 而且这种偏差又在协方差允许范围内，
- 那 NIS 会比较小。

反过来：

- 如果视觉观测偏差很大，
- 或者协方差给得过小，
- NIS 就会变大。

### 8.2 再做门控

配置里有：

- `kf_visual_nis_gate`
- `kf_visual_reject_on_gate`

当前配置是：
- `kf_visual_nis_gate: 16.266`
- `kf_visual_reject_on_gate: 1`

也就是说当前默认行为是：
- 会计算 NIS
- 会记录 `visual_gate_passed`
- **如果 NIS 超过阈值，直接拒绝该观测**，将其视为异常值 (Outlier)

这是系统抵御严重误匹配的最后一道防线。如果把 `kf_visual_reject_on_gate` 关闭 (设为 0)，那么高 NIS 的观测虽然权重会受影响，但依然会被强行融入，可能导致轨迹突变。

### 8.3 自适应协方差缩放

在 [online_ypr_crop_node_fusion.cpp:L281-L292](file:///home/jasonxue/splg_lo_origin/splg_fusion/src/fusion/online_ypr_crop_node_fusion.cpp#L281-L292) 中，`adaptVisualCovarianceScaleLocked()` 会根据：

- 当前 `nis`
- 目标 `kf_visual_adaptive_target_nis`
- 平滑系数 `kf_visual_adaptive_alpha`

去更新：

- `continuous_filter_.visual_cov_scale`

直观解释是：

- 如果最近观测总是比预期更“冲”，说明视觉协方差偏小了，需要放大。
- 如果最近观测总是比预期更“保守”，说明视觉协方差偏大了，可以适当缩小。

当前配置里：

- `kf_visual_adaptive_enable: 0`

所以这套自适应逻辑默认是关闭的。

## 9. 这套视觉置信度设计的优点

### 9.1 比固定协方差更贴近实际

不同帧视觉质量差别很大：

- 有些帧匹配点多、内点率高、几何稳定
- 有些帧虽然也“成功”，但质量明显差

固定协方差无法区分这些差异，而当前实现可以。

### 9.2 把几何退化信息纳入了协方差

系统不是只看 PnP 成没成功，而是进一步看：

- 单应和 PnP 是否一致
- 局部雅可比是否病态
- 地面尺度是否合理

这让协方差更接近真实几何不确定度。

### 9.3 给融合提供了更稳妥的权重调节

最终视觉协方差进入卡尔曼更新时，会直接影响卡尔曼增益。

所以这套设计的意义不是“多输出几个统计字段”，而是：

- 真正决定视觉观测在融合里有多大权重。

## 10. 这套方法的局限

### 10.1 仍然带经验成分

`computeVisualAxisSigmaPrediction()` 里那组斜率和参考值，本质上是经验趋势模型。

这意味着：

- 对当前数据集可能有效
- 对别的数据、别的场景未必最优

### 10.2 协方差是近似对角主导的

虽然保留了 `cov_xy`，但整体上还是以 `x / y / z` 三轴方差为主。

这是一种工程上足够实用的近似，但并不是最完整的视觉后验协方差。

### 10.3 `z` 方向仍然最依赖经验模型

平面方向还能借助 `dual geometry` 的投影传播得到更直接的几何方差，
但高度方向目前还是更依赖经验回归和备用尺度估计。

## 11. 一句话总结

当前工程里的视觉协方差不是拍脑袋给的固定数，而是：

- 先从 `sp_lg` 的匹配质量、单应几何、PnP 几何、地形和视角信息中提取质量指标，
- 再把这些指标映射成 `x / y / z` 三个方向的预测方差，
- 最后加上下限、可选的自适应缩放，形成真正进入融合的视觉量测协方差矩阵。

如果要用最直白的话来概括，就是：

- 视觉置信度高，协方差就小，融合更信它。
- 视觉置信度低，协方差就大，融合就少信它。
