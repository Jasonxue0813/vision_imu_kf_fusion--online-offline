# splg_fusion

`splg_fusion` 是一个面向无人机巡检与导航场景的 视觉-IMU-GNSS 多源融合定位工程。

注意：系统运行时，文件名称为'splg_fusion'

      本系统，匹配算法用到的是 onnx版本的 superpoint+lightglue

系统的核心目标是：利用机载图像在卫星底图上的匹配结果提供绝对位置校正（低频），再结合 IMU 数据进行状态预积分（高频），并通过严格的时序管理和状态重放，最终输出高频、平滑、鲁棒的在线定位结果。

本文档主要说明：
1. 算法的主要流程和思路
2. 工程目录框架
3. 编译与运行方式
4. 配置文件的作用

---

## 1. 算法主要流程和思路

### 1.1 输入数据
系统主要接收三类输入：
- **图像 (Image)**：无人机下视相机的原始图像。
- **IMU**：角速度和线加速度，提供高频的相对运动信息。
- **GNSS**：经纬高和地面航向参考。

其中：
- 图像负责提供相对可靠的**绝对位置校正**。
- IMU 负责在两次视觉结果之间做**高频连续传播**。
- GNSS 主要用于**时间对齐**、**航向估计**、**裁图参考**和最终结果对比评估。

### 1.2 整体思路
整个算法可以理解成“低频绝对校正 + 高频连续传播”的严格时序组合。

#### 第一步：图像预处理与动态裁图
每当图像到来时，系统会先做畸变校正，并利用当前的航向估计对图像进行旋转补偿。
**动态裁图策略**：为了保证连续匹配的平滑性和准确率，卫星地图的裁图中心选取策略为：
1. 优先使用**上一次成功视觉融合的位置**（前提是距当前时间不超过 1 秒）。
2. 若上一次成功融合已超过 1 秒，则退化使用**最近一次的连续滤波器状态，并利用 IMU 积分外推至当前图像时刻的预测位置**。
3. 最差情况下（如系统刚启动时），回退到当前帧对齐的原始 GNSS 坐标。
以此截取适量的卫星子图，极大降低后续视觉匹配的计算量。

#### 第二步：视觉定位 (`sp_lg`)
裁图完成后，系统调用 `sp_lg` 子模块，在相机旋转图和卫星地图裁片之间做特征提取 (SuperPoint)、匹配 (LightGlue)、几何验证与位姿恢复 (PnP)，得到当前图像对应的绝对地理位置估计。

#### 第三步：GNSS 航向参考
系统利用一段时间窗口内的 GNSS 轨迹估计平滑的航向 (Heading)。该航向作为视觉与 IMU 融合时的重要参考量，用于约束朝向和短时传播方向。

#### 第四步：时序严格的视觉-惯性连续融合
由于视觉定位算法存在较长的计算延迟，系统内部维护了严格按时间轴排序的队列：
- **视觉锚点**：视觉定位成功后，该帧的结果作为“视觉锚点”。
- **状态回滚与重放**：系统利用 IMU 的连续预积分，当接收到延迟到达的视觉观测时，会将状态回滚至视觉帧产生时刻，执行卡尔曼滤波 (KF) 状态更新，随后重新利用缓存的 IMU 数据重放 (Replay) 积分至当前最新时刻。
- **IMU-only 模式**：如果长时间没有有效的视觉观测，系统将退化为 IMU-only 的连续积分外推模式。

#### 第五步：多频态结果输出
系统提供两种核心输出，满足不同层级的需求：
- `fusion_position.csv`：**主融合结果**，通常与 GNSS (如 10Hz) 频率对齐，适合作为对外的最终主定位参考。
- `continuous_fusion.csv`：**高频连续状态输出**。经过优化，该文件固定以 **50ms** 间隔节流输出精简的 8 列核心状态（时间戳、经纬高、速度等），文件体积小，非常适合做高频轨迹的动态分析和误差对比。

#### 第六步：异常容错与大 Gap 重启
实际飞行中传感器可能出现断流或丢包。系统具备多层退化与容错处理：
- 视觉暂时缺失时，允许短时间 IMU-only 传播。
- **大 Gap 自动重启**：当检测到 IMU 数据存在过大时间断层（大 Gap）时，系统不会跨越断层强行积分（避免积分爆炸跑飞），而是会自动重置速度与协方差，并重新启动连续滤波器 (Continuous Filter)，保证后续数据恢复后能立刻重新收敛。

---

## 2. 工程文件框架

工程主要分为入口层、主流程层、视觉模块、融合模块、配置模块和测试分析脚本几部分。

### 2.1 顶层目录
```text
splg_fusion/
├── CMakeLists.txt
├── README.md
├── config/             # YAML 配置文件目录
├── include/            # C++ 头文件
├── src/                # C++ 源码实现
├── test/               # Python 离线分析与可视化脚本
└── visual_covariance_confidence_readme.md
```

### 2.2 `src/` 源码目录
- `src/main.cpp`: 程序入口，负责初始化 ROS、解析命令行、加载配置。
- `src/app/online_ypr_crop_node_core.cpp`: 主节点生命周期管理（启动、离线 bag 回放）。
- `src/app/online_ypr_crop_node_runtime.cpp`: 运行时消息处理（处理图像、IMU、GNSS 回调）。
- `src/app/online_ypr_crop_node_utils.cpp`: 辅助逻辑（日志、记录整理、文件落盘）。
- `src/vision/online_ypr_crop_node_vision.cpp`: 图像预处理、姿态补偿、**动态地图裁图**。
- `src/vision/sp_lg.cpp`: 调用 ONNX Runtime 运行 SuperPoint + LightGlue 视觉定位。
- `src/fusion/online_ypr_crop_node_fusion.cpp`: **融合核心逻辑**（包含视觉锚点更新、IMU 预积分传播、延迟回滚重放、大 Gap 异常处理）。
- `src/global_heading.cpp`: GNSS 航向估计。

### 2.3 `test/` 分析与评估脚本目录
`test/` 目录下包含了多个用于离线分析的 Python 脚本：
- `continuous_fusion_error.py`: 读取 `continuous_fusion.csv` 和 GNSS 真值，计算 Up 和 Horizontal 误差并绘制随时间变化的误差图与 2D 航迹对比图。
- `flight_path.py`: 综合绘制 GNSS、视觉、融合结果的完整二维飞行航迹图。
- `evaluation2.py`: 对 `vision_position.csv`、`fusion_position.csv` 与 GNSS 结果进行对齐与量化误差统计。

---

## 3. 如何编译和运行

### 3.1 依赖环境
- C++17
- OpenCV
- GDAL
- Threads
- ROS / catkin 组件 (`roscpp`, `rosbag`, `sensor_msgs`, `std_msgs`)
- **ONNX Runtime** (编译 `sp_lg` 视觉模块必须，若无则只编译融合主体跳过视觉推理)

### 3.2 编译方式
推荐在工程根目录创建 `build/` 目录进行外围编译。如果 ONNX Runtime 不在系统默认路径，请使用 `-DORT_DIR` 指定：

```bash
cd /home/jasonxue/splg_lo_origin/splg_fusion
mkdir -p build && cd build
cmake .. -DORT_DIR=/path/to/onnxruntime   # 替换为你的 ONNX Runtime 实际路径
(cmake .. -DORT_DIR=/home/jasonxue/splg_lo_origin/onnxruntime-linux-x64-gpu-1.24.2)
cmake --build . -j
```
编译成功后会生成可执行文件：`build/visual_imu_fusion` 和 `build/sp_lg`。

### 3.3 离线运行 (Rosbag 回放)
默认配置中，程序工作在离线模式，直接读取指定的 rosbag 文件：

```bash
cd /home/jasonxue/splg_lo_origin/splg_fusion
./build/visual_imu_fusion --config config/fusion_config.yaml
```

### 3.4 在线运行 (ROS 节点)
若需要直接接收实时的 ROS Topic 数据，只需将 `config/fusion_config.yaml` 中的 `input.mode` 改为 `online`，运行后程序将持续监听图像、GNSS、IMU 话题并实时计算融合输出。

### 3.5 运行后输出文件
运行完成后，默认会在 `result/` 目录下生成：
- `vision_position.csv`: 单帧纯视觉定位结果（即 `localization_csv`）。
- `sp_lg_result.csv`: 包含底层特征提取、几何匹配内点率、重投影误差等详细质量指标。
- `fusion_position.csv`: 融合滤波后的主结果 (频率对齐 GNSS)。
- `continuous_fusion.csv`: 高频连续传播轨迹结果 (固定 50ms 间隔)。
- `gnss_position.csv`: GNSS 参考真值。
- `sampled_frame_geo.csv`: 对齐并采样后的图像帧及基础姿态信息。

---

## 4. 配置文件的作用

工程配置采用“主程序配置 (`fusion_config.yaml`) + 视觉网络配置 (`config.yaml`)”的分层结构。

### 4.1 `config/fusion_config.yaml`
这是系统的核心配置文件，控制整个视觉-IMU-GNSS 的调度与融合逻辑。主要包含：

- **`input`**: 运行模式 (`offline`/`online`)、rosbag 路径、地图 TIF 文件路径、相机内外参文件。
- **`topics`**: 绑定的 ROS 话题名称。
- **`tuning`**: 最核心的算法行为参数：
  - **IO 开销控制**: `persist_frame_artifacts` (设为 `0` 可关闭中间图像的保存，极大提升运行速度并节省硬盘)。
  - **IMU 跟踪与融合**: 姿态融合增益、加速度低通、零偏估计阈值、大 Gap 判断阈值 (`imu_preintegration_max_gap_sec`)。
  - **裁图策略**: 裁图物理尺寸 (`crop_half_extent_e_m`)。
  - **视觉锚点策略**: 特征点内点率阈值、最大重投影误差、软更新参数、视觉连续丢失的最大容忍时间 (`fusion_imu_only_max_duration_sec`)。
  - **卡尔曼滤波噪声**: 过程噪声矩阵设置与视觉量测的自适应协方差缩放参数。
