# Config Layout

- `fusion_config.yaml`: `visual_imu_fusion` 主程序配置，包含输入、topic、输出路径、IMU 参数、视觉锚点参数和队列参数。
- `config.yaml`: `sp_lg` 配置，包含模型路径、特征提取、匹配和运行时参数。

建议维护方式：

- 主流程相关参数优先放在 `fusion_config.yaml`
- `sp_lg` 子进程/模型相关参数放在 `config.yaml`
- 输出路径统一集中在 `fusion_config.yaml` 的路径配置区
