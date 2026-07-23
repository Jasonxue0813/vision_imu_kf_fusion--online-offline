/**
 * @file main.cpp
 * @brief 主程序入口文件。
 *
 * 这个文件只负责程序启动层面的工作：
 * 1. 初始化 ROS；
 * 2. 解析命令行参数；
 * 3. 加载配置；
 * 4. 判断在线 / 离线运行模式；
 * 5. 创建并启动 `OnlineYprCropNode`。
 */

// 引入主流程类及其依赖的配置、工具函数声明。
#include "splg_fusion/app/online_ypr_crop_node.h"

// 程序入口：负责组装配置并把控制权交给主节点类。
int main(int argc, char** argv) {
    // 使用 try/catch 包住主流程，避免异常直接导致程序无信息退出。
    try {
        // 初始化 ROS 节点，节点名为 `splg_visual_imu_fusion`。
        ros::init(argc, argv, "splg_visual_imu_fusion");
        // 去掉 ROS remap 参数，保留程序自身真正需要解析的命令行参数。
        const std::vector<std::string> non_ros_args = stripRosRemapArgs(argc, argv);
        // 解析主程序参数，得到配置文件路径和位置参数覆盖项。
        const CliArgs cli = parseArgs(non_ros_args);
        // 创建一份运行配置对象，后续会用 yaml 和命令行参数填充它。
        Options opt;
        // 从 yaml 配置文件中读取主程序运行参数。
        loadOptionsYaml(cli.config_yaml, opt);
        // 用命令行位置参数对部分配置项进行覆盖。
        applyPositionalOverrides(opt, cli.positional_overrides);
        // 根据当前配置判断应该跑离线 bag 模式还是在线订阅模式。
        const bool offline_mode = shouldUseOfflineMode(opt);
        // 把最终输入模式写回配置，供后续主流程统一使用。
        opt.input_mode = offline_mode ? "offline" : "online";
        // 根据准备好的配置创建主流程节点对象。
        OnlineYprCropNode node(opt);
        // 如果是离线模式，就直接回放 rosbag 并执行完整处理流程。
        if (offline_mode) {
            // 运行离线 bag 处理主循环。
            node.runOfflineBag();
        // 否则进入在线订阅模式。
        } else {
            // 创建 ROS 节点句柄，用于建立订阅器等在线资源。
            ros::NodeHandle nh;
            // 启动在线模式下的订阅、线程和内部状态。
            node.start(nh);
            // 进入 ROS 事件循环，持续响应传感器消息。
            ros::spin();
        }
        // 主流程正常结束时返回 0，表示成功退出。
        return 0;
    // 捕获标准异常并打印错误信息，避免静默失败。
    } catch (const std::exception& e) {
        // 把异常内容输出到标准错误流，便于定位问题。
        std::cerr << "[Error] " << e.what() << "\n";
        // 发生异常时返回非零值，表示程序执行失败。
        return 1;
    }
}
