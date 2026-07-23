/**
 * @file online_ypr_crop_node_core.cpp
 * @brief `OnlineYprCropNode` 的核心生命周期实现。
 *
 * 这个文件负责主节点对象的构造、输出准备、在线启动、
 * 离线 rosbag 回放，以及程序退出前的收尾逻辑。
 */

#include "splg_fusion/app/online_ypr_crop_node.h"

// 功能：`OnlineYprCropNode` 的构造函数，负责建立主流程运行所需的初始对象和默认状态。
OnlineYprCropNode::OnlineYprCropNode(Options opt)
    : opt_(std::move(opt)),
      cam_(loadCameraModel(opt_.cam_yaml)),
      ext_(loadCameraExtrinsics(opt_.calib_txt, opt_.camera_name)),
      map_raster_(opt_.map_tif, false),
      dem_raster_(opt_.dem_tif, true),
      heading_est_(opt_.window_sec,
                   opt_.gnss_rate_hz,
                   opt_.min_speed_mps,
                   opt_.max_residual_m,
                   opt_.heading_history_sec),
      imu_tracker_(opt_),
      fix_history_keep_sec_(std::max(30.0, opt_.window_sec * 4.0 + 10.0)) {
    if (opt_.localization_csv.empty()) {
        opt_.localization_csv = opt_.out_dir + "/localization_results.csv";
    }
    if (opt_.sp_lg_result_csv.empty()) {
        opt_.sp_lg_result_csv = opt_.out_dir + "/sp_lg_result.csv";
    }
    if (opt_.fix_fusion_csv.empty()) {
        opt_.fix_fusion_csv = opt_.out_dir + "/fix_fusion_localization.csv";
    }
    if (opt_.gnss_position_csv.empty()) {
        opt_.gnss_position_csv = opt_.out_dir + "/gnss_position.csv";
    }
    if (opt_.sampled_geo_csv.empty()) {
        opt_.sampled_geo_csv = opt_.out_dir + "/sampled_frame_geo.csv";
    }
    if (opt_.high_rate_fusion_csv.empty()) {
        opt_.high_rate_fusion_csv = opt_.out_dir + "/continuous_high_rate_fusion.csv";
    }
    const double min_scale = std::max(1e-6, opt_.kf_visual_adaptive_min_scale);
    const double max_scale = std::max(min_scale, opt_.kf_visual_adaptive_max_scale);
    continuous_filter_.visual_cov_scale =
        std::clamp(opt_.kf_visual_adaptive_initial_scale, min_scale, max_scale);
    if (opt_.enable_sp_lg) {
        sp_lg_template_ = loadSpLgTemplateYaml(opt_.sp_lg_template_yaml);
        sp_lg_executable_ = findSpLgExecutable(opt_);
    }
    ensureDir(opt_.out_dir);
    records_.reserve(opt_.max_output_frames > 0 ? static_cast<size_t>(opt_.max_output_frames) : 256);
    online_mode_ = (opt_.input_mode == "online");
    std::cout << "[Mode] " << (online_mode_ ? "online ROS subscriber" : "offline rosbag") << "\n";
    std::cout << "[Map] " << opt_.map_tif << " extent_lonlat " << map_raster_.describeExtentLonLat() << "\n";
    std::cout << "[DEM] " << opt_.dem_tif << " extent_lonlat " << dem_raster_.describeExtentLonLat() << "\n";
    if (opt_.enable_sp_lg) {
        std::cout << "[sp_lg] executable=" << sp_lg_executable_ << "\n";
        std::cout << "[sp_lg] localization_csv=" << opt_.localization_csv << "\n";
        std::cout << "[sp_lg] sp_lg_result_csv=" << opt_.sp_lg_result_csv << "\n";
    }
    std::cout << "[Fusion] fix_fusion_csv=" << opt_.fix_fusion_csv << "\n";
    std::cout << "[Fusion] gnss_position_csv=" << opt_.gnss_position_csv << "\n";
    std::cout << "[GeoCSV] sampled_geo_csv=" << opt_.sampled_geo_csv << "\n";
    std::cout << "[KF] high_rate_fusion_csv=" << opt_.high_rate_fusion_csv << "\n";
}

// 功能：`OnlineYprCropNode` 的析构函数，负责停止线程并清理资源。
OnlineYprCropNode::~OnlineYprCropNode() {
    stopProcessingThread();
    try {
        flushFixFusionCsvOutput();
    } catch (const std::exception& e) {
        std::cerr << "[Fusion] failed to write csv during shutdown: " << e.what() << "\n";
    }
    shutdownSpLgWorker();
}

// 功能：启动在线订阅模式所需的 ROS 订阅与工作线程。
void OnlineYprCropNode::start(ros::NodeHandle& nh) {
    online_mode_ = true;
    prepareOutputs();
    startProcessingThread();
    fix_fusion_pub_ =
        nh.advertise<std_msgs::String>(vis_fix_fusion_topic_, std::max(1, opt_.fix_queue_size), true);
    image_sub_ = nh.subscribe(opt_.image_topic, std::max(1, opt_.image_queue_size),
                              &OnlineYprCropNode::imageCallback, this);
    fix_sub_ = nh.subscribe(opt_.fix_topic, std::max(1, opt_.fix_queue_size),
                            &OnlineYprCropNode::fixCallback, this);
    imu_sub_ = nh.subscribe(opt_.imu_topic, std::max(1, opt_.imu_queue_size),
                            &OnlineYprCropNode::imuCallback, this);
    ROS_INFO_STREAM("[Start] waiting for live topics: image=" << opt_.image_topic
                                                              << " fix=" << opt_.fix_topic
                                                              << " imu=" << opt_.imu_topic
                                                              << " fusion_pub=" << vis_fix_fusion_topic_
                                                              << " | subscriber_queue(image/fix/imu)="
                                                              << std::max(1, opt_.image_queue_size) << "/"
                                                              << std::max(1, opt_.fix_queue_size) << "/"
                                                              << std::max(1, opt_.imu_queue_size));
}

// 功能：按时间顺序回放离线 rosbag 并处理其中的传感器消息。
void OnlineYprCropNode::runOfflineBag() {
    if (opt_.bag_path.empty()) {
        throw std::runtime_error("Offline mode requires input.bag_path to point to a rosbag.");
    }

    online_mode_ = false;
    prepareOutputs();
    startProcessingThread();

    std::cout << "[Start] offline rosbag=" << opt_.bag_path
              << " image=" << opt_.image_topic
              << " fix=" << opt_.fix_topic
              << " imu=" << opt_.imu_topic
              << " start_image_index=" << opt_.start_image_index
              << " end_image_index=" << opt_.end_image_index
              << " vision_dispatch=single_frame_serial_drop_while_busy"
              << " replay_pacing=real_time_by_msg_stamp"
              << "\n";

    rosbag::Bag bag;
    bag.open(opt_.bag_path, rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.push_back(opt_.fix_topic);
    topics.push_back(opt_.imu_topic);
    topics.push_back(opt_.image_topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    bool replay_clock_initialized = false;
    double replay_start_stamp_sec = 0.0;
    double replay_last_stamp_sec = 0.0;
    auto replay_wall_start = std::chrono::steady_clock::now();

    auto paceOfflineReplayToStamp = [&](double stamp_sec) {
        if (!std::isfinite(stamp_sec) || stamp_sec <= 0.0) return;

        if (!replay_clock_initialized) {
            replay_clock_initialized = true;
            replay_start_stamp_sec = stamp_sec;
            replay_last_stamp_sec = stamp_sec;
            replay_wall_start = std::chrono::steady_clock::now();
            return;
        }

        const double monotonic_stamp_sec = std::max(stamp_sec, replay_last_stamp_sec);
        replay_last_stamp_sec = monotonic_stamp_sec;
        const double replay_elapsed_sec =
            std::max(0.0, monotonic_stamp_sec - replay_start_stamp_sec);
        const auto target_wall_time =
            replay_wall_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                    std::chrono::duration<double>(replay_elapsed_sec));
        std::this_thread::sleep_until(target_wall_time);
    };

    for (const rosbag::MessageInstance& m : view) {
        if (stopped_) break;

        const std::string topic = m.getTopic();
        if (topic == opt_.fix_topic || "/" + topic == opt_.fix_topic) {
            sensor_msgs::NavSatFix::ConstPtr fix_msg = m.instantiate<sensor_msgs::NavSatFix>();
            if (fix_msg) {
                paceOfflineReplayToStamp(fix_msg->header.stamp.toSec());
                handleFixMessage(*fix_msg);
            }
        } else if (topic == opt_.imu_topic || "/" + topic == opt_.imu_topic) {
            sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
            if (imu_msg) {
                paceOfflineReplayToStamp(imu_msg->header.stamp.toSec());
                handleImuMessage(*imu_msg);
            }
        } else if (topic == opt_.image_topic || "/" + topic == opt_.image_topic) {
            sensor_msgs::Image::ConstPtr image_msg = m.instantiate<sensor_msgs::Image>();
            if (image_msg) {
                paceOfflineReplayToStamp(image_msg->header.stamp.toSec());
                handleImageMessage(*image_msg);
            }
        }

        if (stopped_) break;
        if (opt_.end_image_index > 0 &&
            image_count_ >= static_cast<size_t>(opt_.end_image_index) &&
            pending_frames_.empty()) {
            break;
        }
    }

    bag.close();
    stopProcessingThread();
    flushFixFusionCsvOutput();
    writeSummary(opt_.out_dir + "/summary.txt",
                 opt_,
                 records_,
                 fix_count_,
                 imu_count_,
                 image_count_,
                 sampled_count_);

    std::cout << "[Done] offline records=" << records_.size()
              << " seen_images=" << image_count_
              << " accepted_images=" << sampled_count_
              << " out_dir=" << opt_.out_dir << "\n";
}

// 功能：刷新 fix 融合 CSV 的缓冲输出。
void OnlineYprCropNode::flushFixFusionCsvOutput() {
    if (fix_fusion_csv_written_) return;
    if (!online_mode_) {
        writeFixFusionCsvOffline();
    }
    fix_fusion_csv_written_ = true;
}

// 功能：初始化结果目录与各类输出文件。
void OnlineYprCropNode::prepareOutputs() {
    if (outputs_prepared_) return;
    continuous_latency_reference_ready_ = false;
    continuous_latency_reference_stamp_sec_ = std::numeric_limits<double>::quiet_NaN();
    continuous_latency_reference_wall_sec_ = std::numeric_limits<double>::quiet_NaN();
    continuous_last_csv_write_stamp_sec_ = std::numeric_limits<double>::quiet_NaN();
    ensureDir(resultImageDir().string());
    writeSampledGeoCsvHeader(opt_.sampled_geo_csv);
    writeFixFusionCsvHeader(opt_.fix_fusion_csv);
    writeImuPositionCsvHeader(opt_.imu_position_csv);
    writeGnssPositionCsvHeader(opt_.gnss_position_csv);
    writeContinuousFusionCsvHeader(opt_.high_rate_fusion_csv);
    if (opt_.enable_sp_lg) {
        writeLocalizationCsvHeader(opt_.localization_csv);
        writeSpLgResultCsvHeader(opt_.sp_lg_result_csv);
        ensureSpLgWorkerStarted();
    }
    outputs_prepared_ = true;
}
