/**
 * @file online_ypr_crop_node.h
 * @brief 主流程类 `OnlineYprCropNode` 的声明文件。
 *
 * 这个头文件定义了主节点类及其内部状态、嵌套结构和方法接口。
 * 整个视觉定位、消息调度、融合输出流程都由这个类统一编排。
 */

#pragma once

#include "splg_fusion/fusion/fusion_types.h"
#include "splg_fusion/fusion/fusion_geometry.h"
#include "splg_fusion/io/fusion_io_worker.h"

class OnlineYprCropNode {
public:
// 功能：`OnlineYprCropNode` 的构造函数，负责建立主流程运行所需的初始对象和默认状态。
    explicit OnlineYprCropNode(Options opt);

    ~OnlineYprCropNode() ;

// 功能：启动在线订阅模式所需的 ROS 订阅与工作线程。
    void start(ros::NodeHandle& nh) ;

    void runOfflineBag() ;

private:
    struct PreparedFrameState {
        double image_t = 0.0;
        double imu_query_t = 0.0;
        global_heading::HeadingSample heading;
        FixSample gps;
        double imu_raw_roll_deg = 0.0;
        double imu_raw_pitch_deg = 0.0;
    };

    enum class PendingFrameAction {
        kStop,
        kWait,
        kDrop,
        kProcess,
    };

// 功能：刷新 fix 融合 CSV 的缓冲输出。
    void flushFixFusionCsvOutput() ;

    void prepareOutputs() ;

    struct VisualAnchor {
        size_t image_index = 0;
        double t = std::numeric_limits<double>::quiet_NaN();
        GeoPoint pos;
        cv::Vec3d vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        int used_points = 0;
        int inlier_points = 0;
        double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
        double reproj_error = std::numeric_limits<double>::quiet_NaN();
        double heading_compass_rad = std::numeric_limits<double>::quiet_NaN();
        double imu_heading_rel_rad = std::numeric_limits<double>::quiet_NaN();
        bool imu_heading_valid = false;
        double sync_wait_ms = std::numeric_limits<double>::quiet_NaN();
        double processing_ms = std::numeric_limits<double>::quiet_NaN();
        double topic_to_result_ms = std::numeric_limits<double>::quiet_NaN();
    };

    struct FixFusionRow {
        size_t fix_index = 0;
        double fix_stamp_sec = std::numeric_limits<double>::quiet_NaN();
        double gt_lon = std::numeric_limits<double>::quiet_NaN();
        double gt_lat = std::numeric_limits<double>::quiet_NaN();
        double gt_alt = std::numeric_limits<double>::quiet_NaN();
        int pred_available = 0;
        double pred_lon = std::numeric_limits<double>::quiet_NaN();
        double pred_lat = std::numeric_limits<double>::quiet_NaN();
        double pred_alt = std::numeric_limits<double>::quiet_NaN();
        int imu_available = 0;
        double imu_lon = std::numeric_limits<double>::quiet_NaN();
        double imu_lat = std::numeric_limits<double>::quiet_NaN();
        double imu_alt = std::numeric_limits<double>::quiet_NaN();
        int visual_anchor_image_index = -1;
        double visual_anchor_stamp_sec = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_lon = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_lat = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_alt = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_heading_deg = std::numeric_limits<double>::quiet_NaN();
        double imu_heading_delta_deg = std::numeric_limits<double>::quiet_NaN();
        double heading_used_deg = std::numeric_limits<double>::quiet_NaN();
        double speed_mps = std::numeric_limits<double>::quiet_NaN();
        double propagate_dt_sec = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_age_sec = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_sync_wait_ms = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_processing_ms = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_topic_to_result_ms = std::numeric_limits<double>::quiet_NaN();
        double estimated_staleness_ms = std::numeric_limits<double>::quiet_NaN();
        double fix_compute_latency_ms = std::numeric_limits<double>::quiet_NaN();
        double fix_topic_to_fusion_result_ms = std::numeric_limits<double>::quiet_NaN();
        int anchor_reject_count = 0;
        std::string anchor_reject_reason;
        std::string heading_source = "none";
        std::string failure_reason;
    };

    struct ContinuousPropagation {
        bool valid = false;
        cv::Vec3d delta_enu = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d delta_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Matx33d cov_enu = cv::Matx33d::zeros();
        double heading_used_rad = std::numeric_limits<double>::quiet_NaN();
        double speed_mps = 0.0;
        std::string heading_source = "none";
    };

    struct ContinuousFusionFilterState {
        bool reference_ready = false;
        GeoPoint reference_geo;
        cv::Vec3d reference_ecef = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d east = cv::Vec3d(1.0, 0.0, 0.0);
        cv::Vec3d north = cv::Vec3d(0.0, 1.0, 0.0);
        cv::Vec3d up = cv::Vec3d(0.0, 0.0, 1.0);
        bool initialized = false;
        double state_t = std::numeric_limits<double>::quiet_NaN();
        cv::Vec3d state_enu = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d state_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Matx33d cov_enu = makeDiag3(1e6, 1e6, 1e6);
        double last_heading_used_rad = std::numeric_limits<double>::quiet_NaN();
        double last_speed_mps = 0.0;
        double anchor_heading_compass_rad = std::numeric_limits<double>::quiet_NaN();
        double anchor_imu_heading_rel_rad = std::numeric_limits<double>::quiet_NaN();
        bool anchor_imu_valid = false;
        size_t sample_index = 0;
        double visual_cov_scale = 1.0;
        double last_visual_nis = std::numeric_limits<double>::quiet_NaN();
        double last_visual_update_t = std::numeric_limits<double>::quiet_NaN();
        size_t visual_update_count = 0;
        size_t visual_reject_count = 0;
    };

    struct ContinuousFilterSnapshot {
        double state_t = std::numeric_limits<double>::quiet_NaN();
        cv::Vec3d state_enu = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d state_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Matx33d cov_enu = makeDiag3(1e6, 1e6, 1e6);
        double last_heading_used_rad = std::numeric_limits<double>::quiet_NaN();
        double last_speed_mps = 0.0;
        double anchor_heading_compass_rad = std::numeric_limits<double>::quiet_NaN();
        double anchor_imu_heading_rel_rad = std::numeric_limits<double>::quiet_NaN();
        bool anchor_imu_valid = false;
        double last_visual_update_t = std::numeric_limits<double>::quiet_NaN();
        double visual_cov_scale = 1.0;
    };

    struct TimedFusionCropSeed {
        double stamp_sec = std::numeric_limits<double>::quiet_NaN();
        GeoPoint geo;
    };

// 功能：用第一条可用的 fix 初始化融合锚点和参考状态。
    void bootstrapFusionAnchorFromFixLocked(const FixSample& fix) ;

    bool ensureContinuousFusionReferenceLocked(const GeoPoint& geo) ;

// 功能：把经纬高点转换到当前参考 ENU 坐标系。
    bool geoPointToReferenceEnuLocked(const GeoPoint& geo, cv::Vec3d& enu) const ;

    GeoPoint referenceEnuToGeoLocked(const cv::Vec3d& enu) const ;

// 功能：构建从当前状态到目标时刻的连续传播结果。
    ContinuousPropagation buildContinuousPropagationLocked(double t_from,
                                                           double t_to,
                                                           double anchor_heading_compass_rad,
                                                           double anchor_imu_heading_rel_rad,
                                                           bool anchor_imu_valid,
                                                           const cv::Vec3d& initial_velocity_enu,
                                                           double fallback_heading_rad,
                                                           double fallback_speed_mps) const ;

// 功能：构建一条 continuous fusion 输出记录。
    bool buildContinuousFusionRowLocked(const std::string& source,
                                        double stamp_sec,
                                        double heading_used_rad,
                                        double speed_mps,
                                        double propagate_dt_sec,
                                        int visual_update_applied,
                                        double visual_nis,
                                        int visual_gate_passed,
                                        const cv::Vec3d& state_enu,
                                        const cv::Matx33d& cov_enu,
                                        ContinuousFusionRow& row) ;

// 功能：利用 IMU 和当前状态预测连续融合结果。
    bool predictContinuousFusionLocked(double target_t,
                                       const std::string& source,
                                       ContinuousFusionRow* out_row) ;

// 功能：计算三维视觉观测对应的 NIS 指标。
    double computeNis3d(const cv::Vec3d& innovation, const cv::Matx33d& covariance) const ;

    void adaptVisualCovarianceScaleLocked(double nis) ;

// 功能：使用视觉观测更新连续融合状态。
    bool updateContinuousFusionFromVisualLocked(const FrameRecord& rec,
                                                ContinuousFusionRow& out_row) ;

    void appendContinuousFilterSnapshotLocked() ;

    bool tryRestoreContinuousFilterSnapshotAtOrBeforeLocked(double target_t) ;

// 功能：线程安全地追加一条 continuous fusion CSV 记录。
    void appendContinuousFusionCsvRowThreadSafe(const ContinuousFusionRow& row) ;

    static void writeMaybeCsvValue(std::ostream& os, double value) ;

// 功能：向 JSON 文本写一个数值或 `null`。
    static void writeJsonDoubleOrNull(std::ostream& os, double value) ;

    std::string fixFusionRowToJson(const FixFusionRow& row) const ;

// 功能：把一条 fix 融合结果写成 CSV 行。
    void writeFixFusionCsvRow(std::ostream& os, const FixFusionRow& row) const ;

    void writeGnssPositionCsvRow(std::ostream& os, const FixFusionRow& row) const ;
    void writeImuPositionCsvRow(std::ostream& os, const FixFusionRow& row) const ;

// 功能：追加一条 fix 融合结果到输出文件。
    void appendFixFusionCsvRow(const FixFusionRow& row) ;

    void publishFixFusionRow(const FixFusionRow& row) ;

// 功能：从单帧视觉记录中提取可用于融合的视觉锚点。
    bool makeVisualAnchorFromRecord(const FrameRecord& rec, VisualAnchor& anchor) ;

    bool passesVisualAnchorQualityGate(const VisualAnchor& anchor,
                                       std::string& reject_reason) const ;

// 功能：计算锚点创新门限对应的距离阈值。
    double computeAnchorInnovationGateMeters(double speed_mps,
                                             double anchor_dt_sec) const ;

// 功能：计算新旧锚点软更新时的融合系数。
    double computeAnchorBlendAlpha(const VisualAnchor& anchor) const ;

    double computeAnchorAgeSec(const VisualAnchor& anchor, double ref_t) const ;

// 功能：判断某个视觉锚点在指定时刻是否已经过期。
    bool isAnchorExpiredAtTime(const VisualAnchor& anchor, double ref_t) const ;

    void resetFusionAnchorState(bool& has_active_anchor,
                                VisualAnchor& active_anchor,
                                GeoPoint& fused_pos,
                                cv::Vec3d& fused_vel_enu,
                                double& last_prop_t,
                                double& last_heading_used_rad) const ;

// 功能：在锚点过老时主动使其失效。
    bool expireFusionAnchorIfNeeded(bool& has_active_anchor,
                                    VisualAnchor& active_anchor,
                                    GeoPoint& fused_pos,
                                    cv::Vec3d& fused_vel_enu,
                                    double& last_prop_t,
                                    double& last_heading_used_rad,
                                    double ref_t,
                                    std::string& reason_out) const ;

// 功能：尝试在 fix 时刻接纳一个新的视觉锚点。
    bool tryAdoptAnchorAtFixTimeLocked(const VisualAnchor& candidate,
                                       double fix_t,
                                       double speed_hint_mps,
                                       std::string& reject_reason) ;

// 功能：从视觉锚点出发，通过 IMU 传播预测目标时刻地理位置。
    bool predictGeoFromAnchorLocked(const VisualAnchor& anchor,
                                    double start_t,
                                    const GeoPoint& start_geo,
                                    const cv::Vec3d& start_vel_enu,
                                    double target_t,
                                    double fallback_heading_rad,
                                    GeoPoint& pred_geo,
                                    cv::Vec3d& pred_vel_enu,
                                    ContinuousPropagation& prop,
                                    double& imu_heading_delta_deg) const ;

// 功能：同步在线视觉锚点队列直到给定 fix 时刻。
    void syncOnlineAnchorsUntilLocked(double fix_t) ;

    void buildOnlineFixFusionRowLocked(const FixSample& fix, size_t fix_index, FixFusionRow& row) ;

// 功能：在离线模式下统一写出全部 fix 融合结果。
    void writeFixFusionCsvOffline() ;

    void ensureSpLgWorkerStarted() ;

// 功能：关闭 `sp_lg` worker 进程并清理相关资源。
    void shutdownSpLgWorker() ;

    bool shouldPersistFrameArtifacts() const ;

// 功能：生成某一帧最终输出目录的路径。
    std::filesystem::path frameOutputDir(const std::string& frame_tag) const ;

    std::filesystem::path frameWorkingDir(const std::string& frame_tag) const ;

    std::filesystem::path resultImageDir() const ;

// 功能：把图像中间结果写到磁盘。
    void writeImageArtifact(const std::string& path, const cv::Mat& image) const ;

    void writeResultImageArtifact(const std::string& file_name, const cv::Mat& image) const ;

    void copyResultImageArtifactIfExists(const std::string& src_path,
                                         const std::string& file_name) const ;

    void persistRawFrameIfNeeded(const SampledFrame& frame) const ;

// 功能：处理一条 GNSS fix 消息并更新相关缓存与输出。
    void handleFixMessage(const sensor_msgs::NavSatFix& fix_msg) ;

    void handleImuMessage(const sensor_msgs::Imu& imu_msg) ;

// 功能：处理一条图像消息并决定是否进入视觉处理流程。
    void handleImageMessage(const sensor_msgs::Image& image_msg) ;

    void fixCallback(const sensor_msgs::NavSatFix::ConstPtr& fix_msg) ;

// 功能：ROS IMU 回调，把消息转交到统一处理逻辑。
    void imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg) ;

    void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg) ;

// 功能：在加锁状态下裁剪过旧的 fix 历史缓存。
    void trimFixHistoryLocked(double newest_t) ;

    void startProcessingThread() ;

// 功能：请求后台处理线程停止。
    void requestProcessingStop() ;

    void stopProcessingThread() ;

// 功能：后台处理线程主循环，持续消费待处理帧。
    void processingLoop() ;

    void executePendingFrameAction(PendingFrameAction action,
                                   const SampledFrame& frame,
                                   const PreparedFrameState& prepared,
                                   FrameRecord& record) ;

// 功能：在当前线程内直接清空已准备好的帧处理任务。
    void drainReadyFramesInline(bool flush_pending_on_shutdown) ;

    PendingFrameAction waitForNextFrameAction(SampledFrame& frame,
                                              PreparedFrameState& prepared,
                                              FrameRecord& record) ;

// 功能：在加锁状态下准备队列头部的待处理图像帧。
    PendingFrameAction prepareFrontFrameLocked(SampledFrame& frame,
                                               PreparedFrameState& prepared,
                                               FrameRecord& record) ;

// 功能：确保当前图像尺寸对应的去畸变缓存已经准备好。
    bool ensureUndistortCache(const cv::Size& src_size) ;

    bool undistortFisheyeCached(const cv::Mat& src, cv::Mat& undist, cv::Mat& newK) ;

// 功能：实现 `processFrame` 对应的功能。
    void processFrame(const SampledFrame& frame,
                      const PreparedFrameState& prepared,
                      FrameRecord base_record) ;

// 功能：输出错误并终止当前运行流程。
    void shutdownWithError(const std::string& message) ;

    FrameRecord makeFrameRecordTemplate(const SampledFrame& frame) const ;

// 功能：实现 `computeElapsedMs` 对应的功能。
    static double computeElapsedMs(const std::chrono::steady_clock::time_point& start,
                                   const std::chrono::steady_clock::time_point& end) ;

// 功能：获取当前系统时间对应的 epoch 秒数。
    static double currentSystemEpochSec() ;

    static std::string makeFrameTag(size_t image_index) ;

    static std::string makeImageStampTag(double image_t) ;

// 功能：在加锁状态下查询并填充指定图像时刻附近的 GPS 信息。
    bool tryFillGpsLocked(double image_t, FixSample& gps) const ;

// 功能：基于前 5 秒 GPS 高度均值更新“是否允许开启视觉与融合”状态。
    void updateFusionOpenStateLocked(const FixSample& fix) ;

// 功能：当高度降回阈值以下时，清理视觉/融合运行态，避免重新开启时沿用旧状态。
    void resetFusionRuntimeOnCloseLocked() ;

    bool isFusionOpenLocked() const ;

// 功能：读取最近一次成功融合结果得到的裁图参考位置。
    bool tryGetLatestFusionCropSeedLocked(GeoPoint& geo) const ;

// 功能：读取指定时刻附近最新的融合裁图参考位置。
    bool tryGetFusionCropSeedAtOrBeforeLocked(double target_t, GeoPoint& geo) const ;

// 功能：从最近的连续融合状态出发，利用 IMU 传播预测指定时刻的裁图中心。
    bool tryPredictCropCenterFromFusionImuLocked(double target_t, GeoPoint& geo) const ;

// 功能：向历史缓存追加一条融合裁图参考位置并裁剪过旧记录。
    void appendFusionCropSeedHistoryLocked(double stamp_sec, const GeoPoint& geo) ;

// 功能：用当前帧成功融合后的结果更新下一帧裁图种子。
    void updateLatestFusionCropSeedLocked(const FrameRecord& record) ;

    std::string classifyHeadingFailureReason(bool heading_found,
                                             const global_heading::HeadingSample& heading) const ;

// 功能：补全单帧记录并写出相关结果。
    void finalizeRecord(FrameRecord record, const cv::Mat& report_newK = cv::Mat()) ;

    Options opt_;
    CameraModel cam_;
    CameraExtrinsics ext_;
    FisheyeUndistortCache undistort_cache_;
    GeoRaster map_raster_;
    GeoRaster dem_raster_;
    global_heading::GlobalHeadingEstimator heading_est_;
    OnlineImuRollPitchTracker imu_tracker_;
    SpLgTemplateOptions sp_lg_template_;
    std::string sp_lg_executable_;
    std::unique_ptr<SpLgWorkerClient> sp_lg_worker_;
    double fix_history_keep_sec_ = 30.0;
    ros::Subscriber image_sub_;
    ros::Subscriber fix_sub_;
    ros::Subscriber imu_sub_;
    ros::Publisher fix_fusion_pub_;
    std::string vis_fix_fusion_topic_ = "/splg/fix_fusion";
    std::mutex state_mutex_;
    std::mutex records_mutex_;
    std::mutex fix_fusion_csv_mutex_;
    std::mutex continuous_fusion_csv_mutex_;
    std::condition_variable state_cv_;
    std::thread processing_thread_;
    std::deque<FixSample> fixes_;
    std::vector<FixSample> all_fixes_;
    std::deque<SampledFrame> pending_frames_;
    std::vector<FrameRecord> records_;
    size_t fusion_record_scan_index_ = 0;
    std::deque<VisualAnchor> fusion_pending_anchors_;
    bool fusion_has_active_anchor_ = false;
    VisualAnchor fusion_active_anchor_;
    bool has_latest_fusion_crop_seed_ = false;
    GeoPoint latest_fusion_crop_seed_;
    double latest_fusion_crop_seed_t_ = std::numeric_limits<double>::quiet_NaN();
    std::deque<TimedFusionCropSeed> fusion_crop_seed_history_;
    int fusion_reject_count_this_fix_ = 0;
    std::string fusion_last_anchor_reject_reason_;
    GeoPoint fusion_fused_pos_;
    cv::Vec3d fusion_fused_vel_enu_ = cv::Vec3d(0.0, 0.0, 0.0);
    double fusion_last_prop_t_ = std::numeric_limits<double>::quiet_NaN();
    double fusion_last_heading_used_rad_ = std::numeric_limits<double>::quiet_NaN();
    ContinuousFusionFilterState continuous_filter_;
    std::deque<ContinuousFilterSnapshot> continuous_filter_history_;
    bool continuous_latency_reference_ready_ = false;
    double continuous_latency_reference_stamp_sec_ = std::numeric_limits<double>::quiet_NaN();
    double continuous_latency_reference_wall_sec_ = std::numeric_limits<double>::quiet_NaN();
    double continuous_last_csv_write_stamp_sec_ = std::numeric_limits<double>::quiet_NaN();
    DebugRejectStats reject_stats_;
    size_t fix_count_ = 0;
    size_t imu_count_ = 0;
    size_t image_count_ = 0;
    size_t sampled_count_ = 0;
    size_t max_pending_frames_observed_ = 0;
    bool outputs_prepared_ = false;
    bool fix_fusion_csv_written_ = false;
    bool online_mode_ = true;
    bool processing_frame_active_ = false;
    bool processing_stop_requested_ = false;
    bool fusion_opened_ = false;
    bool fusion_open_initialized_ = false;
    double fusion_open_first_fix_t_ = std::numeric_limits<double>::quiet_NaN();
    double fusion_initial_altitude_sum_ = 0.0;
    size_t fusion_initial_altitude_count_ = 0;
    double fusion_initial_altitude_ = std::numeric_limits<double>::quiet_NaN();
    std::atomic<bool> stopped_{false};
};
