/**
 * @file online_ypr_crop_node_utils.cpp
 * @brief `OnlineYprCropNode` 的通用辅助函数实现。
 *
 * 这里放的是一些不直接属于视觉或融合主流程的公共辅助逻辑，
 * 例如错误退出、时间统计、帧标签生成、记录收尾等。
 */

#include "splg_fusion/app/online_ypr_crop_node.h"

// 功能：输出错误并终止当前运行流程。
void OnlineYprCropNode::shutdownWithError(const std::string& message) {
    if (stopped_) return;
    stopped_ = true;
    requestProcessingStop();
    ROS_ERROR_STREAM("[Error] " << message);
    if (online_mode_) {
        ros::shutdown();
    }
}

FrameRecord OnlineYprCropNode::makeFrameRecordTemplate(const SampledFrame& frame) const {
    FrameRecord record;
    record.image_index = frame.image_index;
    record.image_t = frame.stamp.toSec();
    record.imu_query_t = record.image_t + ext_.cam_to_imu_shift_s;
    record.frame_dir = makeFrameTag(frame.image_index);
    if (shouldPersistFrameArtifacts()) {
        record.raw_image_path = (frameOutputDir(record.frame_dir) / "cam0_raw.png").string();
        record.frame_report_path = (frameOutputDir(record.frame_dir) / "frame_report.txt").string();
    }
    record.topic_receive_steady = frame.topic_receive_steady;
    return record;
}

// 功能：计算两个稳态时刻之间的毫秒耗时。
double OnlineYprCropNode::computeElapsedMs(const std::chrono::steady_clock::time_point& start,
                               const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 功能：获取当前系统时间对应的 epoch 秒数。
double OnlineYprCropNode::currentSystemEpochSec() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// 功能：根据图像序号生成统一的帧标签字符串。
std::string OnlineYprCropNode::makeFrameTag(size_t image_index) {
    std::ostringstream name_ss;
    name_ss << "frame_" << std::setw(10) << std::setfill('0') << image_index;
    return name_ss.str();
}

std::string OnlineYprCropNode::makeImageStampTag(double image_t) {
    std::ostringstream name_ss;
    name_ss << std::fixed << std::setprecision(9) << image_t;
    return name_ss.str();
}

// 功能：在加锁状态下查询并填充指定图像时刻附近的 GPS 信息。
bool OnlineYprCropNode::tryFillGpsLocked(double image_t, FixSample& gps) const {
    if (fixes_.empty()) return false;
    if (image_t < fixes_.front().t) return false;
    if (fixes_.back().t < image_t) return false;
    return interpolateFixSeries(fixes_, image_t, gps);
}

void OnlineYprCropNode::resetFusionRuntimeOnCloseLocked() {
    fusion_pending_anchors_.clear();
    fusion_has_active_anchor_ = false;
    fusion_active_anchor_ = VisualAnchor{};
    fusion_reject_count_this_fix_ = 0;
    fusion_last_anchor_reject_reason_.clear();
    fusion_fused_pos_ = GeoPoint{};
    fusion_fused_vel_enu_ = cv::Vec3d(0.0, 0.0, 0.0);
    fusion_last_prop_t_ = std::numeric_limits<double>::quiet_NaN();
    fusion_last_heading_used_rad_ = std::numeric_limits<double>::quiet_NaN();
    has_latest_fusion_crop_seed_ = false;
    latest_fusion_crop_seed_ = GeoPoint{};
    latest_fusion_crop_seed_t_ = std::numeric_limits<double>::quiet_NaN();
    fusion_crop_seed_history_.clear();
    continuous_filter_.initialized = false;
    continuous_filter_.state_t = std::numeric_limits<double>::quiet_NaN();
    continuous_filter_.state_enu = cv::Vec3d(0.0, 0.0, 0.0);
    continuous_filter_.state_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
    continuous_filter_.cov_enu = makeDiag3(1e6, 1e6, 1e6);
    continuous_filter_.last_heading_used_rad = std::numeric_limits<double>::quiet_NaN();
    continuous_filter_.last_speed_mps = 0.0;
    continuous_filter_.anchor_heading_compass_rad = std::numeric_limits<double>::quiet_NaN();
    continuous_filter_.anchor_imu_heading_rel_rad = std::numeric_limits<double>::quiet_NaN();
    continuous_filter_.anchor_imu_valid = false;
    continuous_filter_.last_visual_nis = std::numeric_limits<double>::quiet_NaN();
    continuous_filter_.last_visual_update_t = std::numeric_limits<double>::quiet_NaN();
    continuous_filter_.visual_update_count = 0;
    continuous_filter_.visual_reject_count = 0;
    continuous_filter_history_.clear();
}

void OnlineYprCropNode::updateFusionOpenStateLocked(const FixSample& fix) {
    if (!std::isfinite(fix.t) || !std::isfinite(fix.alt)) return;

    if (!fusion_open_initialized_) {
        fusion_open_initialized_ = true;
        fusion_open_first_fix_t_ = fix.t;
    }

    if (!std::isfinite(fusion_open_first_fix_t_)) return;

    const double elapsed_sec = fix.t - fusion_open_first_fix_t_;
    if (elapsed_sec <= 5.0 + 1e-6) {
        fusion_initial_altitude_sum_ += fix.alt;
        ++fusion_initial_altitude_count_;
        return;
    }

    if (!std::isfinite(fusion_initial_altitude_)) {
        if (fusion_initial_altitude_count_ == 0) return;
        fusion_initial_altitude_ =
            fusion_initial_altitude_sum_ / static_cast<double>(fusion_initial_altitude_count_);
        ROS_INFO_STREAM("[FusionOpen] initial_altitude=" << fusion_initial_altitude_
                        << "m from first 5s GPS altitude average"
                        << " sample_count=" << fusion_initial_altitude_count_
                        << " threshold=" << opt_.fusion_open_height << "m");
    }

    const double delta_alt = fix.alt - fusion_initial_altitude_;
    const bool should_open = delta_alt > opt_.fusion_open_height;

    if (!fusion_opened_ && should_open) {
        fusion_opened_ = true;
        ROS_INFO_STREAM("[FusionOpen] enabled at fix_t=" << fix.t
                        << " current_alt=" << fix.alt
                        << " initial_altitude=" << fusion_initial_altitude_
                        << " delta_alt=" << delta_alt
                        << " threshold=" << opt_.fusion_open_height);
        return;
    }

    if (fusion_opened_ && !should_open) {
        fusion_opened_ = false;
        resetFusionRuntimeOnCloseLocked();
        ROS_INFO_STREAM("[FusionOpen] disabled at fix_t=" << fix.t
                        << " current_alt=" << fix.alt
                        << " initial_altitude=" << fusion_initial_altitude_
                        << " delta_alt=" << delta_alt
                        << " threshold=" << opt_.fusion_open_height);
    }
}

bool OnlineYprCropNode::isFusionOpenLocked() const {
    return fusion_opened_;
}

// 功能：读取最近一次成功融合结果得到的裁图参考位置。
bool OnlineYprCropNode::tryGetLatestFusionCropSeedLocked(GeoPoint& geo) const {
    if (!has_latest_fusion_crop_seed_) return false;
    if (!std::isfinite(latest_fusion_crop_seed_.lon) ||
        !std::isfinite(latest_fusion_crop_seed_.lat) ||
        !std::isfinite(latest_fusion_crop_seed_.alt)) {
        return false;
    }
    geo = latest_fusion_crop_seed_;
    return true;
}

// 功能：读取指定时刻附近最新的融合裁图参考位置。
bool OnlineYprCropNode::tryGetFusionCropSeedAtOrBeforeLocked(double target_t, GeoPoint& geo) const {
    if (!std::isfinite(target_t) || fusion_crop_seed_history_.empty()) return false;

    for (auto it = fusion_crop_seed_history_.rbegin(); it != fusion_crop_seed_history_.rend(); ++it) {
        if (!std::isfinite(it->stamp_sec)) continue;
        if (it->stamp_sec <= target_t + 1e-6 &&
            std::isfinite(it->geo.lon) &&
            std::isfinite(it->geo.lat) &&
            std::isfinite(it->geo.alt)) {
            geo = it->geo;
            return true;
        }
    }

    const TimedFusionCropSeed& oldest = fusion_crop_seed_history_.front();
    if (!std::isfinite(oldest.geo.lon) ||
        !std::isfinite(oldest.geo.lat) ||
        !std::isfinite(oldest.geo.alt)) {
        return false;
    }
    geo = oldest.geo;
    return true;
}

bool OnlineYprCropNode::tryPredictCropCenterFromFusionImuLocked(double target_t, GeoPoint& geo) const {
    if (!std::isfinite(target_t) ||
        !continuous_filter_.reference_ready ||
        !continuous_filter_.initialized) {
        return false;
    }

    ContinuousFilterSnapshot base_snapshot;
    bool has_base_snapshot = false;
    for (auto it = continuous_filter_history_.rbegin(); it != continuous_filter_history_.rend(); ++it) {
        if (!std::isfinite(it->state_t) || it->state_t > target_t + 1e-6) continue;
        base_snapshot = *it;
        has_base_snapshot = true;
        break;
    }

    if (!has_base_snapshot) {
        if (!std::isfinite(continuous_filter_.state_t) ||
            continuous_filter_.state_t > target_t + 1e-6) {
            return false;
        }
        base_snapshot.state_t = continuous_filter_.state_t;
        base_snapshot.state_enu = continuous_filter_.state_enu;
        base_snapshot.state_vel_enu = continuous_filter_.state_vel_enu;
        base_snapshot.cov_enu = continuous_filter_.cov_enu;
        base_snapshot.last_heading_used_rad = continuous_filter_.last_heading_used_rad;
        base_snapshot.last_speed_mps = continuous_filter_.last_speed_mps;
        base_snapshot.anchor_heading_compass_rad = continuous_filter_.anchor_heading_compass_rad;
        base_snapshot.anchor_imu_heading_rel_rad = continuous_filter_.anchor_imu_heading_rel_rad;
        base_snapshot.anchor_imu_valid = continuous_filter_.anchor_imu_valid;
        base_snapshot.last_visual_update_t = continuous_filter_.last_visual_update_t;
        base_snapshot.visual_cov_scale = continuous_filter_.visual_cov_scale;
    }

    cv::Vec3d predicted_enu = base_snapshot.state_enu;
    if (!isFiniteVec(predicted_enu)) return false;

    if (target_t > base_snapshot.state_t + 1e-9) {
        const ContinuousPropagation prop = buildContinuousPropagationLocked(
            base_snapshot.state_t,
            target_t,
            base_snapshot.anchor_heading_compass_rad,
            base_snapshot.anchor_imu_heading_rel_rad,
            base_snapshot.anchor_imu_valid,
            base_snapshot.state_vel_enu,
            base_snapshot.last_heading_used_rad,
            base_snapshot.last_speed_mps);
        if (!prop.valid || !isFiniteVec(prop.delta_enu)) return false;
        predicted_enu += prop.delta_enu;
    }

    geo = referenceEnuToGeoLocked(predicted_enu);
    return isFiniteGeoPoint(geo);
}

// 功能：向历史缓存追加一条融合裁图参考位置并裁剪过旧记录。
void OnlineYprCropNode::appendFusionCropSeedHistoryLocked(double stamp_sec, const GeoPoint& geo) {
    if (!std::isfinite(stamp_sec) ||
        !std::isfinite(geo.lon) ||
        !std::isfinite(geo.lat) ||
        !std::isfinite(geo.alt)) {
        return;
    }

    if (!fusion_crop_seed_history_.empty() &&
        stamp_sec + 1e-6 < fusion_crop_seed_history_.back().stamp_sec) {
        fusion_crop_seed_history_.clear();
    }

    if (fusion_crop_seed_history_.empty() ||
        std::abs(stamp_sec - fusion_crop_seed_history_.back().stamp_sec) > 1e-6) {
        fusion_crop_seed_history_.push_back(TimedFusionCropSeed{stamp_sec, geo});
    } else {
        fusion_crop_seed_history_.back().geo = geo;
    }

    const double history_keep_sec =
        std::max(30.0, std::max(opt_.crop_fusion_seed_max_age_sec, fix_history_keep_sec_) + 5.0);
    while (fusion_crop_seed_history_.size() > 2 &&
           stamp_sec - fusion_crop_seed_history_[1].stamp_sec > history_keep_sec) {
        fusion_crop_seed_history_.pop_front();
    }
}

// 功能：用当前帧成功融合后的结果更新下一帧裁图种子。
void OnlineYprCropNode::updateLatestFusionCropSeedLocked(const FrameRecord& record) {
    if (!record.fusion_update_applied) return;
    if (!std::isfinite(record.fused_lon) ||
        !std::isfinite(record.fused_lat) ||
        !std::isfinite(record.fused_alt) ||
        !std::isfinite(record.image_t)) {
        return;
    }
    latest_fusion_crop_seed_ =
        GeoPoint{record.fused_lon, record.fused_lat, record.fused_alt};
    has_latest_fusion_crop_seed_ = true;
    latest_fusion_crop_seed_t_ = record.image_t;
}

// 功能：根据 heading 查询失败情况给出更具体的失败原因。
std::string OnlineYprCropNode::classifyHeadingFailureReason(bool heading_found,
                                         const global_heading::HeadingSample& heading) const {
    if (!heading_found) return "heading_unavailable";

    const bool low_speed =
        std::isfinite(heading.speed_mps) && heading.speed_mps < opt_.min_speed_mps;
    const bool high_residual =
        std::isfinite(heading.residual_rms) && heading.residual_rms > opt_.max_residual_m;

    if (low_speed && high_residual) {
        return "heading_invalid_low_speed_and_high_residual";
    }
    if (low_speed) return "heading_invalid_low_speed";
    if (high_residual) return "heading_invalid_high_residual";
    return "heading_invalid";
}

// 功能：补全单帧记录并写出相关结果。
void OnlineYprCropNode::finalizeRecord(FrameRecord record, const cv::Mat& report_newK) {
    const auto finish_steady = std::chrono::steady_clock::now();
    if (record.processing_start_steady.time_since_epoch().count() == 0) {
        record.processing_start_steady = finish_steady;
    }
    if (!std::isfinite(record.sync_wait_ms)) {
        record.sync_wait_ms = computeElapsedMs(record.topic_receive_steady, record.processing_start_steady);
    }
    record.processing_ms = computeElapsedMs(record.processing_start_steady, finish_steady);
    record.topic_to_result_ms = computeElapsedMs(record.topic_receive_steady, finish_steady);

    bool should_append_continuous_row = false;
    ContinuousFusionRow continuous_row;
    if (record.localization_success) {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        should_append_continuous_row =
            updateContinuousFusionFromVisualLocked(record, continuous_row);
        if (should_append_continuous_row && continuous_row.visual_update_applied == 1) {
            record.fusion_update_applied = true;
            record.fused_lon = continuous_row.lon;
            record.fused_lat = continuous_row.lat;
            record.fused_alt = continuous_row.alt;
            record.fused_vel_east_mps = continuous_filter_.state_vel_enu[0];
            record.fused_vel_north_mps = continuous_filter_.state_vel_enu[1];
            record.fused_vel_up_mps = continuous_filter_.state_vel_enu[2];
            updateLatestFusionCropSeedLocked(record);
        }
    }

    if (!record.frame_report_path.empty()) {
        writeFrameReport(record.frame_report_path, opt_, record, report_newK);
    }
    size_t records_size = 0;
    {
        std::lock_guard<std::mutex> records_lock(records_mutex_);
        records_.push_back(record);
        records_size = records_.size();
    }

    appendSampledGeoCsvRow(opt_.sampled_geo_csv, record);
    if (opt_.enable_sp_lg) {
        appendLocalizationCsvRow(opt_.localization_csv, record);
        appendSpLgResultCsvRow(opt_.sp_lg_result_csv, record);
    }
    if (should_append_continuous_row) {
        continuous_row.output_latency_ms =
            computeElapsedMs(record.topic_receive_steady, std::chrono::steady_clock::now());
        appendContinuousFusionCsvRowThreadSafe(continuous_row);
    }

    if (records_size <= 8 || (records_size % 100) == 0) {
        ROS_INFO_STREAM(std::fixed << std::setprecision(6)
                        << "[Frame] image_index=" << record.image_index
                        << " stamp=" << record.image_t
                        << " yaw=" << record.yaw_compass_deg
                        << " roll=" << record.roll_deg
                        << " pitch=" << record.pitch_deg
                        << " failure_reason="
                        << (record.failure_reason.empty() ? "none" : record.failure_reason)
                        << " loc_success=" << (record.localization_success ? 1 : 0)
                        << " sync_wait_ms=" << record.sync_wait_ms
                        << " undistort_ms=" << record.undistort_ms
                        << " rotate_ms=" << record.rotate_ms
                        << " crop_tif_ms=" << record.crop_tif_ms
                        << " sp_lg_match_ms=" << record.sp_lg_match_ms
                        << " processing_ms=" << record.processing_ms
                        << " topic_to_result_ms=" << record.topic_to_result_ms
                        << " gps=(" << record.gps.lon << "," << record.gps.lat << "," << record.gps.alt << ")");
    }

    if (opt_.max_output_frames > 0 &&
        records_size >= static_cast<size_t>(opt_.max_output_frames)) {
        ROS_INFO_STREAM("[Done] reached max_output_frames=" << opt_.max_output_frames);
        stopped_ = true;
        requestProcessingStop();
        if (online_mode_) {
            ros::shutdown();
        }
    }
}
