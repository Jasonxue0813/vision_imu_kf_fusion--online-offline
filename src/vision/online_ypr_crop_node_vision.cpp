/**
 * @file online_ypr_crop_node_vision.cpp
 * @brief `OnlineYprCropNode` 的视觉预处理与单帧定位实现。
 *
 * 这里负责图像去畸变、姿态相关旋转、地图裁图、
 * 生成 `sp_lg` 输入并解析单帧视觉定位结果。
 */

#include "splg_fusion/app/online_ypr_crop_node.h"

namespace {

cv::Mat makePreviewableBgr(const cv::Mat& src) {
    if (src.empty()) return cv::Mat();

    cv::Mat normalized;
    if (src.depth() == CV_8U) {
        normalized = src.clone();
    } else {
        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(src.reshape(1), &min_value, &max_value);
        if (!std::isfinite(min_value) || !std::isfinite(max_value) ||
            std::abs(max_value - min_value) < 1e-12) {
            normalized = cv::Mat(src.size(), CV_MAKETYPE(CV_8U, src.channels()), cv::Scalar::all(0));
        } else {
            src.convertTo(normalized,
                          CV_MAKETYPE(CV_8U, src.channels()),
                          255.0 / (max_value - min_value),
                          -min_value * 255.0 / (max_value - min_value));
        }
    }

    cv::Mat bgr;
    if (normalized.channels() == 1) {
        cv::cvtColor(normalized, bgr, cv::COLOR_GRAY2BGR);
    } else if (normalized.channels() == 3) {
        bgr = normalized;
    } else if (normalized.channels() == 4) {
        cv::cvtColor(normalized, bgr, cv::COLOR_BGRA2BGR);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(normalized, channels);
        while (channels.size() < 3) channels.push_back(channels.back());
        cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, bgr);
    }
    return bgr;
}

}  // namespace

// 功能：确保当前图像尺寸对应的去畸变缓存已经准备好。
bool OnlineYprCropNode::ensureUndistortCache(const cv::Size& src_size) {
    if (undistort_cache_.matches(src_size, opt_.undistort_balance, opt_.undistort_fov_scale)) {
        return true;
    }

    if (!buildFisheyeUndistortCache(
            src_size, cam_, opt_.undistort_balance, opt_.undistort_fov_scale, undistort_cache_)) {
        return false;
    }

    ROS_INFO_STREAM("[Undistort] rebuilt cached fisheye remap for size="
                    << src_size.width << "x" << src_size.height
                    << " balance=" << opt_.undistort_balance
                    << " fov_scale=" << opt_.undistort_fov_scale);
    return true;
}

// 功能：使用缓存参数执行鱼眼去畸变。
bool OnlineYprCropNode::undistortFisheyeCached(const cv::Mat& src, cv::Mat& undist, cv::Mat& newK) {
    if (!ensureUndistortCache(src.size())) return false;
    return undistortFisheye(src, undistort_cache_, undist, newK);
}

// 功能：完成单帧图像从预处理到视觉定位再到融合更新的主流程。
void OnlineYprCropNode::processFrame(const SampledFrame& frame,
                  const OnlineYprCropNode::PreparedFrameState& prepared,
                  FrameRecord base_record) {
    const global_heading::HeadingSample& heading = prepared.heading;
    const FixSample& gps = prepared.gps;

    double roll_deg = prepared.imu_raw_roll_deg;
    double pitch_deg = prepared.imu_raw_pitch_deg;
    const double imu_raw_roll_deg = prepared.imu_raw_roll_deg;
    const double imu_raw_pitch_deg = prepared.imu_raw_pitch_deg;
    roll_deg += opt_.mount_roll_deg;
    pitch_deg += opt_.mount_pitch_deg;

    cv::Mat undistorted;
    cv::Mat newK;
    const auto undistort_start_steady = std::chrono::steady_clock::now();
    if (!undistortFisheyeCached(frame.raw, undistorted, newK)) {
        throw std::runtime_error("Fisheye undistort failed.");
    }
    const double undistort_ms =
        computeElapsedMs(undistort_start_steady, std::chrono::steady_clock::now());

    const double yaw_compass_deg = heading.psi_compass * global_heading::RAD2DEG;
    const std::filesystem::path frame_dir = frameWorkingDir(base_record.frame_dir);
    const std::filesystem::path frame_output_dir = frameOutputDir(base_record.frame_dir);
    const bool need_frame_dir = shouldPersistFrameArtifacts() || opt_.enable_sp_lg;
    if (need_frame_dir) {
        ensureDir(frame_dir.string());
    }
    if (shouldPersistFrameArtifacts()) {
        ensureDir(frame_output_dir.string());
    }

    FrameRecord record = base_record;
    const std::string image_stamp_tag = makeImageStampTag(frame.stamp.toSec());
    record.yaw_compass_deg = yaw_compass_deg;
    record.imu_raw_roll_deg = imu_raw_roll_deg;
    record.imu_raw_pitch_deg = imu_raw_pitch_deg;
    record.roll_deg = roll_deg;
    record.pitch_deg = pitch_deg;
    record.speed_mps = heading.speed_mps;
    record.residual_rms_m = heading.residual_rms;
    record.gps = gps;
    record.undistort_ms = undistort_ms;
    if (shouldPersistFrameArtifacts()) {
        record.raw_image_path = (frame_output_dir / "cam0_raw.png").string();
        writeImageArtifact(record.raw_image_path, frame.raw);
        record.undistorted_image_path = (frame_output_dir / "cam0_undistorted.png").string();
        writeImageArtifact(record.undistorted_image_path, undistorted);
    }

    cv::Matx33d image_rotation = cv::Matx33d::eye();
    const auto rotate_start_steady = std::chrono::steady_clock::now();
    const double rotation_deg_ccw =
        -yaw_compass_deg - opt_.image_rotation_offset_deg_clockwise;
    cv::Mat rotated = rotateKeepFullImage(undistorted, rotation_deg_ccw, &image_rotation);
    const cv::Mat rotated_K = rotateCameraMatrix(newK, image_rotation);
    const cv::Matx33d image_rotation_inv = image_rotation.inv();
    record.rotate_ms = computeElapsedMs(rotate_start_steady, std::chrono::steady_clock::now());
    
    if (need_frame_dir) {
        record.rotated_image_path = (frame_dir / "cam0_match_rotated.png").string();
        writeImageArtifact(record.rotated_image_path, rotated);
    } else {
        record.rotated_image_path = "";
    }
    writeResultImageArtifact(image_stamp_tag + "_rotated.png", rotated);

    // 裁图中心：优先使用 1 秒内最新的融合位置；
    // 若上次融合距离当前超过 1 秒，则从最近的连续融合状态出发，
    // 利用 IMU 传播到当前图像时刻；若仍不可用则回退到当前 GPS。
    GeoPoint center_hit{gps.lon, gps.lat, gps.alt};
    bool got_seed = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (has_latest_fusion_crop_seed_ && (frame.stamp.toSec() - latest_fusion_crop_seed_t_ <= 1.0)) {
            if (tryGetLatestFusionCropSeedLocked(center_hit)) {
                got_seed = true;
            }
        }
        if (!got_seed) {
            if (tryPredictCropCenterFromFusionImuLocked(frame.stamp.toSec(), center_hit)) {
                got_seed = true;
            }
        }
    }
    if (!got_seed) {
        center_hit = GeoPoint{gps.lon, gps.lat, gps.alt};
    }
    GeoPoint optical_axis_hit = center_hit;

    // 裁图高度优先使用图像时刻对应的海拔（由 GNSS 插值得到），
    // 若该值不可用，则回退到最近一次读取到的 GNSS 海拔。
    double crop_image_altitude_m = gps.alt;
    if (!std::isfinite(crop_image_altitude_m)) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!all_fixes_.empty() && std::isfinite(all_fixes_.back().alt)) {
            crop_image_altitude_m = all_fixes_.back().alt;
        }
    }

    if (std::abs(opt_.crop_center_offset_forward_m) > 1e-9 ||
        std::abs(opt_.crop_center_offset_right_m) > 1e-9) {
        const double yaw_rad = heading.psi_compass;
        const double delta_e =
            opt_.crop_center_offset_forward_m * std::sin(yaw_rad) +
            opt_.crop_center_offset_right_m * std::cos(yaw_rad);
        const double delta_n =
            opt_.crop_center_offset_forward_m * std::cos(yaw_rad) -
            opt_.crop_center_offset_right_m * std::sin(yaw_rad);
        center_hit = shiftGeoPointByEnu(center_hit, delta_e, delta_n, 0.0);
    }

    CropWindow map_crop;
    CropWindow dem_crop;
    const auto crop_start_steady = std::chrono::steady_clock::now();
    try {
        const std::vector<GeoPoint> crop_box =
            makeCenteredCropBox(center_hit, opt_, rotated.size(), crop_image_altitude_m);
        map_crop = cropFromLonLatBox(map_raster_, crop_box);
        dem_crop = cropFromLonLatBox(dem_raster_, crop_box);

        if (need_frame_dir) {
            record.map_crop_path = (frame_dir / "roi_map.tif").string();
            record.dem_crop_path = (frame_dir / "roi_dem.tif").string();
            map_raster_.cropToFile(map_crop, record.map_crop_path);
            dem_raster_.cropToFile(dem_crop, record.dem_crop_path);
        }
        record.crop_tif_ms = computeElapsedMs(crop_start_steady, std::chrono::steady_clock::now());
    } catch (const std::exception&) {
        ++reject_stats_.crop_failed;
        record.optical_axis_hit = optical_axis_hit;
        record.center_hit = center_hit;
        record.crop_tif_ms = computeElapsedMs(crop_start_steady, std::chrono::steady_clock::now());
        record.failure_reason = "crop_failed";
        finalizeRecord(record, newK);
        return;
    }

    record.optical_axis_hit = optical_axis_hit;
    record.center_hit = center_hit;
    record.map_crop = map_crop;
    record.dem_crop = dem_crop;

    if (opt_.enable_sp_lg) {
        try {
            const std::string rotated_path = record.rotated_image_path;
            const std::string map_crop_path = record.map_crop_path.empty()
                ? (frame_dir / "roi_map.tif").string()
                : record.map_crop_path;
            const std::string dem_crop_path = record.dem_crop_path.empty()
                ? (frame_dir / "roi_dem.tif").string()
                : record.dem_crop_path;

            const SpLgRunResult loc = runSpLgForFrame(
                opt_,
                sp_lg_template_,
                frame_dir,
                map_crop_path,
                dem_crop_path,
                rotated_path,
                rotated_K,
                newK,
                image_rotation_inv,
                sp_lg_worker_.get());
            record.localization_attempted = loc.attempted;
            record.localization_process_ok = loc.process_ok;
            record.localization_success = loc.success;
            record.localization_used_points = loc.used_points;
            record.localization_inlier_points = loc.inlier_points;
            record.sp_lg_map_feature_points = loc.map_feature_points;
            record.sp_lg_aerial_feature_points = loc.aerial_feature_points;
            record.sp_lg_lightglue_match_pairs = loc.lightglue_match_pairs;
            record.sp_lg_homography_used_pairs = loc.homography_used_pairs;
            record.sp_lg_homography_inlier_count = loc.homography_inlier_count;
            record.sp_lg_homography_inlier_rate = loc.lightglue_geom_inlier_rate;
            record.sp_lg_pnp_input_points = loc.pnp_input_points;
            record.sp_lg_pnp_inlier_rate = loc.pnp_inlier_rate;
            record.localization_reproj_error = loc.reproj_error;
            record.sp_lg_terrain_relief_m = loc.terrain_relief_m;
            record.sp_lg_terrain_alt_std_m = loc.terrain_alt_std_m;
            record.sp_lg_terrain_alt_median_m = loc.terrain_alt_median_m;
            record.sp_lg_pnp_f_eff_px = loc.pnp_f_eff_px;
            record.sp_lg_scale_h_m_per_px = loc.scale_h_m_per_px;
            record.sp_lg_point_spread_img_ratio = loc.point_spread_img_ratio;
            record.sp_lg_point_spread_map_xy_m2 = loc.point_spread_map_xy_m2;
            record.sp_lg_depth_spread_m = loc.depth_spread_m;
            record.sp_lg_obliqueness_deg = loc.obliqueness_deg;
            record.sp_lg_dual_geom_valid = loc.dual_geom_valid;
            record.sp_lg_dual_geom_shared_count = loc.dual_geom_shared_count;
            record.sp_lg_dual_geom_h_rmse_px = loc.dual_geom_h_rmse_px;
            record.sp_lg_dual_geom_pnp_rmse_px = loc.dual_geom_pnp_rmse_px;
            record.sp_lg_dual_geom_disagree_rmse_px = loc.dual_geom_disagree_rmse_px;
            record.sp_lg_dual_geom_px_var_u = loc.dual_geom_px_var_u;
            record.sp_lg_dual_geom_px_cov_uv = loc.dual_geom_px_cov_uv;
            record.sp_lg_dual_geom_px_var_v = loc.dual_geom_px_var_v;
            record.sp_lg_dual_geom_local_scale_u_m_per_px = loc.dual_geom_local_scale_u_m_per_px;
            record.sp_lg_dual_geom_local_scale_v_m_per_px = loc.dual_geom_local_scale_v_m_per_px;
            record.sp_lg_dual_geom_local_scale_mean_m_per_px = loc.dual_geom_local_scale_mean_m_per_px;
            record.sp_lg_dual_geom_var_x_m2 = loc.dual_geom_var_x_m2;
            record.sp_lg_dual_geom_cov_xy_m2 = loc.dual_geom_cov_xy_m2;
            record.sp_lg_dual_geom_var_y_m2 = loc.dual_geom_var_y_m2;
            record.localization_lon = loc.camera_lon;
            record.localization_lat = loc.camera_lat;
            record.localization_alt = loc.camera_alt;
            record.sp_lg_match_ms = loc.elapsed_ms;
            record.sp_lg_model_load_ms = loc.model_load_ms;
            record.sp_lg_feature_ms = loc.feature_ms;
            record.sp_lg_matching_ms = loc.matching_ms;
            record.sp_lg_pair_match_ms = loc.pair_match_ms;
            record.sp_lg_homography_ms = loc.homography_ms;
            record.sp_lg_match_vis_ms = loc.match_vis_ms;
            record.sp_lg_geo2d_ms = loc.geo2d_ms;
            record.sp_lg_dem_sample_ms = loc.dem_sample_ms;
            record.sp_lg_pnp_ms = loc.pnp_ms;
            record.localization_config_path = loc.config_path;
            record.localization_report_path = loc.report_path;

            const std::string match_vis_inlier_path =
                (frame_dir / "sp_lg_matches_inliers.png").string();
            const std::string match_vis_raw_path =
                (frame_dir / "sp_lg_matches_raw.png").string();
            const std::string match_vis_compare_path =
                (frame_dir / "sp_lg_homography_compare.png").string();
            if (std::filesystem::exists(match_vis_inlier_path)) {
                copyResultImageArtifactIfExists(match_vis_inlier_path,
                                                image_stamp_tag + "_match_vis.png");
            } else if (std::filesystem::exists(match_vis_raw_path)) {
                copyResultImageArtifactIfExists(match_vis_raw_path,
                                                image_stamp_tag + "_match_vis.png");
            } else if (std::filesystem::exists(match_vis_compare_path)) {
                copyResultImageArtifactIfExists(match_vis_compare_path,
                                                image_stamp_tag + "_match_vis.png");
            }

            if (!loc.process_ok) {
                record.failure_reason = "sp_lg_process_failed";
            } else if (!loc.success) {
                record.failure_reason = "sp_lg_localization_failed";
            }
        } catch (const std::exception& e) {
            record.localization_attempted = true;
            record.failure_reason = "sp_lg_exception: " + std::string(e.what());
            std::cerr << "[sp_lg] frame " << frame.image_index
                      << " failed: " << e.what() << "\n";
        }
        if (!shouldPersistFrameArtifacts()) {
            removeDirectoryQuietly(frame_dir);
        }
    }

    finalizeRecord(record, newK);

}
