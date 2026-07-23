/**
 * @file fusion_types.h
 * @brief 融合流程用到的基础数据结构与配置类型定义。
 *
 * 这里集中定义相机模型、外参、运行选项、采样帧、视觉记录、
 * 融合输出结构等基础类型，是多个模块共享的数据层。
 */

#pragma once

#include "splg_fusion/common.h"

struct CameraModel {
    cv::Mat K;
    cv::Mat D;
    int width = 0;
    int height = 0;
};

struct CameraExtrinsics {
    cv::Matx33d R_ic = cv::Matx33d::eye();
    cv::Vec3d t_ic = cv::Vec3d(0.0, 0.0, 0.0);
    double cam_to_imu_shift_s = 0.0;
};

struct Options {
    std::string input_mode = "auto";
    std::string bag_path = "/home/jasonxue/splg_lo_origin/180m_cropped.bag";
    std::string cam_yaml = "test/cam0.yaml";
    std::string calib_txt = "cyperst.txt";
    std::string map_tif = "date/17.tif";
    std::string dem_tif = "date/18ele.tif";
    std::string out_dir = "result";
    std::string image_topic = "/cam0/image_raw";
    std::string fix_topic = "/fix";
    std::string imu_topic = "/imu0";
    std::string camera_name = "cam0";
    double match_interval_sec = 1.0;
    int sample_every_n_frames = 20;
    int max_output_frames = 0;
    int start_image_index = 0;
    int end_image_index = 0;
    double imu_fusion_kp = 0.8;
    double imu_fusion_ki = 0.05;
    double imu_accel_lpf_alpha = 0.05;
    double imu_accel_min_norm_mps2 = 8.8;
    double imu_accel_max_norm_mps2 = 10.8;
    double imu_fusion_integral_limit_rad_s = 0.2;
    double imu_bias_lpf_alpha = 0.01;
    double imu_bias_gyro_threshold_rad_s = 0.35;
    double imu_bias_residual_threshold_mps2 = 0.50;
    double imu_bias_max_abs_mps2 = 0.80;
    double imu_velocity_damping_per_sec = 0.30;
    double imu_vertical_velocity_damping_per_sec = 0.60;
    double imu_max_speed_mps = 18.0;
    double imu_max_vertical_speed_mps = 6.0;
    double mount_roll_deg = 0.0;
    double mount_pitch_deg = 0.0;
    double image_rotation_offset_deg_clockwise = 0.0;
    double gps_to_imu_x_m = 0.0;
    double gps_to_imu_y_m = 0.0;
    double gps_to_imu_z_m = 0.0;
    double window_sec = 1.5;
    double gnss_rate_hz = 10.0;
    double heading_history_sec = 7200.0;
    double min_speed_mps = 2.0;
    double max_residual_m = 3.0;
    double undistort_balance = 0.0;
    double undistort_fov_scale = 1.0;
    double ray_step_m = 20.0;
    double max_ray_distance_m = 3000.0;
    int binary_refine_iters = 18;
    double crop_center_offset_forward_m = 0.0;
    double crop_center_offset_right_m = 0.0;
    double ground_altitude = 490.0;
    double crop_dfov_deg = 120.1;
    double crop_scale_factor = 1.2;
    double fusion_open_height = 160.0;
    double crop_half_extent_e_m = 50.0;
    double crop_half_extent_n_m = 50.0;
    double crop_fusion_seed_max_age_sec = 1.5;
    bool persist_frame_artifacts = true;
    bool enable_sp_lg = true;
    std::string sp_lg_template_yaml = "config/config.yaml";
    std::string sp_lg_executable;
    std::string localization_csv;
    std::string sp_lg_result_csv;
    std::string fix_fusion_csv;
    std::string imu_position_csv;
    std::string gnss_position_csv;
    std::string sampled_geo_csv;
    std::string high_rate_fusion_csv;
    int fusion_anchor_min_used_points = 180;
    double fusion_anchor_min_inlier_ratio = 0.78;
    double fusion_anchor_max_reproj_error = 4.0;
    double fusion_anchor_innovation_base_m = 12.0;
    double fusion_anchor_innovation_speed_gain = 4.0;
    double fusion_anchor_soft_update_alpha_min = 0.20;
    double fusion_anchor_soft_update_alpha_max = 0.75;
    double fusion_anchor_max_age_sec = 3.0;
    double fusion_imu_only_max_duration_sec = 15.0;
    double kf_process_speed_sigma_mps = 1.5;
    double kf_process_heading_sigma_deg = 8.0;
    double kf_process_xy_rw_sigma_m_sqrt_s = 0.35;
    double kf_process_z_rw_sigma_m_sqrt_s = 0.80;
    double imu_preintegration_accel_sigma_mps2 = 0.8;
    double imu_preintegration_max_gap_sec = 0.05;
    double kf_visual_min_xy_sigma_m = 2.0;
    double kf_visual_min_z_sigma_m = 4.0;
    bool kf_visual_adaptive_enable = true;
    double kf_visual_adaptive_initial_scale = 1.0;
    double kf_visual_adaptive_min_scale = 0.8;
    double kf_visual_adaptive_max_scale = 3.0;
    double kf_visual_adaptive_alpha = 0.08;
    double kf_visual_adaptive_target_nis = 3.0;
    double kf_visual_nis_gate = 16.266;
    bool kf_visual_reject_on_gate = true;
    int image_queue_size = 256;
    int fix_queue_size = 256;
    int imu_queue_size = 2048;
};

struct CliArgs {
    std::string config_yaml = "config/fusion_config.yaml";
    std::vector<std::string> positional_overrides;
};

struct FixSample {
    double t = std::numeric_limits<double>::quiet_NaN();
    double lat = std::numeric_limits<double>::quiet_NaN();
    double lon = std::numeric_limits<double>::quiet_NaN();
    double alt = std::numeric_limits<double>::quiet_NaN();
};

struct ImuSample {
    double t = 0.0;
    cv::Vec3d gyro = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d accel = cv::Vec3d(0.0, 0.0, 0.0);
};

struct SampledFrame {
    size_t image_index = 0;
    ros::Time stamp;
    std::chrono::steady_clock::time_point topic_receive_steady;
    cv::Mat raw;
};

struct GeoPoint {
    double lon = std::numeric_limits<double>::quiet_NaN();
    double lat = std::numeric_limits<double>::quiet_NaN();
    double alt = std::numeric_limits<double>::quiet_NaN();
};

struct CropWindow {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
};

struct FrameRecord {
    size_t image_index = 0;
    double image_t = std::numeric_limits<double>::quiet_NaN();
    double sync_wait_ms = std::numeric_limits<double>::quiet_NaN();
    double undistort_ms = std::numeric_limits<double>::quiet_NaN();
    double rotate_ms = std::numeric_limits<double>::quiet_NaN();
    double crop_tif_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_match_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_model_load_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_feature_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_matching_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_pair_match_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_homography_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_match_vis_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_geo2d_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dem_sample_ms = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_pnp_ms = std::numeric_limits<double>::quiet_NaN();
    double processing_ms = std::numeric_limits<double>::quiet_NaN();
    double topic_to_result_ms = std::numeric_limits<double>::quiet_NaN();
    double imu_query_t = std::numeric_limits<double>::quiet_NaN();
    double yaw_compass_deg = std::numeric_limits<double>::quiet_NaN();
    double imu_raw_roll_deg = std::numeric_limits<double>::quiet_NaN();
    double imu_raw_pitch_deg = std::numeric_limits<double>::quiet_NaN();
    double roll_deg = std::numeric_limits<double>::quiet_NaN();
    double pitch_deg = std::numeric_limits<double>::quiet_NaN();
    double speed_mps = std::numeric_limits<double>::quiet_NaN();
    double residual_rms_m = std::numeric_limits<double>::quiet_NaN();
    FixSample gps;
    GeoPoint optical_axis_hit;
    GeoPoint center_hit;
    CropWindow map_crop;
    CropWindow dem_crop;
    bool localization_attempted = false;
    bool localization_process_ok = false;
    bool localization_success = false;
    int localization_used_points = 0;
    int localization_inlier_points = 0;
    int sp_lg_map_feature_points = 0;
    int sp_lg_aerial_feature_points = 0;
    int sp_lg_lightglue_match_pairs = 0;
    int sp_lg_homography_used_pairs = 0;
    int sp_lg_homography_inlier_count = 0;
    int sp_lg_pnp_input_points = 0;
    double sp_lg_homography_inlier_rate = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_pnp_inlier_rate = std::numeric_limits<double>::quiet_NaN();
    double localization_reproj_error = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_terrain_relief_m = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_terrain_alt_std_m = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_terrain_alt_median_m = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_pnp_f_eff_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_scale_h_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_point_spread_img_ratio = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_point_spread_map_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_depth_spread_m = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_obliqueness_deg = std::numeric_limits<double>::quiet_NaN();
    bool sp_lg_dual_geom_valid = false;
    int sp_lg_dual_geom_shared_count = 0;
    double sp_lg_dual_geom_h_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_pnp_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_disagree_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_px_var_u = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_px_cov_uv = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_px_var_v = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_local_scale_u_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_local_scale_v_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_local_scale_mean_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_var_x_m2 = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_cov_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    double sp_lg_dual_geom_var_y_m2 = std::numeric_limits<double>::quiet_NaN();
    double localization_lon = std::numeric_limits<double>::quiet_NaN();
    double localization_lat = std::numeric_limits<double>::quiet_NaN();
    double localization_alt = std::numeric_limits<double>::quiet_NaN();
    bool fusion_update_applied = false;
    double fused_lon = std::numeric_limits<double>::quiet_NaN();
    double fused_lat = std::numeric_limits<double>::quiet_NaN();
    double fused_alt = std::numeric_limits<double>::quiet_NaN();
    double fused_vel_east_mps = std::numeric_limits<double>::quiet_NaN();
    double fused_vel_north_mps = std::numeric_limits<double>::quiet_NaN();
    double fused_vel_up_mps = std::numeric_limits<double>::quiet_NaN();
    std::string localization_config_path;
    std::string localization_report_path;
    std::string frame_dir;
    std::string raw_image_path;
    std::string undistorted_image_path;
    std::string rotated_image_path;
    std::string map_crop_path;
    std::string map_crop_preview_path;
    std::string crop_visualization_path;
    std::string dem_crop_path;
    std::string frame_report_path;
    std::string failure_reason;
    std::chrono::steady_clock::time_point topic_receive_steady;
    std::chrono::steady_clock::time_point processing_start_steady;
};

struct DebugRejectStats {
    size_t heading_invalid = 0;
    size_t gps_interp_failed = 0;
    size_t imu_query_failed = 0;
    size_t center_ray_upward = 0;
    size_t center_hit_failed = 0;
    size_t crop_failed = 0;
};

struct SpLgTemplateOptions {
    std::string superpoint_onnx = "models/superpoint.onnx";
    std::string lightglue_onnx = "models/lightglue_sim.onnx";
    std::string feature_method = "superpoint";
    int max_features = 4000;
    int fast_threshold = 20;
    bool fast_nonmax = true;
    std::string matcher_method = "lightglue";
    std::string matcher_distance = "auto";
    double matcher_ratio = 0.8;
    bool matcher_cross_check = false;
    int matcher_max_keep = 3000;
    int H = 480;
    int W = 752;
    double mscore_thresh = 0.1;
    double min_kpt_dist = 0.0;
    int max_draw = 200;
    double homography_ransac_reproj = 3.0;
    int homography_max_pairs = 300;
    bool try_cuda = true;
    int device_id = 0;
    double dem_nodata = -9999.0;
};

struct SpLgRunResult {
    bool attempted = false;
    bool process_ok = false;
    bool success = false;
    int used_points = 0;
    int inlier_points = 0;
    int map_feature_points = 0;
    int aerial_feature_points = 0;
    int lightglue_match_pairs = 0;
    int homography_used_pairs = 0;
    int homography_inlier_count = 0;
    int pnp_input_points = 0;
    double lightglue_geom_inlier_rate = std::numeric_limits<double>::quiet_NaN();
    double pnp_inlier_rate = std::numeric_limits<double>::quiet_NaN();
    double reproj_error = std::numeric_limits<double>::quiet_NaN();
    double terrain_relief_m = std::numeric_limits<double>::quiet_NaN();
    double terrain_alt_std_m = std::numeric_limits<double>::quiet_NaN();
    double terrain_alt_median_m = std::numeric_limits<double>::quiet_NaN();
    double pnp_f_eff_px = std::numeric_limits<double>::quiet_NaN();
    double scale_h_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double point_spread_img_ratio = std::numeric_limits<double>::quiet_NaN();
    double point_spread_map_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    double depth_spread_m = std::numeric_limits<double>::quiet_NaN();
    double obliqueness_deg = std::numeric_limits<double>::quiet_NaN();
    bool dual_geom_valid = false;
    int dual_geom_shared_count = 0;
    double dual_geom_h_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_pnp_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_disagree_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_px_var_u = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_px_cov_uv = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_px_var_v = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_local_scale_u_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_local_scale_v_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_local_scale_mean_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_var_x_m2 = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_cov_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    double dual_geom_var_y_m2 = std::numeric_limits<double>::quiet_NaN();
    double camera_lon = std::numeric_limits<double>::quiet_NaN();
    double camera_lat = std::numeric_limits<double>::quiet_NaN();
    double camera_alt = std::numeric_limits<double>::quiet_NaN();
    double elapsed_ms = std::numeric_limits<double>::quiet_NaN();
    double model_load_ms = std::numeric_limits<double>::quiet_NaN();
    double feature_ms = std::numeric_limits<double>::quiet_NaN();
    double matching_ms = std::numeric_limits<double>::quiet_NaN();
    double pair_match_ms = std::numeric_limits<double>::quiet_NaN();
    double homography_ms = std::numeric_limits<double>::quiet_NaN();
    double match_vis_ms = std::numeric_limits<double>::quiet_NaN();
    double geo2d_ms = std::numeric_limits<double>::quiet_NaN();
    double dem_sample_ms = std::numeric_limits<double>::quiet_NaN();
    double pnp_ms = std::numeric_limits<double>::quiet_NaN();
    std::string config_path;
    std::string report_path;
};

struct FisheyeUndistortCache {
    cv::Size src_size;
    double balance = std::numeric_limits<double>::quiet_NaN();
    double fov_scale = std::numeric_limits<double>::quiet_NaN();
    cv::Mat newK;
    cv::Mat map1;
    cv::Mat map2;

    bool matches(const cv::Size& size, double expected_balance, double expected_fov_scale) const {
        return !map1.empty() &&
               !map2.empty() &&
               size == src_size &&
               std::abs(balance - expected_balance) <= 1e-12 &&
               std::abs(fov_scale - expected_fov_scale) <= 1e-12;
    }
};

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog
              << " [--config path.yaml] [cam0_yaml] [calib_txt] [map_tif] [dem_tif] [out_dir]\n"
              << "Runs in online topic mode or offline rosbag mode depending on input.mode.\n"
              << "Defaults:\n"
              << "  config    = config/fusion_config.yaml\n"
              << "  cam0_yaml = test/cam0.yaml\n"
              << "  calib_txt = cyperst.txt\n"
              << "  map_tif   = date/17.tif\n"
              << "  dem_tif   = date/18ele.tif\n"
              << "  out_dir   = result\n";
}

static bool looksLikeYamlPath(const std::string& path) {
    if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".yaml") == 0) return true;
    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".yml") == 0) return true;
    return false;
}

static std::string trimCopy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

static std::string toLowerCopy(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

static std::string normalizeInputMode(const std::string& mode) {
    const std::string lowered = toLowerCopy(trimCopy(mode));
    if (lowered.empty()) return "auto";
    if (lowered == "auto" || lowered == "online" || lowered == "offline") return lowered;
    throw std::runtime_error("Unsupported input.mode: " + mode + " (expected auto/online/offline)");
}

static double safeRatio(int numerator, int denominator) {
    if (denominator <= 0) return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

struct VisualAxisSigmaPrediction {
    double pred_x_var_m2 = std::numeric_limits<double>::quiet_NaN();
    double pred_y_var_m2 = std::numeric_limits<double>::quiet_NaN();
    double pred_z_var_m2 = std::numeric_limits<double>::quiet_NaN();
    double pred_x_sigma_m = std::numeric_limits<double>::quiet_NaN();
    double pred_y_sigma_m = std::numeric_limits<double>::quiet_NaN();
    double pred_z_sigma_m = std::numeric_limits<double>::quiet_NaN();
    double pred_x_quality_multiplier = std::numeric_limits<double>::quiet_NaN();
    double pred_y_quality_multiplier = std::numeric_limits<double>::quiet_NaN();
};

struct ContinuousFusionRow {
    size_t sample_index = 0;
    double stamp_sec = std::numeric_limits<double>::quiet_NaN();
    double lon = std::numeric_limits<double>::quiet_NaN();
    double lat = std::numeric_limits<double>::quiet_NaN();
    double alt = std::numeric_limits<double>::quiet_NaN();
    double east_m = std::numeric_limits<double>::quiet_NaN();
    double north_m = std::numeric_limits<double>::quiet_NaN();
    double up_m = std::numeric_limits<double>::quiet_NaN();
    double cov_xx_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_xz_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_yx_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_yy_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_yz_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_zx_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_zy_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_zz_m2 = std::numeric_limits<double>::quiet_NaN();
    double heading_used_deg = std::numeric_limits<double>::quiet_NaN();
    double speed_mps = std::numeric_limits<double>::quiet_NaN();
    double propagate_dt_sec = std::numeric_limits<double>::quiet_NaN();
    int visual_update_applied = 0;
    double visual_nis = std::numeric_limits<double>::quiet_NaN();
    double visual_cov_scale = std::numeric_limits<double>::quiet_NaN();
    int visual_gate_passed = -1;
    double output_latency_ms = std::numeric_limits<double>::quiet_NaN();
    double output_wall_epoch_sec = std::numeric_limits<double>::quiet_NaN();
    double stamp_to_output_latency_ms = std::numeric_limits<double>::quiet_NaN();
    double replay_relative_latency_ms = std::numeric_limits<double>::quiet_NaN();
    std::string source;
};

static cv::Matx33d makeDiag3(double xx, double yy, double zz) {
    return cv::Matx33d(xx, 0.0, 0.0,
                       0.0, yy, 0.0,
                       0.0, 0.0, zz);
}

static cv::Matx33d scaleMatx33d(const cv::Matx33d& m, double scale) {
    return cv::Matx33d(m(0, 0) * scale, m(0, 1) * scale, m(0, 2) * scale,
                       m(1, 0) * scale, m(1, 1) * scale, m(1, 2) * scale,
                       m(2, 0) * scale, m(2, 1) * scale, m(2, 2) * scale);
}

static bool isFiniteMatx33d(const cv::Matx33d& m) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(m(r, c))) return false;
        }
    }
    return true;
}

static double safeLogFeature(double value, double floor = 1e-6) {
    if (!std::isfinite(value) || value <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(std::max(value, floor));
}

static double safeLog1pFeature(double value) {
    if (!std::isfinite(value) || value < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log1p(value);
}

static double computeClippedMultiplierFromTrend(
    std::initializer_list<std::tuple<double, double, double>> terms,
    double clip_min,
    double clip_max) {
    double log_multiplier = 0.0;
    for (const auto& term : terms) {
        const double value = std::get<0>(term);
        const double ref = std::get<1>(term);
        const double slope = std::get<2>(term);
        if (!std::isfinite(value) || value <= 0.0 || !std::isfinite(ref) || ref <= 0.0 || !std::isfinite(slope)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        log_multiplier += slope * std::log(value / ref);
    }
    const double multiplier = std::exp(log_multiplier);
    if (!std::isfinite(multiplier) || multiplier <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::max(clip_min, std::min(clip_max, multiplier));
}

static VisualAxisSigmaPrediction computeVisualAxisSigmaPrediction(const FrameRecord& rec) {
    VisualAxisSigmaPrediction pred;

    if (std::isfinite(rec.sp_lg_dual_geom_var_x_m2) && rec.sp_lg_dual_geom_var_x_m2 > 0.0) {
        const double x_multiplier = computeClippedMultiplierFromTrend(
            {
                {rec.localization_reproj_error, 2.432221685, 2.408108378560803},
                {rec.sp_lg_pnp_inlier_rate, 0.87037037, -1.4013746091746202},
                {rec.sp_lg_homography_inlier_rate, 0.5821840149999999, -2.6099066443146337},
                {rec.sp_lg_point_spread_img_ratio, 0.548319605, -0.7602740316102015},
                {rec.sp_lg_depth_spread_m, 73.29779595, -4.040753789430454},
                {rec.sp_lg_obliqueness_deg, 15.329001485, -0.7383878156800521},
            },
            0.8,
            4.0);
        if (std::isfinite(x_multiplier)) {
            pred.pred_x_quality_multiplier = x_multiplier;
            pred.pred_x_var_m2 = rec.sp_lg_dual_geom_var_x_m2 * x_multiplier;
            if (std::isfinite(pred.pred_x_var_m2) && pred.pred_x_var_m2 > 0.0) {
                pred.pred_x_sigma_m = std::sqrt(pred.pred_x_var_m2);
            }
        }
    }

    if (std::isfinite(rec.sp_lg_dual_geom_var_y_m2) && rec.sp_lg_dual_geom_var_y_m2 > 0.0) {
        const double y_multiplier = computeClippedMultiplierFromTrend(
            {
                {rec.localization_reproj_error, 2.432221685, 1.204661918117212},
                {rec.sp_lg_pnp_inlier_rate, 0.87037037, -1.6854797117431217},
                {rec.sp_lg_homography_inlier_rate, 0.5821840149999999, -0.6768314253096079},
                {rec.sp_lg_dual_geom_h_rmse_px, 2.296074, 2.704897772110435},
                {rec.sp_lg_obliqueness_deg, 15.329001485, 0.24877347196563604},
            },
            0.4,
            2.5);
        if (std::isfinite(y_multiplier)) {
            pred.pred_y_quality_multiplier = y_multiplier;
            pred.pred_y_var_m2 = rec.sp_lg_dual_geom_var_y_m2 * y_multiplier;
            if (std::isfinite(pred.pred_y_var_m2) && pred.pred_y_var_m2 > 0.0) {
                pred.pred_y_sigma_m = std::sqrt(pred.pred_y_var_m2);
            }
        }
    }

    const double log_reproj = safeLogFeature(rec.localization_reproj_error);
    const double log_terrain_relief = safeLogFeature(rec.sp_lg_terrain_relief_m);
    const double log_pnp_inlier_count = safeLog1pFeature(static_cast<double>(rec.localization_inlier_points));
    const double log_pnp_inlier_rate = safeLogFeature(rec.sp_lg_pnp_inlier_rate);
    if (std::isfinite(log_reproj) &&
        std::isfinite(log_terrain_relief) &&
        std::isfinite(log_pnp_inlier_count) &&
        std::isfinite(log_pnp_inlier_rate)) {
        const double log_var_z =
            -0.7629712389036853 +
            2.59690597871322 * log_reproj +
            -3.660977674357853 * log_terrain_relief +
            1.3849830922402342 * log_pnp_inlier_count +
            -3.22296166062672 * log_pnp_inlier_rate;
        pred.pred_z_var_m2 = std::exp(log_var_z);
        if (std::isfinite(pred.pred_z_var_m2) && pred.pred_z_var_m2 > 0.0) {
            pred.pred_z_sigma_m = std::sqrt(pred.pred_z_var_m2);
        }
    }

    return pred;
}

static std::string formatPredictedVarianceDiag(const VisualAxisSigmaPrediction& pred) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    oss << '['
        << pred.pred_x_var_m2 << ','
        << pred.pred_y_var_m2 << ','
        << pred.pred_z_var_m2 << ']';
    return oss.str();
}

static cv::Matx33d computeVisualMeasurementCovariance(const FrameRecord& rec,
                                                      const Options& opt) {
    const VisualAxisSigmaPrediction pred = computeVisualAxisSigmaPrediction(rec);

    const double min_xy_var = std::max(1e-6, opt.kf_visual_min_xy_sigma_m * opt.kf_visual_min_xy_sigma_m);
    const double min_z_var = std::max(1e-6, opt.kf_visual_min_z_sigma_m * opt.kf_visual_min_z_sigma_m);

    auto chooseVar = [](double preferred, double fallback, double floor_var) {
        if (std::isfinite(preferred) && preferred > 0.0) return std::max(preferred, floor_var);
        if (std::isfinite(fallback) && fallback > 0.0) return std::max(fallback, floor_var);
        return floor_var;
    };

    const double var_x = chooseVar(pred.pred_x_var_m2, rec.sp_lg_dual_geom_var_x_m2, min_xy_var);
    const double var_y = chooseVar(pred.pred_y_var_m2, rec.sp_lg_dual_geom_var_y_m2, min_xy_var);
    const double fallback_var_z =
        std::isfinite(rec.sp_lg_scale_h_m_per_px) && rec.sp_lg_scale_h_m_per_px > 0.0
            ? std::max(min_z_var, std::pow(4.0 * rec.sp_lg_scale_h_m_per_px, 2.0))
            : min_z_var;
    const double var_z = chooseVar(pred.pred_z_var_m2, fallback_var_z, min_z_var);

    cv::Matx33d cov = makeDiag3(var_x, var_y, var_z);
    if (std::isfinite(rec.sp_lg_dual_geom_cov_xy_m2)) {
        cov(0, 1) = rec.sp_lg_dual_geom_cov_xy_m2;
        cov(1, 0) = rec.sp_lg_dual_geom_cov_xy_m2;
    }
    return cov;
}

static bool shouldUseOfflineMode(const Options& opt) {
    if (opt.input_mode == "offline") return true;
    if (opt.input_mode == "online") return false;
    return !trimCopy(opt.bag_path).empty();
}
