/**
 * @file fusion_io_worker.h
 * @brief CSV 输出、worker 进程通信和结果读写辅助函数。
 *
 * 该文件负责单帧运行配置写出、`sp_lg` worker 进程管理、
 * 结果 CSV / 报告解析，以及若干 I/O 相关的辅助工具。
 */

#pragma once

#include "splg_fusion/common.h"
#include "splg_fusion/fusion/fusion_types.h"
#include "splg_fusion/config/fusion_config_camera.h"
#include "splg_fusion/fusion/fusion_geometry.h"

// 功能：写出 `writeFrameReport` 相关结果。
static void writeFrameReport(const std::string& path,
                             const Options& opt,
                             const FrameRecord& record,
                             const cv::Mat& newK) {
    ensureParentDir(path);
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to write report: " + path);
    }

    ofs << std::fixed << std::setprecision(9);
    ofs << "input_mode=" << opt.input_mode << "\n";
    ofs << "bag_path=" << opt.bag_path << "\n";
    ofs << "image_topic=" << opt.image_topic << "\n";
    ofs << "fix_topic=" << opt.fix_topic << "\n";
    ofs << "imu_topic=" << opt.imu_topic << "\n";
    ofs << "image_index=" << record.image_index << "\n";
    ofs << "image_stamp=" << record.image_t << "\n";
    ofs << "sync_wait_ms=" << record.sync_wait_ms << "\n";
    ofs << "undistort_ms=" << record.undistort_ms << "\n";
    ofs << "rotate_ms=" << record.rotate_ms << "\n";
    ofs << "crop_tif_ms=" << record.crop_tif_ms << "\n";
    ofs << "sp_lg_match_ms=" << record.sp_lg_match_ms << "\n";
    ofs << "sp_lg_model_load_ms=" << record.sp_lg_model_load_ms << "\n";
    ofs << "sp_lg_feature_ms=" << record.sp_lg_feature_ms << "\n";
    ofs << "sp_lg_matching_ms=" << record.sp_lg_matching_ms << "\n";
    ofs << "sp_lg_pair_match_ms=" << record.sp_lg_pair_match_ms << "\n";
    ofs << "sp_lg_homography_ms=" << record.sp_lg_homography_ms << "\n";
    ofs << "sp_lg_match_vis_ms=" << record.sp_lg_match_vis_ms << "\n";
    ofs << "sp_lg_geo2d_ms=" << record.sp_lg_geo2d_ms << "\n";
    ofs << "sp_lg_dem_sample_ms=" << record.sp_lg_dem_sample_ms << "\n";
    ofs << "sp_lg_pnp_ms=" << record.sp_lg_pnp_ms << "\n";
    ofs << "processing_ms=" << record.processing_ms << "\n";
    ofs << "topic_to_result_ms=" << record.topic_to_result_ms << "\n";
    ofs << "imu_query_stamp=" << record.imu_query_t << "\n";
    ofs << "yaw_compass_deg=" << record.yaw_compass_deg << "\n";
    ofs << "imu_raw_roll_deg=" << record.imu_raw_roll_deg << "\n";
    ofs << "imu_raw_pitch_deg=" << record.imu_raw_pitch_deg << "\n";
    ofs << "roll_deg=" << record.roll_deg << "\n";
    ofs << "pitch_deg=" << record.pitch_deg << "\n";
    ofs << "speed_mps=" << record.speed_mps << "\n";
    ofs << "residual_rms_m=" << record.residual_rms_m << "\n";
    ofs << "gps_lon=" << record.gps.lon << "\n";
    ofs << "gps_lat=" << record.gps.lat << "\n";
    ofs << "gps_alt=" << record.gps.alt << "\n";
    ofs << "optical_axis_hit_lon=" << record.optical_axis_hit.lon << "\n";
    ofs << "optical_axis_hit_lat=" << record.optical_axis_hit.lat << "\n";
    ofs << "optical_axis_hit_alt=" << record.optical_axis_hit.alt << "\n";
    ofs << "center_hit_lon=" << record.center_hit.lon << "\n";
    ofs << "center_hit_lat=" << record.center_hit.lat << "\n";
    ofs << "center_hit_alt=" << record.center_hit.alt << "\n";
    ofs << "map_crop_pixels=" << record.map_crop.x0 << "," << record.map_crop.y0
        << "," << record.map_crop.x1 << "," << record.map_crop.y1 << "\n";
    ofs << "dem_crop_pixels=" << record.dem_crop.x0 << "," << record.dem_crop.y0
        << "," << record.dem_crop.x1 << "," << record.dem_crop.y1 << "\n";
    ofs << "vision_dispatch_policy=single_frame_serial_drop_while_busy\n";
    ofs << "match_interval_sec_legacy=" << opt.match_interval_sec << "\n";
    ofs << "sample_every_n_frames_legacy=" << opt.sample_every_n_frames << "\n";
    ofs << "max_output_frames=" << opt.max_output_frames << "\n";
    ofs << "start_image_index=" << opt.start_image_index << "\n";
    ofs << "end_image_index=" << opt.end_image_index << "\n";
    ofs << "imu_fusion_kp=" << opt.imu_fusion_kp << "\n";
    ofs << "imu_fusion_ki=" << opt.imu_fusion_ki << "\n";
    ofs << "imu_accel_lpf_alpha=" << opt.imu_accel_lpf_alpha << "\n";
    ofs << "imu_accel_min_norm_mps2=" << opt.imu_accel_min_norm_mps2 << "\n";
    ofs << "imu_accel_max_norm_mps2=" << opt.imu_accel_max_norm_mps2 << "\n";
    ofs << "imu_fusion_integral_limit_rad_s=" << opt.imu_fusion_integral_limit_rad_s << "\n";
    ofs << "mount_roll_deg=" << opt.mount_roll_deg << "\n";
    ofs << "mount_pitch_deg=" << opt.mount_pitch_deg << "\n";
    ofs << "gps_to_imu_x_m=" << opt.gps_to_imu_x_m << "\n";
    ofs << "gps_to_imu_y_m=" << opt.gps_to_imu_y_m << "\n";
    ofs << "gps_to_imu_z_m=" << opt.gps_to_imu_z_m << "\n";
    ofs << "crop_center_offset_forward_m=" << opt.crop_center_offset_forward_m << "\n";
    ofs << "crop_center_offset_right_m=" << opt.crop_center_offset_right_m << "\n";
    ofs << "ground_altitude=" << opt.ground_altitude << "\n";
    ofs << "crop_half_extent_e_m=" << opt.crop_half_extent_e_m << "\n";
    ofs << "crop_half_extent_n_m=" << opt.crop_half_extent_n_m << "\n";
    ofs << "persist_frame_artifacts=" << (opt.persist_frame_artifacts ? 1 : 0) << "\n";
    ofs << "high_rate_fusion_csv=" << opt.high_rate_fusion_csv << "\n";
    ofs << "gnss_position_csv=" << opt.gnss_position_csv << "\n";
    ofs << "kf_process_speed_sigma_mps=" << opt.kf_process_speed_sigma_mps << "\n";
    ofs << "kf_process_heading_sigma_deg=" << opt.kf_process_heading_sigma_deg << "\n";
    ofs << "kf_process_xy_rw_sigma_m_sqrt_s=" << opt.kf_process_xy_rw_sigma_m_sqrt_s << "\n";
    ofs << "kf_process_z_rw_sigma_m_sqrt_s=" << opt.kf_process_z_rw_sigma_m_sqrt_s << "\n";
    ofs << "imu_preintegration_accel_sigma_mps2=" << opt.imu_preintegration_accel_sigma_mps2 << "\n";
    ofs << "imu_preintegration_max_gap_sec=" << opt.imu_preintegration_max_gap_sec << "\n";
    ofs << "kf_visual_min_xy_sigma_m=" << opt.kf_visual_min_xy_sigma_m << "\n";
    ofs << "kf_visual_min_z_sigma_m=" << opt.kf_visual_min_z_sigma_m << "\n";
    ofs << "kf_visual_adaptive_enable=" << (opt.kf_visual_adaptive_enable ? 1 : 0) << "\n";
    ofs << "kf_visual_adaptive_initial_scale=" << opt.kf_visual_adaptive_initial_scale << "\n";
    ofs << "kf_visual_adaptive_min_scale=" << opt.kf_visual_adaptive_min_scale << "\n";
    ofs << "kf_visual_adaptive_max_scale=" << opt.kf_visual_adaptive_max_scale << "\n";
    ofs << "kf_visual_adaptive_alpha=" << opt.kf_visual_adaptive_alpha << "\n";
    ofs << "kf_visual_adaptive_target_nis=" << opt.kf_visual_adaptive_target_nis << "\n";
    ofs << "kf_visual_nis_gate=" << opt.kf_visual_nis_gate << "\n";
    ofs << "kf_visual_reject_on_gate=" << (opt.kf_visual_reject_on_gate ? 1 : 0) << "\n";
    ofs << "localization_attempted=" << (record.localization_attempted ? 1 : 0) << "\n";
    ofs << "localization_process_ok=" << (record.localization_process_ok ? 1 : 0) << "\n";
    ofs << "localization_success=" << (record.localization_success ? 1 : 0) << "\n";
    ofs << "localization_used_points=" << record.localization_used_points << "\n";
    ofs << "localization_inlier_points=" << record.localization_inlier_points << "\n";
    ofs << "sp_lg_map_feature_points=" << record.sp_lg_map_feature_points << "\n";
    ofs << "sp_lg_aerial_feature_points=" << record.sp_lg_aerial_feature_points << "\n";
    ofs << "sp_lg_lightglue_match_pairs=" << record.sp_lg_lightglue_match_pairs << "\n";
    ofs << "sp_lg_homography_used_pairs=" << record.sp_lg_homography_used_pairs << "\n";
    ofs << "sp_lg_homography_inlier_count=" << record.sp_lg_homography_inlier_count << "\n";
    ofs << "sp_lg_homography_inlier_rate=" << record.sp_lg_homography_inlier_rate << "\n";
    ofs << "sp_lg_pnp_input_points=" << record.sp_lg_pnp_input_points << "\n";
    ofs << "sp_lg_pnp_inlier_rate=" << record.sp_lg_pnp_inlier_rate << "\n";
    ofs << "localization_reproj_error=" << record.localization_reproj_error << "\n";
    ofs << "sp_lg_terrain_relief_m=" << record.sp_lg_terrain_relief_m << "\n";
    ofs << "sp_lg_terrain_alt_std_m=" << record.sp_lg_terrain_alt_std_m << "\n";
    ofs << "sp_lg_terrain_alt_median_m=" << record.sp_lg_terrain_alt_median_m << "\n";
    ofs << "sp_lg_pnp_f_eff_px=" << record.sp_lg_pnp_f_eff_px << "\n";
    ofs << "sp_lg_scale_h_m_per_px=" << record.sp_lg_scale_h_m_per_px << "\n";
    ofs << "sp_lg_point_spread_img_ratio=" << record.sp_lg_point_spread_img_ratio << "\n";
    ofs << "sp_lg_point_spread_map_xy_m2=" << record.sp_lg_point_spread_map_xy_m2 << "\n";
    ofs << "sp_lg_depth_spread_m=" << record.sp_lg_depth_spread_m << "\n";
    ofs << "sp_lg_obliqueness_deg=" << record.sp_lg_obliqueness_deg << "\n";
    ofs << "localization_lon=" << record.localization_lon << "\n";
    ofs << "localization_lat=" << record.localization_lat << "\n";
    ofs << "localization_alt=" << record.localization_alt << "\n";
    ofs << "localization_config_path=" << record.localization_config_path << "\n";
    ofs << "localization_report_path=" << record.localization_report_path << "\n";
    ofs << "raw_image_path=" << record.raw_image_path << "\n";
    ofs << "undistorted_image_path=" << record.undistorted_image_path << "\n";
    ofs << "rotated_image_path=" << record.rotated_image_path << "\n";
    ofs << "map_crop_path=" << record.map_crop_path << "\n";
    ofs << "map_crop_preview_path=" << record.map_crop_preview_path << "\n";
    ofs << "crop_visualization_path=" << record.crop_visualization_path << "\n";
    ofs << "dem_crop_path=" << record.dem_crop_path << "\n";
    ofs << "failure_reason=" << record.failure_reason << "\n";
    ofs << "newK=";
    if (!newK.empty()) {
        for (int r = 0; r < newK.rows; ++r) {
            for (int c = 0; c < newK.cols; ++c) {
                if (r != 0 || c != 0) ofs << ",";
                ofs << newK.at<double>(r, c);
            }
        }
    }
    ofs << "\n";
}

// 功能：写出 `writeSummary` 相关结果。
static void writeSummary(const std::string& path,
                         const Options& opt,
                         const std::vector<FrameRecord>& records,
                         size_t fix_count,
                         size_t imu_count,
                         size_t image_count,
                         size_t sampled_count) {
    ensureParentDir(path);
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to write summary: " + path);
    }

    ofs << std::fixed << std::setprecision(6);
    ofs << "input_mode=" << opt.input_mode << "\n";
    ofs << "bag_path=" << opt.bag_path << "\n";
    ofs << "processed_fix_count=" << fix_count << "\n";
    ofs << "processed_imu_count=" << imu_count << "\n";
    ofs << "seen_image_count=" << image_count << "\n";
    ofs << "accepted_image_count=" << sampled_count << "\n";
    ofs << "vision_dispatch_policy=single_frame_serial_drop_while_busy\n";
    ofs << "match_interval_sec_legacy=" << opt.match_interval_sec << "\n";
    ofs << "sample_every_n_frames_legacy=" << opt.sample_every_n_frames << "\n";
    ofs << "start_image_index=" << opt.start_image_index << "\n";
    ofs << "end_image_index=" << opt.end_image_index << "\n";
    ofs << "imu_fusion_kp=" << opt.imu_fusion_kp << "\n";
    ofs << "imu_fusion_ki=" << opt.imu_fusion_ki << "\n";
    ofs << "imu_accel_lpf_alpha=" << opt.imu_accel_lpf_alpha << "\n";
    ofs << "imu_accel_min_norm_mps2=" << opt.imu_accel_min_norm_mps2 << "\n";
    ofs << "imu_accel_max_norm_mps2=" << opt.imu_accel_max_norm_mps2 << "\n";
    ofs << "imu_fusion_integral_limit_rad_s=" << opt.imu_fusion_integral_limit_rad_s << "\n";
    ofs << "mount_roll_deg=" << opt.mount_roll_deg << "\n";
    ofs << "mount_pitch_deg=" << opt.mount_pitch_deg << "\n";
    ofs << "gps_to_imu_x_m=" << opt.gps_to_imu_x_m << "\n";
    ofs << "gps_to_imu_y_m=" << opt.gps_to_imu_y_m << "\n";
    ofs << "gps_to_imu_z_m=" << opt.gps_to_imu_z_m << "\n";
    ofs << "crop_center_offset_forward_m=" << opt.crop_center_offset_forward_m << "\n";
    ofs << "crop_center_offset_right_m=" << opt.crop_center_offset_right_m << "\n";
    ofs << "ground_altitude=" << opt.ground_altitude << "\n";
    ofs << "enable_sp_lg=" << (opt.enable_sp_lg ? 1 : 0) << "\n";
    ofs << "sp_lg_template_yaml=" << opt.sp_lg_template_yaml << "\n";
    ofs << "sp_lg_executable=" << opt.sp_lg_executable << "\n";
    ofs << "persist_frame_artifacts=" << (opt.persist_frame_artifacts ? 1 : 0) << "\n";
    ofs << "localization_csv=" << opt.localization_csv << "\n";
    ofs << "sp_lg_result_csv=" << opt.sp_lg_result_csv << "\n";
    ofs << "fix_fusion_csv=" << opt.fix_fusion_csv << "\n";
    ofs << "imu_position_csv=" << opt.imu_position_csv << "\n";
    ofs << "gnss_position_csv=" << opt.gnss_position_csv << "\n";
    ofs << "sampled_geo_csv=" << opt.sampled_geo_csv << "\n";
    ofs << "high_rate_fusion_csv=" << opt.high_rate_fusion_csv << "\n";
    ofs << "kf_process_speed_sigma_mps=" << opt.kf_process_speed_sigma_mps << "\n";
    ofs << "kf_process_heading_sigma_deg=" << opt.kf_process_heading_sigma_deg << "\n";
    ofs << "kf_process_xy_rw_sigma_m_sqrt_s=" << opt.kf_process_xy_rw_sigma_m_sqrt_s << "\n";
    ofs << "kf_process_z_rw_sigma_m_sqrt_s=" << opt.kf_process_z_rw_sigma_m_sqrt_s << "\n";
    ofs << "imu_preintegration_accel_sigma_mps2=" << opt.imu_preintegration_accel_sigma_mps2 << "\n";
    ofs << "imu_preintegration_max_gap_sec=" << opt.imu_preintegration_max_gap_sec << "\n";
    ofs << "kf_visual_min_xy_sigma_m=" << opt.kf_visual_min_xy_sigma_m << "\n";
    ofs << "kf_visual_min_z_sigma_m=" << opt.kf_visual_min_z_sigma_m << "\n";
    ofs << "kf_visual_adaptive_enable=" << (opt.kf_visual_adaptive_enable ? 1 : 0) << "\n";
    ofs << "kf_visual_adaptive_initial_scale=" << opt.kf_visual_adaptive_initial_scale << "\n";
    ofs << "kf_visual_adaptive_min_scale=" << opt.kf_visual_adaptive_min_scale << "\n";
    ofs << "kf_visual_adaptive_max_scale=" << opt.kf_visual_adaptive_max_scale << "\n";
    ofs << "kf_visual_adaptive_alpha=" << opt.kf_visual_adaptive_alpha << "\n";
    ofs << "kf_visual_adaptive_target_nis=" << opt.kf_visual_adaptive_target_nis << "\n";
    ofs << "kf_visual_nis_gate=" << opt.kf_visual_nis_gate << "\n";
    ofs << "kf_visual_reject_on_gate=" << (opt.kf_visual_reject_on_gate ? 1 : 0) << "\n";
    ofs << "successful_frame_count=" << records.size() << "\n";
    double sum_undistort_ms = 0.0;
    double sum_rotate_ms = 0.0;
    double sum_crop_tif_ms = 0.0;
    double sum_sp_lg_match_ms = 0.0;
    double sum_sp_lg_model_load_ms = 0.0;
    double sum_sp_lg_feature_ms = 0.0;
    double sum_sp_lg_matching_ms = 0.0;
    double sum_sp_lg_pair_match_ms = 0.0;
    double sum_sp_lg_homography_ms = 0.0;
    double sum_sp_lg_match_vis_ms = 0.0;
    double sum_sp_lg_geo2d_ms = 0.0;
    double sum_sp_lg_dem_sample_ms = 0.0;
    double sum_sp_lg_pnp_ms = 0.0;
    size_t count_undistort_ms = 0;
    size_t count_rotate_ms = 0;
    size_t count_crop_tif_ms = 0;
    size_t count_sp_lg_match_ms = 0;
    size_t count_sp_lg_model_load_ms = 0;
    size_t count_sp_lg_feature_ms = 0;
    size_t count_sp_lg_matching_ms = 0;
    size_t count_sp_lg_pair_match_ms = 0;
    size_t count_sp_lg_homography_ms = 0;
    size_t count_sp_lg_match_vis_ms = 0;
    size_t count_sp_lg_geo2d_ms = 0;
    size_t count_sp_lg_dem_sample_ms = 0;
    size_t count_sp_lg_pnp_ms = 0;
    for (const FrameRecord& rec : records) {
        if (std::isfinite(rec.undistort_ms)) {
            sum_undistort_ms += rec.undistort_ms;
            ++count_undistort_ms;
        }
        if (std::isfinite(rec.rotate_ms)) {
            sum_rotate_ms += rec.rotate_ms;
            ++count_rotate_ms;
        }
        if (std::isfinite(rec.crop_tif_ms)) {
            sum_crop_tif_ms += rec.crop_tif_ms;
            ++count_crop_tif_ms;
        }
        if (std::isfinite(rec.sp_lg_match_ms)) {
            sum_sp_lg_match_ms += rec.sp_lg_match_ms;
            ++count_sp_lg_match_ms;
        }
        if (std::isfinite(rec.sp_lg_model_load_ms)) {
            sum_sp_lg_model_load_ms += rec.sp_lg_model_load_ms;
            ++count_sp_lg_model_load_ms;
        }
        if (std::isfinite(rec.sp_lg_feature_ms)) {
            sum_sp_lg_feature_ms += rec.sp_lg_feature_ms;
            ++count_sp_lg_feature_ms;
        }
        if (std::isfinite(rec.sp_lg_matching_ms)) {
            sum_sp_lg_matching_ms += rec.sp_lg_matching_ms;
            ++count_sp_lg_matching_ms;
        }
        if (std::isfinite(rec.sp_lg_pair_match_ms)) {
            sum_sp_lg_pair_match_ms += rec.sp_lg_pair_match_ms;
            ++count_sp_lg_pair_match_ms;
        }
        if (std::isfinite(rec.sp_lg_homography_ms)) {
            sum_sp_lg_homography_ms += rec.sp_lg_homography_ms;
            ++count_sp_lg_homography_ms;
        }
        if (std::isfinite(rec.sp_lg_match_vis_ms)) {
            sum_sp_lg_match_vis_ms += rec.sp_lg_match_vis_ms;
            ++count_sp_lg_match_vis_ms;
        }
        if (std::isfinite(rec.sp_lg_geo2d_ms)) {
            sum_sp_lg_geo2d_ms += rec.sp_lg_geo2d_ms;
            ++count_sp_lg_geo2d_ms;
        }
        if (std::isfinite(rec.sp_lg_dem_sample_ms)) {
            sum_sp_lg_dem_sample_ms += rec.sp_lg_dem_sample_ms;
            ++count_sp_lg_dem_sample_ms;
        }
        if (std::isfinite(rec.sp_lg_pnp_ms)) {
            sum_sp_lg_pnp_ms += rec.sp_lg_pnp_ms;
            ++count_sp_lg_pnp_ms;
        }
        ofs << "frame=" << rec.image_index
            << ",stamp=" << rec.image_t
            << ",yaw=" << rec.yaw_compass_deg
            << ",roll=" << rec.roll_deg
            << ",pitch=" << rec.pitch_deg
            << ",localization_success=" << (rec.localization_success ? 1 : 0)
            << ",camera_lon=" << rec.localization_lon
            << ",camera_lat=" << rec.localization_lat
            << ",camera_alt=" << rec.localization_alt
            << ",dir=" << rec.frame_dir << "\n";
    }
    ofs << "avg_undistort_ms=" << (count_undistort_ms > 0 ? sum_undistort_ms / count_undistort_ms : 0.0) << "\n";
    ofs << "avg_rotate_ms=" << (count_rotate_ms > 0 ? sum_rotate_ms / count_rotate_ms : 0.0) << "\n";
    ofs << "avg_crop_tif_ms=" << (count_crop_tif_ms > 0 ? sum_crop_tif_ms / count_crop_tif_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_match_ms=" << (count_sp_lg_match_ms > 0 ? sum_sp_lg_match_ms / count_sp_lg_match_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_model_load_ms=" << (count_sp_lg_model_load_ms > 0 ? sum_sp_lg_model_load_ms / count_sp_lg_model_load_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_feature_ms=" << (count_sp_lg_feature_ms > 0 ? sum_sp_lg_feature_ms / count_sp_lg_feature_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_matching_ms=" << (count_sp_lg_matching_ms > 0 ? sum_sp_lg_matching_ms / count_sp_lg_matching_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_pair_match_ms=" << (count_sp_lg_pair_match_ms > 0 ? sum_sp_lg_pair_match_ms / count_sp_lg_pair_match_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_homography_ms=" << (count_sp_lg_homography_ms > 0 ? sum_sp_lg_homography_ms / count_sp_lg_homography_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_match_vis_ms=" << (count_sp_lg_match_vis_ms > 0 ? sum_sp_lg_match_vis_ms / count_sp_lg_match_vis_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_geo2d_ms=" << (count_sp_lg_geo2d_ms > 0 ? sum_sp_lg_geo2d_ms / count_sp_lg_geo2d_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_dem_sample_ms=" << (count_sp_lg_dem_sample_ms > 0 ? sum_sp_lg_dem_sample_ms / count_sp_lg_dem_sample_ms : 0.0) << "\n";
    ofs << "avg_sp_lg_pnp_ms=" << (count_sp_lg_pnp_ms > 0 ? sum_sp_lg_pnp_ms / count_sp_lg_pnp_ms : 0.0) << "\n";
}

// 功能：实现 `shellQuote` 对应的功能。
static std::string shellQuote(const std::string& value) {
    std::string out = "'";
    for (char c : value) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

// 功能：实现 `firstNonEmptyLineFromFile` 对应的功能。
static std::string firstNonEmptyLineFromFile(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    std::string line;
    while (std::getline(ifs, line)) {
        if (!trimCopy(line).empty()) return line;
    }
    return "";
}

// 功能：实现 `removeDirectoryQuietly` 对应的功能。
static void removeDirectoryQuietly(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

static std::string findSpLgExecutable(const Options& opt) {
    if (!opt.sp_lg_executable.empty()) return opt.sp_lg_executable;

    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && !self.empty()) {
        const std::filesystem::path candidate = self.parent_path() / "sp_lg";
        if (std::filesystem::exists(candidate)) return candidate.lexically_normal().string();
    }

    const std::filesystem::path cwd_candidate = std::filesystem::current_path() / "build" / "sp_lg";
    if (std::filesystem::exists(cwd_candidate)) return cwd_candidate.lexically_normal().string();

    return "sp_lg";
}

// 功能：实现 `csvEscape` 对应的功能。
static std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;

    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// 功能：实现 `jsonEscape` 对应的功能。
static std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    out.push_back('"');
    for (unsigned char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    std::ostringstream esc;
                    esc << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c);
                    out += esc.str();
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

// 功能：实现 `localizationCsvHeaderLine` 对应的功能。
static const char* localizationCsvHeaderLine() {
    return "matched_feature_points,image_index,image_stamp_sec,sync_wait_ms,sp_lg_match_ms,"
           "camera_lat,camera_lon,camera_alt,failure_reason\n";
}

// 功能：实现 `spLgResultCsvHeaderLine` 对应的功能。
static const char* spLgResultCsvHeaderLine() {
    return "image_index,image_stamp_sec,map_feature_points,aerial_feature_points,lightglue_match_pairs,"
           "homography_used_pairs,homography_inlier_count,lightglue_geom_inlier_rate,"
           "pnp_input_points,localization_used_points,localization_inlier_points,pnp_inlier_rate,"
           "reproj_error,terrain_relief_m,terrain_alt_std_m,terrain_alt_median_m,pnp_f_eff_px,scale_h_m_per_px,"
           "point_spread_img_ratio,point_spread_map_xy_m2,depth_spread_m,obliqueness_deg,"
           "dual_geom_valid,dual_geom_pnp_rmse_px,dual_geom_disagree_rmse_px,"
           "dual_geom_px_var_u,dual_geom_px_cov_uv,dual_geom_px_var_v,"
           "dual_geom_local_scale_u_m_per_px,dual_geom_local_scale_v_m_per_px,dual_geom_local_scale_mean_m_per_px,"
           "dual_geom_var_x_m2,dual_geom_cov_xy_m2,dual_geom_var_y_m2,"
           "pred_x_sigma_m,pred_y_sigma_m,pred_z_sigma_m,"
           "pred_var_diag_m2,"
           "yaw_compass_deg,heading_east_weight,heading_north_weight,"
           "localization_success,failure_reason\n";
}

// 功能：实现 `fixFusionCsvHeaderLine` 对应的功能。
static const char* fixFusionCsvHeaderLine() {
    return "fix_index,fix_stamp_sec,gt_lon,gt_lat,gt_alt,pred_available,pred_lon,pred_lat,pred_alt,"
           "visual_anchor_image_index,visual_anchor_stamp_sec,visual_anchor_lon,visual_anchor_lat,visual_anchor_alt,"
           "visual_anchor_heading_deg,imu_heading_delta_deg,heading_used_deg,speed_mps,propagate_dt_sec,"
           "visual_anchor_age_sec,visual_anchor_sync_wait_ms,visual_anchor_processing_ms,"
           "visual_anchor_topic_to_result_ms,estimated_staleness_ms,"
           "fix_compute_latency_ms,fix_topic_to_fusion_result_ms,"
           "anchor_reject_count,anchor_reject_reason,"
           "heading_source,failure_reason\n";
}

// 功能：实现 `gnssPositionCsvHeaderLine` 对应的功能。
static const char* gnssPositionCsvHeaderLine() {
    return "fix_index,fix_stamp_sec,gt_lon,gt_lat,gt_alt\n";
}

static const char* imuPositionCsvHeaderLine() {
    return "fix_index,fix_stamp_sec,imu_available,imu_lon,imu_lat,imu_alt,visual_anchor_image_index,visual_anchor_stamp_sec\n";
}

// 功能：实现 `sampledGeoCsvHeaderLine` 对应的功能。
static const char* sampledGeoCsvHeaderLine() {
    return "image_index,image_stamp_sec,gps_lon,gps_lat,gps_alt\n";
}

// 功能：实现 `continuousFusionCsvHeaderLine` 对应的功能。
static const char* continuousFusionCsvHeaderLine() {
    return "sample_index,stamp_sec,lon,lat,alt,east_m,north_m,up_m\n";
}

// 功能：确保 `ensureCsvHeaderPresent` 对应的条件已经满足。
static void ensureCsvHeaderPresent(const std::string& path, const char* header_line) {
    ensureParentDir(path);

    std::error_code ec;
    bool should_write_header = !std::filesystem::exists(path, ec);
    if (!should_write_header) {
        const auto size = std::filesystem::file_size(path, ec);
        if (ec || size == 0) {
            should_write_header = true;
        }
    }

    if (!should_write_header) return;

    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create csv with header: " + path);
    }
    ofs << header_line;
}

// 功能：写出 `writeLocalizationCsvHeader` 相关结果。
static void writeLocalizationCsvHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create localization csv: " + path);
    }
    ofs << localizationCsvHeaderLine();
}

// 功能：追加 `appendLocalizationCsvRow` 对应的内容。
static void appendLocalizationCsvRow(const std::string& path, const FrameRecord& rec) {
    ensureCsvHeaderPresent(path, localizationCsvHeaderLine());
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to append localization csv: " + path);
    }
    ofs << std::fixed << std::setprecision(9);
    ofs << rec.sp_lg_lightglue_match_pairs << ','
        << rec.image_index << ','
        << rec.image_t << ','
        << rec.sync_wait_ms << ','
        << rec.sp_lg_match_ms << ','
        << rec.localization_lat << ','
        << rec.localization_lon << ','
        << rec.localization_alt << ','
        << csvEscape(rec.failure_reason) << '\n';
}

static void writeSpLgResultCsvHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create sp_lg result csv: " + path);
    }
    ofs << spLgResultCsvHeaderLine();
}

// 功能：追加 `appendSpLgResultCsvRow` 对应的内容。
static void appendSpLgResultCsvRow(const std::string& path, const FrameRecord& rec) {
    ensureCsvHeaderPresent(path, spLgResultCsvHeaderLine());
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to append sp_lg result csv: " + path);
    }
    const double yaw_rad = std::isfinite(rec.yaw_compass_deg)
        ? rec.yaw_compass_deg * CV_PI / 180.0
        : std::numeric_limits<double>::quiet_NaN();
    const double heading_east_weight = std::isfinite(yaw_rad)
        ? std::sin(yaw_rad) * std::sin(yaw_rad)
        : std::numeric_limits<double>::quiet_NaN();
    const double heading_north_weight = std::isfinite(yaw_rad)
        ? std::cos(yaw_rad) * std::cos(yaw_rad)
        : std::numeric_limits<double>::quiet_NaN();
    const VisualAxisSigmaPrediction sigma_pred = computeVisualAxisSigmaPrediction(rec);
    const std::string pred_var_diag_m2 = formatPredictedVarianceDiag(sigma_pred);
    ofs << std::fixed << std::setprecision(9);
    ofs << rec.image_index << ','
        << rec.image_t << ','
        << rec.sp_lg_map_feature_points << ','
        << rec.sp_lg_aerial_feature_points << ','
        << rec.sp_lg_lightglue_match_pairs << ','
        << rec.sp_lg_homography_used_pairs << ','
        << rec.sp_lg_homography_inlier_count << ','
        << rec.sp_lg_homography_inlier_rate << ','
        << rec.sp_lg_pnp_input_points << ','
        << rec.localization_used_points << ','
        << rec.localization_inlier_points << ','
        << rec.sp_lg_pnp_inlier_rate << ','
        << rec.localization_reproj_error << ','
        << rec.sp_lg_terrain_relief_m << ','
        << rec.sp_lg_terrain_alt_std_m << ','
        << rec.sp_lg_terrain_alt_median_m << ','
        << rec.sp_lg_pnp_f_eff_px << ','
        << rec.sp_lg_scale_h_m_per_px << ','
        << rec.sp_lg_point_spread_img_ratio << ','
        << rec.sp_lg_point_spread_map_xy_m2 << ','
        << rec.sp_lg_depth_spread_m << ','
        << rec.sp_lg_obliqueness_deg << ','
        << (rec.sp_lg_dual_geom_valid ? 1 : 0) << ','
        << rec.sp_lg_dual_geom_pnp_rmse_px << ','
        << rec.sp_lg_dual_geom_disagree_rmse_px << ','
        << rec.sp_lg_dual_geom_px_var_u << ','
        << rec.sp_lg_dual_geom_px_cov_uv << ','
        << rec.sp_lg_dual_geom_px_var_v << ','
        << rec.sp_lg_dual_geom_local_scale_u_m_per_px << ','
        << rec.sp_lg_dual_geom_local_scale_v_m_per_px << ','
        << rec.sp_lg_dual_geom_local_scale_mean_m_per_px << ','
        << rec.sp_lg_dual_geom_var_x_m2 << ','
        << rec.sp_lg_dual_geom_cov_xy_m2 << ','
        << rec.sp_lg_dual_geom_var_y_m2 << ','
        << sigma_pred.pred_x_sigma_m << ','
        << sigma_pred.pred_y_sigma_m << ','
        << sigma_pred.pred_z_sigma_m << ','
        << csvEscape(pred_var_diag_m2) << ','
        << rec.yaw_compass_deg << ','
        << heading_east_weight << ','
        << heading_north_weight << ','
        << (rec.localization_success ? 1 : 0) << ','
        << csvEscape(rec.failure_reason) << '\n';
}

// 功能：写出 `writeFixFusionCsvHeader` 相关结果。
static void writeFixFusionCsvHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create fix fusion csv: " + path);
    }
    ofs << fixFusionCsvHeaderLine();
}

// 功能：写出 `writeGnssPositionCsvHeader` 相关结果。
static void writeImuPositionCsvHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create imu position csv: " + path);
    }
    ofs << imuPositionCsvHeaderLine();
}

static void writeGnssPositionCsvHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create gnss position csv: " + path);
    }
    ofs << gnssPositionCsvHeaderLine();
}

// 功能：写出 `writeSampledGeoCsvHeader` 相关结果。
static void writeSampledGeoCsvHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create sampled geo csv: " + path);
    }
    ofs << sampledGeoCsvHeaderLine();
}

// 功能：追加 `appendSampledGeoCsvRow` 对应的内容。
static void appendSampledGeoCsvRow(const std::string& path, const FrameRecord& rec) {
    ensureCsvHeaderPresent(path, sampledGeoCsvHeaderLine());
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to append sampled geo csv: " + path);
    }
    ofs << std::fixed << std::setprecision(9);
    ofs << rec.image_index << ','
        << rec.image_t << ','
        << rec.gps.lon << ','
        << rec.gps.lat << ','
        << rec.gps.alt << '\n';
}

// 功能：写出 `writeContinuousFusionCsvHeader` 相关结果。
static void writeContinuousFusionCsvHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create continuous fusion csv: " + path);
    }
    ofs << continuousFusionCsvHeaderLine();
}

// 功能：追加 `appendContinuousFusionCsvRow` 对应的内容。
static void appendContinuousFusionCsvRow(const std::string& path, const ContinuousFusionRow& row) {
    ensureCsvHeaderPresent(path, continuousFusionCsvHeaderLine());
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to append continuous fusion csv: " + path);
    }
    ofs << std::fixed << std::setprecision(9);
    ofs << row.sample_index << ','
        << row.stamp_sec << ','
        << row.lon << ','
        << row.lat << ','
        << row.alt << ','
        << row.east_m << ','
        << row.north_m << ','
        << row.up_m << '\n';
}

static void writeProcessedImagesLogHeader(const std::string& path) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to create processed images log: " + path);
    }
    ofs << "image_index,image_stamp_sec,yaw_compass_deg,roll_deg,pitch_deg,localization_success\n";
}

// 功能：追加 `appendProcessedImagesLogRow` 对应的内容。
static void appendProcessedImagesLogRow(const std::string& path, const FrameRecord& rec) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to append processed images log: " + path);
    }
    ofs << std::fixed << std::setprecision(9);
    ofs << rec.image_index << ','
        << rec.image_t << ','
        << rec.yaw_compass_deg << ','
        << rec.roll_deg << ','
        << rec.pitch_deg << ','
        << (rec.localization_success ? 1 : 0) << '\n';
}

static void writeSpLgRunConfig(const std::string& path,
                               const SpLgTemplateOptions& tpl,
                               const std::string& map_tif,
                               const std::string& image_rotated,
                               const std::string& dem_tif,
                               const cv::Mat& rotated_K,
                               const cv::Mat& pnp_K,
                               const cv::Matx33d& image1_inverse_affine,
                               const std::filesystem::path& frame_dir) {
    ensureParentDir(path);
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to write sp_lg run config: " + path);
    }

    const std::string pnp_report = (frame_dir / "sp_lg_pnp_report.txt").string();
    auto yamlQuote = [](const std::string& s) {
        std::string q = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') q.push_back('\\');
            q.push_back(c);
        }
        q.push_back('"');
        return q;
    };
    auto absPath = [](const std::string& s) {
        return std::filesystem::absolute(std::filesystem::path(s)).lexically_normal().string();
    };

    const std::string map_tif_abs = absPath(map_tif);
    const std::string image_rotated_abs = absPath(image_rotated);
    const std::string dem_tif_abs = absPath(dem_tif);
    const std::string superpoint_abs = absPath(tpl.superpoint_onnx);
    const std::string lightglue_abs = absPath(tpl.lightglue_onnx);
    const std::string pnp_report_abs = absPath(pnp_report);
    const std::string sp0_kpts_abs = absPath((frame_dir / "sp_lg_map_keypoints.png").string());
    const std::string sp1_kpts_abs = absPath((frame_dir / "sp_lg_cam_keypoints.png").string());
    const std::string matches_abs = absPath((frame_dir / "sp_lg_matches_raw.png").string());
    const std::string matches_inlier_abs = absPath((frame_dir / "sp_lg_matches_inliers.png").string());
    const std::string warp_abs = absPath((frame_dir / "sp_lg_map_warp.png").string());
    const std::string compare_abs = absPath((frame_dir / "sp_lg_homography_compare.png").string());
    const std::string map2d_csv_abs = absPath((frame_dir / "sp_lg_map2d.csv").string());
    const std::string map3d_csv_abs = absPath((frame_dir / "sp_lg_map3d.csv").string());

    ofs << "%YAML:1.0\n\n";
    ofs << "input:\n";
    ofs << "  image0: " << yamlQuote(map_tif_abs) << "\n";
    ofs << "  image1: " << yamlQuote(image_rotated_abs) << "\n";
    ofs << "  dem_tif: " << yamlQuote(dem_tif_abs) << "\n\n";

    ofs << "model:\n";
    ofs << "  superpoint: " << yamlQuote(superpoint_abs) << "\n";
    ofs << "  lightglue: " << yamlQuote(lightglue_abs) << "\n\n";

    ofs << "feature:\n";
    ofs << "  method: " << tpl.feature_method << "\n";
    ofs << "  max_features: " << tpl.max_features << "\n";
    ofs << "  fast_threshold: " << tpl.fast_threshold << "\n";
    ofs << "  fast_nonmax: " << (tpl.fast_nonmax ? 1 : 0) << "\n\n";

    ofs << "matching:\n";
    ofs << "  method: " << tpl.matcher_method << "\n";
    ofs << "  distance: " << tpl.matcher_distance << "\n";
    ofs << "  ratio: " << tpl.matcher_ratio << "\n";
    ofs << "  cross_check: " << (tpl.matcher_cross_check ? 1 : 0) << "\n";
    ofs << "  max_keep: " << tpl.matcher_max_keep << "\n\n";

    ofs << "output:\n";
    ofs << "  sp0_kpts: " << yamlQuote(sp0_kpts_abs) << "\n";
    ofs << "  sp1_kpts: " << yamlQuote(sp1_kpts_abs) << "\n";
    ofs << "  matches: " << yamlQuote(matches_abs) << "\n";
    ofs << "  matches_inlier: " << yamlQuote(matches_inlier_abs) << "\n";
    ofs << "  warp: " << yamlQuote(warp_abs) << "\n";
    ofs << "  compare: " << yamlQuote(compare_abs) << "\n";
    ofs << "  map2d_csv: " << yamlQuote(map2d_csv_abs) << "\n";
    ofs << "  map3d_csv: " << yamlQuote(map3d_csv_abs) << "\n";
    ofs << "  pnp_report: " << yamlQuote(pnp_report_abs) << "\n\n";

    ofs << "camera:\n";
    ofs << "  use_fisheye_undistort: 0\n";
    ofs << "  undistort_balance: 0.0\n";
    ofs << "  undistort_fov_scale: 1.0\n";
    ofs << "  intrinsics_file: \"\"\n";
    ofs << "  fx: " << rotated_K.at<double>(0, 0) << "\n";
    ofs << "  fy: " << rotated_K.at<double>(1, 1) << "\n";
    ofs << "  cx: " << rotated_K.at<double>(0, 2) << "\n";
    ofs << "  cy: " << rotated_K.at<double>(1, 2) << "\n";
    ofs << "  camera_matrix: ["
        << rotated_K.at<double>(0, 0) << ", " << rotated_K.at<double>(0, 1) << ", " << rotated_K.at<double>(0, 2) << ", "
        << rotated_K.at<double>(1, 0) << ", " << rotated_K.at<double>(1, 1) << ", " << rotated_K.at<double>(1, 2) << ", "
        << rotated_K.at<double>(2, 0) << ", " << rotated_K.at<double>(2, 1) << ", " << rotated_K.at<double>(2, 2) << "]\n";
    ofs << "  pnp_camera_matrix: ["
        << pnp_K.at<double>(0, 0) << ", " << pnp_K.at<double>(0, 1) << ", " << pnp_K.at<double>(0, 2) << ", "
        << pnp_K.at<double>(1, 0) << ", " << pnp_K.at<double>(1, 1) << ", " << pnp_K.at<double>(1, 2) << ", "
        << pnp_K.at<double>(2, 0) << ", " << pnp_K.at<double>(2, 1) << ", " << pnp_K.at<double>(2, 2) << "]\n";
    ofs << "  image1_inverse_affine: ["
        << image1_inverse_affine(0, 0) << ", " << image1_inverse_affine(0, 1) << ", " << image1_inverse_affine(0, 2) << ", "
        << image1_inverse_affine(1, 0) << ", " << image1_inverse_affine(1, 1) << ", " << image1_inverse_affine(1, 2) << ", "
        << image1_inverse_affine(2, 0) << ", " << image1_inverse_affine(2, 1) << ", " << image1_inverse_affine(2, 2) << "]\n";
    ofs << "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n\n";

    ofs << "geo:\n";
    ofs << "  map_affine_valid: 0\n";
    ofs << "  dem_affine_valid: 0\n";
    ofs << "  map_wgs84_affine: [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n";
    ofs << "  dem_wgs84_affine: [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n";
    ofs << "  dem_nodata: " << tpl.dem_nodata << "\n\n";

    ofs << "runtime:\n";
    ofs << "  H: " << tpl.H << "\n";
    ofs << "  W: " << tpl.W << "\n";
    ofs << "  mscore_thresh: " << tpl.mscore_thresh << "\n";
    ofs << "  min_kpt_dist: " << tpl.min_kpt_dist << "\n";
    ofs << "  max_draw: " << tpl.max_draw << "\n";
    ofs << "  homography_ransac_reproj: " << tpl.homography_ransac_reproj << "\n";
    ofs << "  homography_max_pairs: " << tpl.homography_max_pairs << "\n";
    ofs << "  try_cuda: " << (tpl.try_cuda ? 1 : 0) << "\n";
    ofs << "  device_id: " << tpl.device_id << "\n";
}

// 功能：解析 `parseSpLgPnpReport` 相关输入。
static SpLgRunResult parseSpLgPnpReport(const std::string& report_path) {
    SpLgRunResult result;
    result.report_path = report_path;

    std::ifstream ifs(report_path);
    if (!ifs.is_open()) {
        return result;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trimCopy(line.substr(0, eq));
        const std::string value = trimCopy(line.substr(eq + 1));
        if (key == "success") result.success = (std::stoi(value) != 0);
        else if (key == "used_points") result.used_points = std::stoi(value);
        else if (key == "inlier_points") result.inlier_points = std::stoi(value);
        else if (key == "map_feature_points") result.map_feature_points = std::stoi(value);
        else if (key == "aerial_feature_points") result.aerial_feature_points = std::stoi(value);
        else if (key == "lightglue_match_pairs") result.lightglue_match_pairs = std::stoi(value);
        else if (key == "homography_used_pairs") result.homography_used_pairs = std::stoi(value);
        else if (key == "homography_inlier_count") result.homography_inlier_count = std::stoi(value);
        else if (key == "lightglue_geom_inlier_rate") result.lightglue_geom_inlier_rate = std::stod(value);
        else if (key == "pnp_input_points") result.pnp_input_points = std::stoi(value);
        else if (key == "pnp_inlier_rate") result.pnp_inlier_rate = std::stod(value);
        else if (key == "reproj_error") result.reproj_error = std::stod(value);
        else if (key == "terrain_relief_m") result.terrain_relief_m = std::stod(value);
        else if (key == "terrain_alt_std_m") result.terrain_alt_std_m = std::stod(value);
        else if (key == "terrain_alt_median_m") result.terrain_alt_median_m = std::stod(value);
        else if (key == "pnp_f_eff_px") result.pnp_f_eff_px = std::stod(value);
        else if (key == "scale_h_m_per_px") result.scale_h_m_per_px = std::stod(value);
        else if (key == "point_spread_img_ratio") result.point_spread_img_ratio = std::stod(value);
        else if (key == "point_spread_map_xy_m2") result.point_spread_map_xy_m2 = std::stod(value);
        else if (key == "depth_spread_m") result.depth_spread_m = std::stod(value);
        else if (key == "obliqueness_deg") result.obliqueness_deg = std::stod(value);
        else if (key == "dual_geom_valid") result.dual_geom_valid = (std::stoi(value) != 0);
        else if (key == "dual_geom_shared_count") result.dual_geom_shared_count = std::stoi(value);
        else if (key == "dual_geom_h_rmse_px") result.dual_geom_h_rmse_px = std::stod(value);
        else if (key == "dual_geom_pnp_rmse_px") result.dual_geom_pnp_rmse_px = std::stod(value);
        else if (key == "dual_geom_disagree_rmse_px") result.dual_geom_disagree_rmse_px = std::stod(value);
        else if (key == "dual_geom_px_var_u") result.dual_geom_px_var_u = std::stod(value);
        else if (key == "dual_geom_px_cov_uv") result.dual_geom_px_cov_uv = std::stod(value);
        else if (key == "dual_geom_px_var_v") result.dual_geom_px_var_v = std::stod(value);
        else if (key == "dual_geom_local_scale_u_m_per_px") result.dual_geom_local_scale_u_m_per_px = std::stod(value);
        else if (key == "dual_geom_local_scale_v_m_per_px") result.dual_geom_local_scale_v_m_per_px = std::stod(value);
        else if (key == "dual_geom_local_scale_mean_m_per_px") result.dual_geom_local_scale_mean_m_per_px = std::stod(value);
        else if (key == "dual_geom_var_x_m2") result.dual_geom_var_x_m2 = std::stod(value);
        else if (key == "dual_geom_cov_xy_m2") result.dual_geom_cov_xy_m2 = std::stod(value);
        else if (key == "dual_geom_var_y_m2") result.dual_geom_var_y_m2 = std::stod(value);
        else if (key == "camera_lon") result.camera_lon = std::stod(value);
        else if (key == "camera_lat") result.camera_lat = std::stod(value);
        else if (key == "camera_alt") result.camera_alt = std::stod(value);
        else if (key == "model_load_ms") result.model_load_ms = std::stod(value);
        else if (key == "feature_ms") result.feature_ms = std::stod(value);
        else if (key == "matching_ms") result.matching_ms = std::stod(value);
        else if (key == "pair_match_ms") result.pair_match_ms = std::stod(value);
        else if (key == "homography_ms") result.homography_ms = std::stod(value);
        else if (key == "match_vis_ms") result.match_vis_ms = std::stod(value);
        else if (key == "geo2d_ms") result.geo2d_ms = std::stod(value);
        else if (key == "dem_sample_ms") result.dem_sample_ms = std::stod(value);
        else if (key == "pnp_ms") result.pnp_ms = std::stod(value);
    }

    if (!std::isfinite(result.lightglue_geom_inlier_rate)) {
        result.lightglue_geom_inlier_rate = safeRatio(result.homography_inlier_count, result.homography_used_pairs);
    }
    if (!std::isfinite(result.pnp_inlier_rate)) {
        result.pnp_inlier_rate = safeRatio(result.inlier_points, result.pnp_input_points);
    }
    result.process_ok = true;
    return result;
}

class SpLgWorkerClient {
public:
    explicit SpLgWorkerClient(std::string executable)
        : executable_(std::move(executable)) {}

    ~SpLgWorkerClient() {
        stop();
    }

    void start() {
        if (running_) return;

        int stdin_pipe[2]{-1, -1};
        int stdout_pipe[2]{-1, -1};
        if (::pipe(stdin_pipe) != 0) {
            throw std::runtime_error("Failed to create worker stdin pipe.");
        }
        if (::pipe(stdout_pipe) != 0) {
            ::close(stdin_pipe[0]);
            ::close(stdin_pipe[1]);
            throw std::runtime_error("Failed to create worker pipes.");
        }

        const pid_t child_pid = ::fork();
        if (child_pid < 0) {
            ::close(stdin_pipe[0]);
            ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]);
            ::close(stdout_pipe[1]);
            throw std::runtime_error("Failed to fork sp_lg worker process.");
        }

        if (child_pid == 0) {
            ::dup2(stdin_pipe[0], STDIN_FILENO);
            ::dup2(stdout_pipe[1], STDOUT_FILENO);
            ::dup2(stdout_pipe[1], STDERR_FILENO);

            ::close(stdin_pipe[0]);
            ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]);
            ::close(stdout_pipe[1]);

            ::execl(executable_.c_str(), executable_.c_str(), "--worker", static_cast<char*>(nullptr));
            _exit(127);
        }

        ::close(stdin_pipe[0]);
        ::close(stdout_pipe[1]);

        in_fp_ = ::fdopen(stdin_pipe[1], "w");
        out_fp_ = ::fdopen(stdout_pipe[0], "r");
        if (in_fp_ == nullptr || out_fp_ == nullptr) {
            if (in_fp_ != nullptr) ::fclose(in_fp_);
            if (out_fp_ != nullptr) ::fclose(out_fp_);
            ::kill(child_pid, SIGTERM);
            int status = 0;
            ::waitpid(child_pid, &status, 0);
            throw std::runtime_error("Failed to attach stdio for sp_lg worker.");
        }

        pid_ = child_pid;
        running_ = true;

        std::string startup_line;
        if (!readWorkerLine(startup_line) || !startsWith(startup_line, "[SP_LG_WORKER_READY]")) {
            stop();
            throw std::runtime_error("sp_lg worker failed to report ready state.");
        }
    }

    void stop() {
        if (!running_) return;

        if (in_fp_ != nullptr) {
            ::fprintf(in_fp_, "__exit__\n");
            ::fflush(in_fp_);
            ::fclose(in_fp_);
            in_fp_ = nullptr;
        }

        if (out_fp_ != nullptr) {
            ::fclose(out_fp_);
            out_fp_ = nullptr;
        }

        if (pid_ > 0) {
            int status = 0;
            pid_t waited = ::waitpid(pid_, &status, WNOHANG);
            if (waited == 0) {
                ::kill(pid_, SIGTERM);
                ::waitpid(pid_, &status, 0);
            }
            pid_ = -1;
        }

        running_ = false;
    }

    int runConfig(const std::string& config_path,
                  const std::string& stdout_log_path,
                  const std::string& stderr_log_path) {
        if (!running_ || in_fp_ == nullptr || out_fp_ == nullptr) {
            throw std::runtime_error("sp_lg worker is not running.");
        }

        ensureParentDir(stdout_log_path);
        ensureParentDir(stderr_log_path);
        std::ofstream stdout_log(stdout_log_path, std::ios::out | std::ios::trunc);
        std::ofstream stderr_log(stderr_log_path, std::ios::out | std::ios::trunc);

        if (::fprintf(in_fp_, "%s\n", config_path.c_str()) < 0 || ::fflush(in_fp_) != 0) {
            throw std::runtime_error("Failed to send config to sp_lg worker.");
        }

        std::string line;
        while (readWorkerLine(line)) {
            if (startsWith(line, "[Error]")) {
                if (stderr_log.is_open()) stderr_log << line;
            } else {
                if (stdout_log.is_open()) stdout_log << line;
            }

            if (startsWith(line, "[SP_LG_WORKER_DONE]")) {
                const std::size_t rc_pos = line.find("rc=");
                if (rc_pos == std::string::npos) {
                    if (stderr_log.is_open()) stderr_log << "Worker done line missing rc field." << "\n";
                    return -1;
                }

                std::size_t rc_end = rc_pos + 3;
                while (rc_end < line.size() &&
                       (std::isdigit(static_cast<unsigned char>(line[rc_end])) || line[rc_end] == '-')) {
                    ++rc_end;
                }

                try {
                    return std::stoi(line.substr(rc_pos + 3, rc_end - (rc_pos + 3)));
                } catch (...) {
                    if (stderr_log.is_open()) stderr_log << "Failed to parse worker rc from line: " << line << "\n";
                    return -1;
                }
            }
        }

        if (stderr_log.is_open()) {
            stderr_log << "sp_lg worker stream closed unexpectedly." << "\n";
        }
        stop();
        return -1;
    }

private:
    static bool startsWith(const std::string& line, const char* prefix) {
        return line.rfind(prefix, 0) == 0;
    }

    bool readWorkerLine(std::string& out_line) {
        if (out_fp_ == nullptr) return false;
        char line_buf[4096];
        if (::fgets(line_buf, sizeof(line_buf), out_fp_) == nullptr) {
            return false;
        }
        out_line.assign(line_buf);
        return true;
    }

    std::string executable_;
    pid_t pid_ = -1;
    FILE* in_fp_ = nullptr;
    FILE* out_fp_ = nullptr;
    bool running_ = false;
};

// 功能：实现 `runSpLgForFrame` 对应的功能。
static SpLgRunResult runSpLgForFrame(const Options& opt,
                                     const SpLgTemplateOptions& tpl,
                                     const std::filesystem::path& frame_dir,
                                     const std::string& map_tif,
                                     const std::string& dem_tif,
                                     const std::string& rotated_image,
                                     const cv::Mat& rotated_K,
                                     const cv::Mat& pnp_K,
                                     const cv::Matx33d& image1_inverse_affine,
                                     SpLgWorkerClient* worker) {
    SpLgRunResult result;
    result.attempted = true;
    result.config_path = (frame_dir / "sp_lg_run.yaml").string();
    result.report_path = (frame_dir / "sp_lg_pnp_report.txt").string();
    const auto start_steady = std::chrono::steady_clock::now();

    writeSpLgRunConfig(result.config_path, tpl, map_tif, rotated_image, dem_tif,
                       rotated_K, pnp_K, image1_inverse_affine, frame_dir);

    const std::string executable = findSpLgExecutable(opt);
    const std::string stdout_log = (frame_dir / "sp_lg_stdout.log").string();
    const std::string stderr_log = (frame_dir / "sp_lg_stderr.log").string();
    const std::string command =
        shellQuote(executable) + " " + shellQuote(result.config_path) +
        " > " + shellQuote(stdout_log) +
        " 2> " + shellQuote(stderr_log);

    int rc = -1;
    if (worker != nullptr) {
        rc = worker->runConfig(result.config_path, stdout_log, stderr_log);
    } else {
        rc = std::system(command.c_str());
    }
    if (rc != 0) {
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_steady).count();
        std::cerr << "[sp_lg] command failed rc=" << rc
                  << " config=" << result.config_path;
        const std::string err_line = firstNonEmptyLineFromFile(stderr_log);
        if (!err_line.empty()) std::cerr << " err=" << err_line;
        std::cerr << "\n";
        return result;
    }

    result = parseSpLgPnpReport(result.report_path);
    result.attempted = true;
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_steady).count();
    result.config_path = (frame_dir / "sp_lg_run.yaml").string();
    result.report_path = (frame_dir / "sp_lg_pnp_report.txt").string();
    return result;
}
