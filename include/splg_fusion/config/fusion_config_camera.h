/**
 * @file fusion_config_camera.h
 * @brief 配置解析、相机参数加载与图像几何辅助函数。
 *
 * 该文件负责命令行参数解析、yaml 配置读取、相机内外参加载，
 * 以及若干和路径解析、图像尺寸、外参文本解析相关的辅助函数。
 */

#pragma once

#include "splg_fusion/common.h"
#include "splg_fusion/fusion/fusion_types.h"

// 功能：解析命令行参数并生成主程序配置入口参数。
static CliArgs parseArgs(const std::vector<std::string>& argv) {
    CliArgs cli;
    int argi = 1;
    const int argc = static_cast<int>(argv.size());
    if (argc >= 3 && argv[1] == "--config") {
        cli.config_yaml = argv[2];
        argi = 3;
    } else if (argc >= 2 && looksLikeYamlPath(argv[1])) {
        cli.config_yaml = argv[1];
        argi = 2;
    }

    if (argc - argi > 5) {
        printUsage(argv.empty() ? "visual_imu_fusion" : argv[0].c_str());
        throw std::runtime_error("Too many arguments.");
    }

    for (int i = argi; i < argc; ++i) cli.positional_overrides.push_back(argv[i]);
    return cli;
}

// 功能：移除 ROS remap 语法参数，保留程序自身参数。
static std::vector<std::string> stripRosRemapArgs(int argc, char** argv) {
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(argc));
    if (argc > 0) out.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg.find(":=") != std::string::npos) continue;
        out.push_back(arg);
    }
    return out;
}

// 功能：确保目录存在，不存在时自动创建。
static void ensureDir(const std::string& dir_path) {
    if (!dir_path.empty()) std::filesystem::create_directories(dir_path);
}

// 功能：确保某个文件路径对应的父目录存在。
static void ensureParentDir(const std::string& file_path) {
    const std::filesystem::path p(file_path);
    const std::filesystem::path parent = p.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

// 功能：把配置文件中的相对路径解析成基于配置文件位置的实际路径。
static std::string resolvePathFromConfig(const std::filesystem::path& config_path,
                                         const std::string& value) {
    const std::filesystem::path p(value);
    if (p.empty() || p.is_absolute()) return value;
    const std::filesystem::path base = config_path.parent_path();
    if (base.empty()) return value;
    return (base / p).lexically_normal().string();
}

// 功能：在配置节点存在时读取字符串字段。
static void readStringIfPresent(const cv::FileNode& node,
                                const char* key,
                                std::string& value) {
    if (!node.empty() && !node[key].empty()) value = static_cast<std::string>(node[key]);
}

// 功能：在配置节点存在时读取浮点数字段。
static void readDoubleIfPresent(const cv::FileNode& node,
                                const char* key,
                                double& value) {
    if (!node.empty() && !node[key].empty()) value = static_cast<double>(node[key]);
}

// 功能：在配置节点存在时读取整数字段。
static void readIntIfPresent(const cv::FileNode& node,
                             const char* key,
                             int& value) {
    if (!node.empty() && !node[key].empty()) value = static_cast<int>(node[key]);
}

// 功能：在配置节点存在时读取布尔字段。
static void readBoolIfPresent(const cv::FileNode& node,
                              const char* key,
                              bool& value) {
    if (!node.empty() && !node[key].empty()) value = static_cast<int>(node[key]) != 0;
}

// 功能：从 YAML 文件读取主程序运行配置。
static void loadOptionsYaml(const std::string& config_yaml, Options& opt) {
    const std::filesystem::path config_path(config_yaml);
    if (!std::filesystem::exists(config_path)) {
        std::cout << "[Config] yaml not found, using built-in defaults: " << config_yaml << "\n";
        return;
    }

    cv::FileStorage fs(config_yaml, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open config yaml: " + config_yaml);
    }

    const cv::FileNode input = fs["input"];
    const cv::FileNode topics = fs["topics"];
    const cv::FileNode tuning = fs["tuning"];

    readStringIfPresent(input, "mode", opt.input_mode);
    readStringIfPresent(input, "bag_path", opt.bag_path);
    readStringIfPresent(input, "cam_yaml", opt.cam_yaml);
    readStringIfPresent(input, "calib_txt", opt.calib_txt);
    readStringIfPresent(input, "map_tif", opt.map_tif);
    readStringIfPresent(input, "dem_tif", opt.dem_tif);
    readStringIfPresent(input, "out_dir", opt.out_dir);

    readStringIfPresent(topics, "image_topic", opt.image_topic);
    readStringIfPresent(topics, "fix_topic", opt.fix_topic);
    readStringIfPresent(topics, "imu_topic", opt.imu_topic);
    readStringIfPresent(topics, "camera_name", opt.camera_name);

    readDoubleIfPresent(tuning, "match_interval_sec", opt.match_interval_sec);
    readIntIfPresent(tuning, "sample_every_n_frames", opt.sample_every_n_frames);
    readIntIfPresent(tuning, "max_output_frames", opt.max_output_frames);
    readIntIfPresent(tuning, "start_image_index", opt.start_image_index);
    readIntIfPresent(tuning, "end_image_index", opt.end_image_index);
    readDoubleIfPresent(tuning, "imu_fusion_kp", opt.imu_fusion_kp);
    readDoubleIfPresent(tuning, "imu_fusion_ki", opt.imu_fusion_ki);
    readDoubleIfPresent(tuning, "imu_accel_lpf_alpha", opt.imu_accel_lpf_alpha);
    readDoubleIfPresent(tuning, "imu_accel_min_norm_mps2", opt.imu_accel_min_norm_mps2);
    readDoubleIfPresent(tuning, "imu_accel_max_norm_mps2", opt.imu_accel_max_norm_mps2);
    readDoubleIfPresent(tuning, "imu_fusion_integral_limit_rad_s", opt.imu_fusion_integral_limit_rad_s);
    readDoubleIfPresent(tuning, "imu_bias_lpf_alpha", opt.imu_bias_lpf_alpha);
    readDoubleIfPresent(tuning, "imu_bias_gyro_threshold_rad_s", opt.imu_bias_gyro_threshold_rad_s);
    readDoubleIfPresent(tuning, "imu_bias_residual_threshold_mps2", opt.imu_bias_residual_threshold_mps2);
    readDoubleIfPresent(tuning, "imu_bias_max_abs_mps2", opt.imu_bias_max_abs_mps2);
    readDoubleIfPresent(tuning, "imu_velocity_damping_per_sec", opt.imu_velocity_damping_per_sec);
    readDoubleIfPresent(tuning, "imu_vertical_velocity_damping_per_sec", opt.imu_vertical_velocity_damping_per_sec);
    readDoubleIfPresent(tuning, "imu_max_speed_mps", opt.imu_max_speed_mps);
    readDoubleIfPresent(tuning, "imu_max_vertical_speed_mps", opt.imu_max_vertical_speed_mps);
    readDoubleIfPresent(tuning, "mount_roll_deg", opt.mount_roll_deg);
    readDoubleIfPresent(tuning, "mount_pitch_deg", opt.mount_pitch_deg);
    readDoubleIfPresent(tuning, "image_rotation_offset_deg_clockwise", opt.image_rotation_offset_deg_clockwise);
    readDoubleIfPresent(tuning, "gps_to_imu_x_m", opt.gps_to_imu_x_m);
    readDoubleIfPresent(tuning, "gps_to_imu_y_m", opt.gps_to_imu_y_m);
    readDoubleIfPresent(tuning, "gps_to_imu_z_m", opt.gps_to_imu_z_m);
    readDoubleIfPresent(tuning, "window_sec", opt.window_sec);
    readDoubleIfPresent(tuning, "gnss_rate_hz", opt.gnss_rate_hz);
    readDoubleIfPresent(tuning, "heading_history_sec", opt.heading_history_sec);
    readDoubleIfPresent(tuning, "min_speed_mps", opt.min_speed_mps);
    readDoubleIfPresent(tuning, "max_residual_m", opt.max_residual_m);
    readDoubleIfPresent(tuning, "undistort_balance", opt.undistort_balance);
    readDoubleIfPresent(tuning, "undistort_fov_scale", opt.undistort_fov_scale);
    readDoubleIfPresent(tuning, "ray_step_m", opt.ray_step_m);
    readDoubleIfPresent(tuning, "max_ray_distance_m", opt.max_ray_distance_m);
    readIntIfPresent(tuning, "binary_refine_iters", opt.binary_refine_iters);
    readDoubleIfPresent(tuning, "crop_center_offset_forward_m", opt.crop_center_offset_forward_m);
    readDoubleIfPresent(tuning, "crop_center_offset_right_m", opt.crop_center_offset_right_m);
    readDoubleIfPresent(tuning, "ground_altitude", opt.ground_altitude);
    readDoubleIfPresent(tuning, "crop_dfov_deg", opt.crop_dfov_deg);
    readDoubleIfPresent(tuning, "crop_scale_factor", opt.crop_scale_factor);
    readDoubleIfPresent(tuning, "fusion_open_height", opt.fusion_open_height);
    readDoubleIfPresent(tuning, "crop_fusion_seed_max_age_sec", opt.crop_fusion_seed_max_age_sec);
    readBoolIfPresent(tuning, "persist_frame_artifacts", opt.persist_frame_artifacts);
    readBoolIfPresent(tuning, "enable_sp_lg", opt.enable_sp_lg);
    readStringIfPresent(tuning, "sp_lg_template_yaml", opt.sp_lg_template_yaml);
    readStringIfPresent(tuning, "sp_lg_executable", opt.sp_lg_executable);
    readStringIfPresent(tuning, "localization_csv", opt.localization_csv);
    readStringIfPresent(tuning, "sp_lg_result_csv", opt.sp_lg_result_csv);
    readStringIfPresent(tuning, "fix_fusion_csv", opt.fix_fusion_csv);
    readStringIfPresent(tuning, "imu_position_csv", opt.imu_position_csv);
    readStringIfPresent(tuning, "gnss_position_csv", opt.gnss_position_csv);
    readStringIfPresent(tuning, "sampled_geo_csv", opt.sampled_geo_csv);
    readStringIfPresent(tuning, "high_rate_fusion_csv", opt.high_rate_fusion_csv);
    readIntIfPresent(tuning, "fusion_anchor_min_used_points", opt.fusion_anchor_min_used_points);
    readDoubleIfPresent(tuning, "fusion_anchor_min_inlier_ratio", opt.fusion_anchor_min_inlier_ratio);
    readDoubleIfPresent(tuning, "fusion_anchor_max_reproj_error", opt.fusion_anchor_max_reproj_error);
    readDoubleIfPresent(tuning, "fusion_anchor_innovation_base_m", opt.fusion_anchor_innovation_base_m);
    readDoubleIfPresent(tuning, "fusion_anchor_innovation_speed_gain", opt.fusion_anchor_innovation_speed_gain);
    readDoubleIfPresent(tuning, "fusion_anchor_soft_update_alpha_min", opt.fusion_anchor_soft_update_alpha_min);
    readDoubleIfPresent(tuning, "fusion_anchor_soft_update_alpha_max", opt.fusion_anchor_soft_update_alpha_max);
    readDoubleIfPresent(tuning, "fusion_anchor_max_age_sec", opt.fusion_anchor_max_age_sec);
    readDoubleIfPresent(tuning, "fusion_imu_only_max_duration_sec", opt.fusion_imu_only_max_duration_sec);
    readDoubleIfPresent(tuning, "kf_process_speed_sigma_mps", opt.kf_process_speed_sigma_mps);
    readDoubleIfPresent(tuning, "kf_process_heading_sigma_deg", opt.kf_process_heading_sigma_deg);
    readDoubleIfPresent(tuning, "kf_process_xy_rw_sigma_m_sqrt_s", opt.kf_process_xy_rw_sigma_m_sqrt_s);
    readDoubleIfPresent(tuning, "kf_process_z_rw_sigma_m_sqrt_s", opt.kf_process_z_rw_sigma_m_sqrt_s);
    readDoubleIfPresent(tuning, "imu_preintegration_accel_sigma_mps2", opt.imu_preintegration_accel_sigma_mps2);
    readDoubleIfPresent(tuning, "imu_preintegration_max_gap_sec", opt.imu_preintegration_max_gap_sec);
    readDoubleIfPresent(tuning, "kf_visual_min_xy_sigma_m", opt.kf_visual_min_xy_sigma_m);
    readDoubleIfPresent(tuning, "kf_visual_min_z_sigma_m", opt.kf_visual_min_z_sigma_m);
    readBoolIfPresent(tuning, "kf_visual_adaptive_enable", opt.kf_visual_adaptive_enable);
    readDoubleIfPresent(tuning, "kf_visual_adaptive_initial_scale", opt.kf_visual_adaptive_initial_scale);
    readDoubleIfPresent(tuning, "kf_visual_adaptive_min_scale", opt.kf_visual_adaptive_min_scale);
    readDoubleIfPresent(tuning, "kf_visual_adaptive_max_scale", opt.kf_visual_adaptive_max_scale);
    readDoubleIfPresent(tuning, "kf_visual_adaptive_alpha", opt.kf_visual_adaptive_alpha);
    readDoubleIfPresent(tuning, "kf_visual_adaptive_target_nis", opt.kf_visual_adaptive_target_nis);
    readDoubleIfPresent(tuning, "kf_visual_nis_gate", opt.kf_visual_nis_gate);
    readBoolIfPresent(tuning, "kf_visual_reject_on_gate", opt.kf_visual_reject_on_gate);
    readIntIfPresent(tuning, "image_queue_size", opt.image_queue_size);
    readIntIfPresent(tuning, "fix_queue_size", opt.fix_queue_size);
    readIntIfPresent(tuning, "imu_queue_size", opt.imu_queue_size);
    bool has_crop_half_extent_e_m = false;
    bool has_crop_half_extent_n_m = false;
    if (!tuning.empty() && !tuning["crop_half_extent_e_m"].empty()) {
        opt.crop_half_extent_e_m = static_cast<double>(tuning["crop_half_extent_e_m"]);
        has_crop_half_extent_e_m = true;
    }
    if (!tuning.empty() && !tuning["crop_half_extent_n_m"].empty()) {
        opt.crop_half_extent_n_m = static_cast<double>(tuning["crop_half_extent_n_m"]);
        has_crop_half_extent_n_m = true;
    }
    if (!has_crop_half_extent_e_m || !has_crop_half_extent_n_m) {
        double legacy_half_extent_m = 0.0;
        bool has_legacy_half_extent_m = false;
        if (!tuning.empty() && !tuning["crop_min_half_extent_m"].empty()) {
            legacy_half_extent_m = static_cast<double>(tuning["crop_min_half_extent_m"]);
            has_legacy_half_extent_m = true;
        }
        if (has_legacy_half_extent_m) {
            if (!has_crop_half_extent_e_m) opt.crop_half_extent_e_m = legacy_half_extent_m;
            if (!has_crop_half_extent_n_m) opt.crop_half_extent_n_m = legacy_half_extent_m;
        }
    }

    opt.bag_path = resolvePathFromConfig(config_path, opt.bag_path);
    opt.cam_yaml = resolvePathFromConfig(config_path, opt.cam_yaml);
    opt.calib_txt = resolvePathFromConfig(config_path, opt.calib_txt);
    opt.map_tif = resolvePathFromConfig(config_path, opt.map_tif);
    opt.dem_tif = resolvePathFromConfig(config_path, opt.dem_tif);
    opt.out_dir = resolvePathFromConfig(config_path, opt.out_dir);
    opt.sp_lg_template_yaml = resolvePathFromConfig(config_path, opt.sp_lg_template_yaml);
    opt.sp_lg_executable = resolvePathFromConfig(config_path, opt.sp_lg_executable);
    opt.localization_csv = resolvePathFromConfig(config_path, opt.localization_csv);
    opt.sp_lg_result_csv = resolvePathFromConfig(config_path, opt.sp_lg_result_csv);
    opt.fix_fusion_csv = resolvePathFromConfig(config_path, opt.fix_fusion_csv);
    opt.imu_position_csv = resolvePathFromConfig(config_path, opt.imu_position_csv);
    opt.gnss_position_csv = resolvePathFromConfig(config_path, opt.gnss_position_csv);
    opt.sampled_geo_csv = resolvePathFromConfig(config_path, opt.sampled_geo_csv);
    opt.high_rate_fusion_csv = resolvePathFromConfig(config_path, opt.high_rate_fusion_csv);
    opt.input_mode = normalizeInputMode(opt.input_mode);

    std::cout << "[Config] loaded yaml=" << config_yaml << "\n";
}

// 功能：用位置参数覆盖部分运行配置。
static void applyPositionalOverrides(Options& opt, const std::vector<std::string>& args) {
    if (args.size() >= 1) opt.cam_yaml = args[0];
    if (args.size() >= 2) opt.calib_txt = args[1];
    if (args.size() >= 3) opt.map_tif = args[2];
    if (args.size() >= 4) opt.dem_tif = args[3];
    if (args.size() >= 5) opt.out_dir = args[4];
}

// 功能：从文本中解析中括号数组。
static bool parseBracketArray(const std::string& text, const std::string& key, std::vector<double>& out) {
    const size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) return false;
    const size_t lb = text.find('[', key_pos);
    const size_t rb = text.find(']', lb == std::string::npos ? key_pos : lb + 1);
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb + 1) return false;

    std::string body = text.substr(lb + 1, rb - lb - 1);
    for (char& c : body) {
        if (c == ',') c = ' ';
    }

    std::stringstream ss(body);
    out.clear();
    double v = 0.0;
    while (ss >> v) out.push_back(v);
    return !out.empty();
}

// 功能：从相机 YAML 中读取内参与分辨率。
static bool loadCameraIntrinsicsYaml(const std::string& path,
                                     double& fx,
                                     double& fy,
                                     double& cx,
                                     double& cy,
                                     std::array<double, 4>& d4,
                                     int& width,
                                     int& height) {
    auto parseByTextFallback = [&](const std::string& file_path) -> bool {
        std::ifstream ifs(file_path);
        if (!ifs.is_open()) return false;

        std::string all;
        std::string line;
        while (std::getline(ifs, line)) {
            all += line;
            all.push_back('\n');
        }

        std::vector<double> intr;
        std::vector<double> dist;
        std::vector<double> res;
        if (!parseBracketArray(all, "intrinsics", intr)) return false;
        if (!parseBracketArray(all, "distortion_coeffs", dist)) return false;
        parseBracketArray(all, "resolution", res);
        if (intr.size() < 4 || dist.size() < 4) return false;

        fx = intr[0];
        fy = intr[1];
        cx = intr[2];
        cy = intr[3];
        for (int i = 0; i < 4; ++i) d4[static_cast<size_t>(i)] = dist[static_cast<size_t>(i)];

        width = 0;
        height = 0;
        if (res.size() >= 2) {
            width = static_cast<int>(std::round(res[0]));
            height = static_cast<int>(std::round(res[1]));
        }
        return true;
    };

    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (fs.isOpened()) {
            cv::FileNode root = fs["cam0"];
            if (root.empty()) root = fs.root();

            cv::FileNode intr = root["intrinsics"];
            cv::FileNode dist = root["distortion_coeffs"];
            cv::FileNode res = root["resolution"];

            if (!intr.empty() && intr.isSeq() && intr.size() >= 4 &&
                !dist.empty() && dist.isSeq() && dist.size() >= 4) {
                fx = static_cast<double>(intr[0]);
                fy = static_cast<double>(intr[1]);
                cx = static_cast<double>(intr[2]);
                cy = static_cast<double>(intr[3]);
                for (int i = 0; i < 4; ++i) d4[static_cast<size_t>(i)] = static_cast<double>(dist[i]);

                width = 0;
                height = 0;
                if (!res.empty() && res.isSeq() && res.size() >= 2) {
                    width = static_cast<int>(res[0]);
                    height = static_cast<int>(res[1]);
                }
                return true;
            }
        }
    } catch (const cv::Exception&) {
    }

    return parseByTextFallback(path);
}

// 功能：加载相机模型。
static CameraModel loadCameraModel(const std::string& yaml_path) {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    int width = 0;
    int height = 0;
    std::array<double, 4> d4{{0.0, 0.0, 0.0, 0.0}};
    if (!loadCameraIntrinsicsYaml(yaml_path, fx, fy, cx, cy, d4, width, height)) {
        throw std::runtime_error("Failed to parse camera intrinsics: " + yaml_path);
    }

    CameraModel cam;
    cam.K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
             0.0, fy, cy,
             0.0, 0.0, 1.0);
    cam.D = cv::Mat::zeros(1, 4, CV_64F);
    cam.width = width;
    cam.height = height;
    for (int i = 0; i < 4; ++i) cam.D.at<double>(0, i) = d4[static_cast<size_t>(i)];

    std::cout << "[Camera] yaml=" << yaml_path << "\n"
              << "[Camera] K=" << cam.K << "\n"
              << "[Camera] D=" << cam.D << "\n";
    if (width > 0 && height > 0) {
        std::cout << "[Camera] resolution=" << width << "x" << height << "\n";
    }
    return cam;
}

// 功能：加载 `loadSpLgTemplateYaml` 相关数据或配置。
static SpLgTemplateOptions loadSpLgTemplateYaml(const std::string& yaml_path) {
    SpLgTemplateOptions cfg;
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open sp_lg template yaml: " + yaml_path);
    }

    const std::filesystem::path cfg_path(yaml_path);
    const cv::FileNode model = fs["model"];
    const cv::FileNode feature = fs["feature"];
    const cv::FileNode matching = fs["matching"];
    const cv::FileNode runtime = fs["runtime"];
    const cv::FileNode geo = fs["geo"];

    readStringIfPresent(model, "superpoint", cfg.superpoint_onnx);
    readStringIfPresent(model, "lightglue", cfg.lightglue_onnx);
    readStringIfPresent(feature, "method", cfg.feature_method);
    readIntIfPresent(feature, "max_features", cfg.max_features);
    readIntIfPresent(feature, "fast_threshold", cfg.fast_threshold);
    readBoolIfPresent(feature, "fast_nonmax", cfg.fast_nonmax);
    readStringIfPresent(matching, "method", cfg.matcher_method);
    readStringIfPresent(matching, "distance", cfg.matcher_distance);
    readDoubleIfPresent(matching, "ratio", cfg.matcher_ratio);
    readBoolIfPresent(matching, "cross_check", cfg.matcher_cross_check);
    readIntIfPresent(matching, "max_keep", cfg.matcher_max_keep);
    readIntIfPresent(runtime, "H", cfg.H);
    readIntIfPresent(runtime, "W", cfg.W);
    readDoubleIfPresent(runtime, "mscore_thresh", cfg.mscore_thresh);
    readDoubleIfPresent(runtime, "min_kpt_dist", cfg.min_kpt_dist);
    readIntIfPresent(runtime, "max_draw", cfg.max_draw);
    readDoubleIfPresent(runtime, "homography_ransac_reproj", cfg.homography_ransac_reproj);
    readIntIfPresent(runtime, "homography_max_pairs", cfg.homography_max_pairs);
    readBoolIfPresent(runtime, "try_cuda", cfg.try_cuda);
    readIntIfPresent(runtime, "device_id", cfg.device_id);
    readDoubleIfPresent(geo, "dem_nodata", cfg.dem_nodata);

    cfg.superpoint_onnx = resolvePathFromConfig(cfg_path, cfg.superpoint_onnx);
    cfg.lightglue_onnx = resolvePathFromConfig(cfg_path, cfg.lightglue_onnx);

    std::cout << "[sp_lg] template=" << yaml_path << "\n"
              << "[sp_lg] superpoint=" << cfg.superpoint_onnx << "\n"
              << "[sp_lg] lightglue=" << cfg.lightglue_onnx << "\n"
              << "[sp_lg] feature.method=" << cfg.feature_method
              << " matcher.method=" << cfg.matcher_method
              << " runtime=" << cfg.W << "x" << cfg.H << "\n";
    return cfg;
}

// 功能：解析 `parseNumericRow` 相关输入。
static std::vector<double> parseNumericRow(const std::string& row_text) {
    std::string cleaned = row_text;
    for (char& c : cleaned) {
        if (c == '[' || c == ']' || c == ',') c = ' ';
    }
    std::stringstream ss(cleaned);
    std::vector<double> row;
    double v = 0.0;
    while (ss >> v) row.push_back(v);
    return row;
}

// 功能：从外参文本文件中加载相机到 IMU 的外参。
static CameraExtrinsics loadCameraExtrinsics(const std::string& calib_path, const std::string& camera_name) {
    std::ifstream ifs(calib_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open calibration file: " + calib_path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) lines.push_back(line);

    CameraExtrinsics ext;
    bool got_matrix = false;
    bool got_shift = false;

    auto nextNonEmptyLine = [&](size_t start, std::string& out_line) -> bool {
        for (size_t i = start; i < lines.size(); ++i) {
            if (!lines[i].empty()) {
                out_line = lines[i];
                return true;
            }
        }
        return false;
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string& cur = lines[i];
        const std::string matrix_key = "T_ic:  (" + camera_name + " to imu0):";
        const std::string shift_key = "timeshift " + camera_name + " to imu0:";

        if (!got_matrix && cur.find(matrix_key) != std::string::npos) {
            std::array<std::array<double, 4>, 4> mat{};
            size_t row_count = 0;
            for (size_t j = i + 1; j < lines.size() && row_count < 4; ++j) {
                if (lines[j].find('[') == std::string::npos) continue;
                const std::vector<double> row = parseNumericRow(lines[j]);
                if (row.size() < 4) continue;
                for (size_t k = 0; k < 4; ++k) mat[row_count][k] = row[k];
                ++row_count;
            }
            if (row_count != 4) {
                throw std::runtime_error("Failed to parse T_ic from: " + calib_path);
            }

            ext.R_ic = cv::Matx33d(
                mat[0][0], mat[0][1], mat[0][2],
                mat[1][0], mat[1][1], mat[1][2],
                mat[2][0], mat[2][1], mat[2][2]);
            ext.t_ic = cv::Vec3d(mat[0][3], mat[1][3], mat[2][3]);
            got_matrix = true;
        }

        if (!got_shift && cur.find(shift_key) != std::string::npos) {
            std::string shift_line;
            if (!nextNonEmptyLine(i + 1, shift_line)) {
                throw std::runtime_error("Failed to parse cam-to-imu timeshift from: " + calib_path);
            }
            ext.cam_to_imu_shift_s = std::stod(shift_line);
            got_shift = true;
        }
    }

    if (!got_matrix) {
        throw std::runtime_error("Did not find T_ic for " + camera_name + " in " + calib_path);
    }
    if (!got_shift) {
        throw std::runtime_error("Did not find timeshift for " + camera_name + " in " + calib_path);
    }

    std::cout << "[Extrinsics] calib=" << calib_path << "\n"
              << "[Extrinsics] R_ic=\n" << cv::Mat(ext.R_ic) << "\n"
              << "[Extrinsics] t_ic=[" << ext.t_ic[0] << ", " << ext.t_ic[1] << ", " << ext.t_ic[2] << "]\n"
              << "[Extrinsics] cam_to_imu_shift_s=" << ext.cam_to_imu_shift_s << "\n";
    return ext;
}

// 功能：构建 `buildFisheyeUndistortCache` 相关对象或中间结果。
static bool buildFisheyeUndistortCache(const cv::Size& src_size,
                                       const CameraModel& cam,
                                       double balance,
                                       double fov_scale,
                                       FisheyeUndistortCache& cache) {
    if (src_size.width <= 0 || src_size.height <= 0 || cam.K.empty() || cam.D.empty()) return false;

    const cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cache.src_size = src_size;
    cache.balance = balance;
    cache.fov_scale = fov_scale;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        cam.K, cam.D, src_size, R, cache.newK, balance, src_size, fov_scale);

    cv::fisheye::initUndistortRectifyMap(
        cam.K, cam.D, R, cache.newK, src_size, CV_32FC1, cache.map1, cache.map2);
    return !cache.newK.empty() && !cache.map1.empty() && !cache.map2.empty();
}

// 功能：实现 `undistortFisheye` 对应的功能。
static bool undistortFisheye(const cv::Mat& src,
                             const FisheyeUndistortCache& cache,
                             cv::Mat& undist,
                             cv::Mat& newK) {
    if (src.empty()) return false;
    if (cache.map1.empty() || cache.map2.empty() || cache.newK.empty()) return false;

    newK = cache.newK.clone();
    cv::remap(src, undist, cache.map1, cache.map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return !undist.empty();
}

// 功能：实现 `imageMsgToMat` 对应的功能。
static cv::Mat imageMsgToMat(const sensor_msgs::Image& msg) {
    int type = -1;
    bool swap_rgb = false;
    bool has_alpha = false;

    if (msg.encoding == "mono8" || msg.encoding == "8UC1") {
        type = CV_8UC1;
    } else if (msg.encoding == "bgr8" || msg.encoding == "8UC3") {
        type = CV_8UC3;
    } else if (msg.encoding == "rgb8") {
        type = CV_8UC3;
        swap_rgb = true;
    } else if (msg.encoding == "bgra8" || msg.encoding == "8UC4") {
        type = CV_8UC4;
        has_alpha = true;
    } else if (msg.encoding == "rgba8") {
        type = CV_8UC4;
        swap_rgb = true;
        has_alpha = true;
    } else if (msg.encoding == "mono16" || msg.encoding == "16UC1") {
        type = CV_16UC1;
    } else {
        throw std::runtime_error("Unsupported image encoding: " + msg.encoding);
    }

    const int channels = CV_MAT_CN(type);
    const size_t min_step = static_cast<size_t>(msg.width) * channels * CV_ELEM_SIZE1(type);
    if (msg.step < min_step) {
        throw std::runtime_error("Image step is smaller than expected.");
    }
    if (msg.data.size() < static_cast<size_t>(msg.step) * msg.height) {
        throw std::runtime_error("Image data is smaller than step * height.");
    }

    cv::Mat wrapped(static_cast<int>(msg.height),
                    static_cast<int>(msg.width),
                    type,
                    const_cast<unsigned char*>(msg.data.data()),
                    msg.step);
    cv::Mat out = wrapped.clone();

    if (swap_rgb && has_alpha) {
        cv::cvtColor(out, out, cv::COLOR_RGBA2BGRA);
    } else if (swap_rgb) {
        cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
    }
    return out;
}

// 功能：实现 `rotationTransformKeepFullImage` 对应的功能。
static cv::Matx33d rotationTransformKeepFullImage(int width, int height, double angle_deg_ccw) {
    const cv::Point2f center((static_cast<float>(width) - 1.0f) * 0.5f,
                             (static_cast<float>(height) - 1.0f) * 0.5f);
    cv::Mat M = cv::getRotationMatrix2D(center, angle_deg_ccw, 1.0);

    const double cos_a = std::abs(M.at<double>(0, 0));
    const double sin_a = std::abs(M.at<double>(0, 1));
    const int new_w = static_cast<int>(std::round(static_cast<double>(height) * sin_a +
                                                  static_cast<double>(width) * cos_a));
    const int new_h = static_cast<int>(std::round(static_cast<double>(height) * cos_a +
                                                  static_cast<double>(width) * sin_a));

    M.at<double>(0, 2) += static_cast<double>(new_w) * 0.5 - center.x;
    M.at<double>(1, 2) += static_cast<double>(new_h) * 0.5 - center.y;

    return cv::Matx33d(
        M.at<double>(0, 0), M.at<double>(0, 1), M.at<double>(0, 2),
        M.at<double>(1, 0), M.at<double>(1, 1), M.at<double>(1, 2),
        0.0, 0.0, 1.0);
}

// 功能：实现 `rotateKeepFullImage` 对应的功能。
static cv::Mat rotateKeepFullImage(const cv::Mat& src,
                                   double angle_deg_ccw,
                                   cv::Matx33d* transform_out = nullptr) {
    if (src.empty()) return {};

    const cv::Matx33d A = rotationTransformKeepFullImage(src.cols, src.rows, angle_deg_ccw);
    if (transform_out) *transform_out = A;

    cv::Mat M = (cv::Mat_<double>(2, 3) << A(0, 0), A(0, 1), A(0, 2),
                 A(1, 0), A(1, 1), A(1, 2));
    const int new_w = static_cast<int>(std::round(std::abs(A(0, 0)) * src.cols + std::abs(A(0, 1)) * src.rows));
    const int new_h = static_cast<int>(std::round(std::abs(A(1, 0)) * src.cols + std::abs(A(1, 1)) * src.rows));

    cv::Mat rotated;
    cv::warpAffine(src, rotated, M, cv::Size(new_w, new_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return rotated;
}

// 功能：实现 `rotateCameraMatrix` 对应的功能。
static cv::Mat rotateCameraMatrix(const cv::Mat& K, const cv::Matx33d& image_transform) {
    cv::Matx33d K_rotated = image_transform;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            K_rotated(r, c) = image_transform(r, 0) * K.at<double>(0, c) +
                              image_transform(r, 1) * K.at<double>(1, c) +
                              image_transform(r, 2) * K.at<double>(2, c);
        }
    }
    return cv::Mat(K_rotated);
}
