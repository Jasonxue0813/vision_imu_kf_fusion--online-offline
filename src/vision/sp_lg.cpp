/**
 * @file sp_lg.cpp
 * @brief 单帧视觉定位程序。
 *
 * 这个程序读取单帧运行配置，完成特征提取、特征匹配、
 * 单应估计、PnP 求解、质量评估以及 worker 模式下的请求处理。
 */

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <gdal_priv.h>
#include <ogr_spatialref.h>

#if __has_include(<opencv2/xfeatures2d.hpp>)
#include <opencv2/xfeatures2d.hpp>
#define HAS_OPENCV_XFEATURES2D 1
#else
#define HAS_OPENCV_XFEATURES2D 0
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct SPFeatures {
    std::vector<cv::KeyPoint> kpts;
    std::vector<float> scores;
    std::vector<float> desc;
    int64_t N = 0;
    int64_t D = 0;
};

struct MatchWithScore {
    int q = -1;
    int t = -1;
    float s = 0.f;
};

enum class FeatureMethod {
    SuperPoint = 0,
    SIFT = 1,
    SURF = 2,
    FAST = 3,
    ORB = 4
};

enum class MatcherMethod {
    LightGlue = 0,
    BF = 1,
    FLANN = 2
};

enum class DistanceType {
    Auto = 0,
    L2 = 1,
    Hamming = 2
};

struct AppConfig {
    std::string image0;
    std::string image1;
    std::string superpoint_onnx;
    std::string lightglue_onnx;

    std::string sp0_kpts_out;
    std::string sp1_kpts_out;
    std::string matches_out;
    std::string matches_inlier_out;
    std::string warp_out;
    std::string compare_out;
    std::string undistort_out;
    std::string map2d_csv_out;
    std::string map3d_csv_out;
    std::string pnp_report_out;
    std::string dem_tif;

    int H = 480;
    int W = 752;
    float mscore_thresh = 0.1f;
    float min_kpt_dist = 0.f;
    int max_draw = 200;
    double homography_ransac_reproj = 3.0;
    int homography_max_pairs = 300;

    std::string feature_method = "superpoint";
    int max_features = 4000;
    int fast_threshold = 20;
    bool fast_nonmax = true;

    std::string matcher_method = "lightglue";
    std::string matcher_distance = "auto";
    float matcher_ratio = 0.8f;
    bool matcher_cross_check = false;
    int matcher_max_keep = 3000;

    bool use_fisheye_undistort = true;
    double undistort_balance = 0.0;
    double undistort_fov_scale = 1.0;
    std::string camera_intrinsics_file;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::array<double, 4> fish_distortion{{0.0, 0.0, 0.0, 0.0}};
    std::array<double, 9> camera_matrix{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    bool camera_matrix_valid = false;
    std::array<double, 9> pnp_camera_matrix{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}};
    bool pnp_camera_matrix_valid = false;
    std::array<double, 9> image1_inverse_affine{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    bool image1_inverse_affine_valid = false;

    std::array<double, 6> map_wgs84_affine{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}};
    std::array<double, 6> dem_wgs84_affine{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}};
    bool map_affine_valid = false;
    bool dem_affine_valid = false;
    float dem_nodata = std::numeric_limits<float>::quiet_NaN();

    bool try_cuda = true;
    int device_id = 0;
};

struct CameraModel {
    cv::Mat K;  // CV_64F 3x3
    cv::Mat D;  // CV_64F 1x4(+) fisheye coefficients
};

struct Geo2DRecord {
    float cam_x = 0.f;
    float cam_y = 0.f;
    float map_x = 0.f;
    float map_y = 0.f;
    double lon = 0.0;
    double lat = 0.0;
    float score = 0.f;
};

struct Geo3DRecord {
    Geo2DRecord m;
    float alt = 0.f;
};

struct TerrainStats {
    double relief_m = std::numeric_limits<double>::quiet_NaN();
    double alt_std_m = std::numeric_limits<double>::quiet_NaN();
    double alt_median_m = std::numeric_limits<double>::quiet_NaN();
};

struct GeometryFeatureStats {
    double point_spread_img_ratio = std::numeric_limits<double>::quiet_NaN();
    double point_spread_map_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    double depth_spread_m = std::numeric_limits<double>::quiet_NaN();
    double obliqueness_deg = std::numeric_limits<double>::quiet_NaN();
};

struct DualGeometryStats {
    bool valid = false;
    int shared_count = 0;
    double homography_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double pnp_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double disagree_rmse_px = std::numeric_limits<double>::quiet_NaN();
    double px_var_u = std::numeric_limits<double>::quiet_NaN();
    double px_cov_uv = std::numeric_limits<double>::quiet_NaN();
    double px_var_v = std::numeric_limits<double>::quiet_NaN();
    double jacobian_11 = std::numeric_limits<double>::quiet_NaN();
    double jacobian_12 = std::numeric_limits<double>::quiet_NaN();
    double jacobian_21 = std::numeric_limits<double>::quiet_NaN();
    double jacobian_22 = std::numeric_limits<double>::quiet_NaN();
    double jacobian_det = std::numeric_limits<double>::quiet_NaN();
    double jacobian_cond = std::numeric_limits<double>::quiet_NaN();
    double local_scale_u_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double local_scale_v_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double local_scale_mean_m_per_px = std::numeric_limits<double>::quiet_NaN();
    double var_x_m2 = std::numeric_limits<double>::quiet_NaN();
    double cov_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    double var_y_m2 = std::numeric_limits<double>::quiet_NaN();
};

struct LocalFeatures {
    std::vector<cv::KeyPoint> kpts;
    std::vector<float> scores;
    cv::Mat desc;
    bool binary_desc = false;
};

struct RasterGeoRef {
    bool valid = false;
    std::array<double, 6> gt{{0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    std::array<double, 6> inv_gt{{0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    std::string projection_wkt;
};

// 功能：返回字符串的小写副本。
static std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// 功能：返回去除首尾空白后的字符串副本。
static std::string trimCopy(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// 功能：安全地计算比值并处理分母为零的情况。
static double safeRatio(int numerator, int denominator) {
    if (denominator <= 0) return std::numeric_limits<double>::quiet_NaN();
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

// 功能：统计地形点集合的起伏和分布信息。
static TerrainStats computeTerrainStats(const std::vector<Geo3DRecord>& geo3d) {
    TerrainStats stats;
    if (geo3d.empty()) return stats;

    std::vector<double> alts;
    alts.reserve(geo3d.size());
    double sum = 0.0;
    double min_alt = std::numeric_limits<double>::infinity();
    double max_alt = -std::numeric_limits<double>::infinity();
    for (const auto& row : geo3d) {
        const double alt = static_cast<double>(row.alt);
        if (!std::isfinite(alt)) continue;
        alts.push_back(alt);
        sum += alt;
        min_alt = std::min(min_alt, alt);
        max_alt = std::max(max_alt, alt);
    }
    if (alts.empty()) return stats;

    stats.relief_m = max_alt - min_alt;

    const double mean = sum / static_cast<double>(alts.size());
    double sq_sum = 0.0;
    for (double alt : alts) {
        const double delta = alt - mean;
        sq_sum += delta * delta;
    }
    stats.alt_std_m = std::sqrt(sq_sum / static_cast<double>(alts.size()));

    const size_t mid = alts.size() / 2;
    std::nth_element(alts.begin(), alts.begin() + static_cast<std::ptrdiff_t>(mid), alts.end());
    stats.alt_median_m = alts[mid];
    if ((alts.size() % 2) == 0) {
        const double upper = stats.alt_median_m;
        std::nth_element(alts.begin(),
                         alts.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                         alts.begin() + static_cast<std::ptrdiff_t>(mid));
        stats.alt_median_m = 0.5 * (alts[mid - 1] + upper);
    }

    return stats;
}

// 功能：计算 `computePointSpreadImgRatio` 对应的结果。
static double computePointSpreadImgRatio(const std::vector<cv::Point2f>& pts,
                                         double image_width,
                                         double image_height) {
    if (pts.size() < 2 || image_width <= 1.0 || image_height <= 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& pt : pts) {
        min_x = std::min(min_x, static_cast<double>(pt.x));
        max_x = std::max(max_x, static_cast<double>(pt.x));
        min_y = std::min(min_y, static_cast<double>(pt.y));
        max_y = std::max(max_y, static_cast<double>(pt.y));
    }

    const double bbox_w = std::max(0.0, max_x - min_x);
    const double bbox_h = std::max(0.0, max_y - min_y);
    return (bbox_w * bbox_h) / (image_width * image_height);
}

// 功能：计算地图平面上点集的覆盖面积。
static double computePointSpreadMapXYM2(const std::vector<cv::Point3f>& obj_pts) {
    if (obj_pts.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto& pt : obj_pts) {
        min_x = std::min(min_x, static_cast<double>(pt.x));
        max_x = std::max(max_x, static_cast<double>(pt.x));
        min_y = std::min(min_y, static_cast<double>(pt.y));
        max_y = std::max(max_y, static_cast<double>(pt.y));
    }

    const double bbox_w = std::max(0.0, max_x - min_x);
    const double bbox_h = std::max(0.0, max_y - min_y);
    return bbox_w * bbox_h;
}

// 功能：计算 `computeDepthSpreadM` 对应的结果。
static double computeDepthSpreadM(const std::vector<cv::Point3f>& obj_pts,
                                  const cv::Vec3d& camera_center_enu) {
    if (obj_pts.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> dists;
    dists.reserve(obj_pts.size());
    double sum = 0.0;
    for (const auto& pt : obj_pts) {
        const cv::Vec3d delta(
            static_cast<double>(pt.x) - camera_center_enu[0],
            static_cast<double>(pt.y) - camera_center_enu[1],
            static_cast<double>(pt.z) - camera_center_enu[2]);
        const double dist = cv::norm(delta);
        dists.push_back(dist);
        sum += dist;
    }

    const double mean = sum / static_cast<double>(dists.size());
    double sq_sum = 0.0;
    for (double dist : dists) {
        const double diff = dist - mean;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / static_cast<double>(dists.size()));
}

// 功能：计算相机视角相对地面的斜视角。
static double computeObliquenessDeg(const cv::Mat& R_world_to_camera) {
    if (R_world_to_camera.empty() || R_world_to_camera.rows != 3 || R_world_to_camera.cols != 3) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const cv::Mat optical_axis_world_mat =
        R_world_to_camera.t() * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
    const cv::Vec3d optical_axis_world(
        optical_axis_world_mat.at<double>(0),
        optical_axis_world_mat.at<double>(1),
        optical_axis_world_mat.at<double>(2));

    const double axis_norm = cv::norm(optical_axis_world);
    if (!std::isfinite(axis_norm) || axis_norm <= 1e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const cv::Vec3d downward_world(0.0, 0.0, -1.0);
    const double cosine =
        std::clamp(optical_axis_world.dot(downward_world) / axis_norm, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / CV_PI;
}

// 功能：把配置中的枚举字符串规整成统一格式。
static std::string sanitizeConfigToken(const std::string& s_in) {
    std::string s = s_in;
    const size_t hash = s.find('#');
    if (hash != std::string::npos) {
        s = s.substr(0, hash);
    }
    return trimCopy(s);
}

// 功能：解析特征提取方法配置。
static FeatureMethod parseFeatureMethod(const std::string& s_in) {
    const std::string s = toLowerCopy(sanitizeConfigToken(s_in));
    if (s == "superpoint") return FeatureMethod::SuperPoint;
    if (s == "sift") return FeatureMethod::SIFT;
    if (s == "surf") return FeatureMethod::SURF;
    if (s == "fast") return FeatureMethod::FAST;
    if (s == "orb") return FeatureMethod::ORB;
    throw std::runtime_error("Unsupported feature.method: " + s_in);
}

static MatcherMethod parseMatcherMethod(const std::string& s_in) {
    const std::string s = toLowerCopy(sanitizeConfigToken(s_in));
    if (s == "lightglue") return MatcherMethod::LightGlue;
    if (s == "bf" || s == "bruteforce" || s == "brute_force") return MatcherMethod::BF;
    if (s == "flann") return MatcherMethod::FLANN;
    if (s == "euclidean" || s == "l2") return MatcherMethod::BF;
    if (s == "hamming") return MatcherMethod::BF;
    throw std::runtime_error("Unsupported matching.method: " + s_in);
}

static DistanceType parseDistanceType(const std::string& s_in) {
    const std::string s = toLowerCopy(sanitizeConfigToken(s_in));
    if (s == "auto") return DistanceType::Auto;
    if (s == "l2" || s == "euclidean") return DistanceType::L2;
    if (s == "hamming") return DistanceType::Hamming;
    throw std::runtime_error("Unsupported matching.distance: " + s_in);
}

static const char* featureMethodName(FeatureMethod m) {
    switch (m) {
        case FeatureMethod::SuperPoint: return "superpoint";
        case FeatureMethod::SIFT: return "sift";
        case FeatureMethod::SURF: return "surf";
        case FeatureMethod::FAST: return "fast";
        case FeatureMethod::ORB: return "orb";
    }
    return "unknown";
}

// 功能：把匹配方法枚举转成可读字符串。
static const char* matcherMethodName(MatcherMethod m) {
    switch (m) {
        case MatcherMethod::LightGlue: return "lightglue";
        case MatcherMethod::BF: return "bf";
        case MatcherMethod::FLANN: return "flann";
    }
    return "unknown";
}

// 功能：把距离类型枚举转成可读字符串。
static const char* distanceTypeName(DistanceType d) {
    switch (d) {
        case DistanceType::Auto: return "auto";
        case DistanceType::L2: return "l2";
        case DistanceType::Hamming: return "hamming";
    }
    return "unknown";
}

// 功能：判断某种特征方法是否生成二值描述子。
static bool featureProducesBinaryDesc(FeatureMethod m) {
    return (m == FeatureMethod::ORB || m == FeatureMethod::FAST);
}

// 功能：校验特征方法、匹配器和距离度量的组合是否合法。
static void validateCombo(FeatureMethod feat, MatcherMethod matcher, DistanceType dist) {
    if (matcher == MatcherMethod::LightGlue && feat != FeatureMethod::SuperPoint) {
        throw std::runtime_error("Invalid combination: matching.method=lightglue requires feature.method=superpoint.");
    }

    if (matcher != MatcherMethod::LightGlue) {
        const bool binary = featureProducesBinaryDesc(feat);
        DistanceType d = dist;
        if (d == DistanceType::Auto) d = binary ? DistanceType::Hamming : DistanceType::L2;

        if (binary && d != DistanceType::Hamming) {
            throw std::runtime_error("Invalid combination: binary descriptors (orb/fast) require hamming distance.");
        }
        if (!binary && d != DistanceType::L2) {
            throw std::runtime_error("Invalid combination: float descriptors (superpoint/sift/surf) require l2 distance.");
        }
    }
}

static bool hasTiffExt(const std::string& path) {
    const std::string lower = toLowerCopy(path);
    return lower.size() >= 4 && (lower.rfind(".tif") == lower.size() - 4 || lower.rfind(".tiff") == lower.size() - 5);
}

// 功能：把相对路径解析到给定基目录下。
static std::filesystem::path resolvePath(const std::filesystem::path& base_dir, const std::string& p) {
    std::filesystem::path pp(p);
    if (pp.is_absolute()) return pp.lexically_normal();
    return (base_dir / pp).lexically_normal();
}

// 功能：确保某个文件路径对应的父目录存在。
static void ensureParentDir(const std::string& file_path) {
    std::filesystem::path p(file_path);
    auto parent = p.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}

// 功能：基于参考文件路径生成同目录下的兄弟文件路径。
static std::string makeSiblingPath(const std::string& ref_path, const std::string& suffix) {
    const std::filesystem::path p(ref_path);
    const std::filesystem::path dir = p.parent_path();
    std::string ext = p.extension().string();
    if (ext.empty()) ext = ".png";
    const std::string name = p.stem().string() + suffix + ext;
    return (dir / name).string();
}

// 功能：基于参考文件路径生成指定后缀和扩展名的兄弟文件路径。
static std::string makeSiblingPathWithExt(const std::string& ref_path, const std::string& suffix, const std::string& ext) {
    const std::filesystem::path p(ref_path);
    const std::filesystem::path dir = p.parent_path();
    const std::string name = p.stem().string() + suffix + ext;
    return (dir / name).string();
}

// 功能：确保 GDAL 驱动已经完成注册。
static void ensureGdalRegistered() {
    static bool registered = false;
    if (!registered) {
        GDALAllRegister();
        registered = true;
    }
}

// 功能：读取栅格文件的地理参考信息。
static bool loadRasterGeoRef(const std::string& path, RasterGeoRef& ref, std::string& reason) {
    ensureGdalRegistered();
    ref = RasterGeoRef{};
    reason.clear();

    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (!ds) {
        reason = "GDALOpen failed";
        return false;
    }

    double gt[6] = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    if (ds->GetGeoTransform(gt) != CE_None) {
        reason = "GeoTransform not found";
        GDALClose(ds);
        return false;
    }

    double inv_gt[6] = {0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    if (!GDALInvGeoTransform(gt, inv_gt)) {
        reason = "GeoTransform is not invertible";
        GDALClose(ds);
        return false;
    }

    const char* proj = ds->GetProjectionRef();
    if (proj) ref.projection_wkt = proj;

    for (int i = 0; i < 6; ++i) {
        ref.gt[static_cast<size_t>(i)] = gt[i];
        ref.inv_gt[static_cast<size_t>(i)] = inv_gt[i];
    }
    ref.valid = true;

    GDALClose(ds);
    return true;
}

// 功能：实现 `transformXY` 对应的功能。
static bool transformXY(const std::string& src_wkt,
                        const std::string& dst_wkt,
                        double& x,
                        double& y) {
    OGRSpatialReference src;
    OGRSpatialReference dst;
    if (src.SetFromUserInput(src_wkt.c_str()) != OGRERR_NONE) return false;
    if (dst.SetFromUserInput(dst_wkt.c_str()) != OGRERR_NONE) return false;

#if GDAL_VERSION_NUM >= 3000000
    src.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    dst.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

    OGRCoordinateTransformation* ct = OGRCreateCoordinateTransformation(&src, &dst);
    if (!ct) return false;

    double z = 0.0;
    const int ok = ct->Transform(1, &x, &y, &z);
    OCTDestroyCoordinateTransformation(ct);
    return ok != 0;
}

// 功能：实现 `applyGeoTransformArray` 对应的功能。
static void applyGeoTransformArray(const std::array<double, 6>& gt,
                                   double x,
                                   double y,
                                   double& out_x,
                                   double& out_y) {
    double g[6] = {gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]};
    GDALApplyGeoTransform(g, x, y, &out_x, &out_y);
}

static bool rasterPixelToWgs84(const RasterGeoRef& ref,
                               double px,
                               double py,
                               double& lon,
                               double& lat) {
    if (!ref.valid) return false;

    double gx = 0.0;
    double gy = 0.0;
    applyGeoTransformArray(ref.gt, px, py, gx, gy);

    if (ref.projection_wkt.empty()) {
        lon = gx;
        lat = gy;
        return true;
    }

    lon = gx;
    lat = gy;
    return transformXY(ref.projection_wkt, "EPSG:4326", lon, lat);
}

// 功能：实现 `wgs84ToRasterPixel` 对应的功能。
static bool wgs84ToRasterPixel(const RasterGeoRef& ref,
                               double lon,
                               double lat,
                               double& px,
                               double& py) {
    if (!ref.valid) return false;

    double gx = lon;
    double gy = lat;
    if (!ref.projection_wkt.empty()) {
        if (!transformXY("EPSG:4326", ref.projection_wkt, gx, gy)) return false;
    }

    applyGeoTransformArray(ref.inv_gt, gx, gy, px, py);
    return true;
}

template <size_t N>
// 功能：从 OpenCV FileNode 中读取定长数组。
static bool readArrayFromNode(const cv::FileNode& n, std::array<double, N>& arr) {
    if (n.empty() || !n.isSeq() || n.size() < N) return false;
    for (size_t i = 0; i < N; ++i) arr[i] = static_cast<double>(n[static_cast<int>(i)]);
    return true;
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
    auto parseBracketArray = [](const std::string& text, const std::string& key, std::vector<double>& out) -> bool {
        const size_t key_pos = text.find(key);
        if (key_pos == std::string::npos) return false;
        const size_t lb = text.find('[', key_pos);
        const size_t rb = text.find(']', lb == std::string::npos ? key_pos : lb + 1);
        if (lb == std::string::npos || rb == std::string::npos || rb <= lb + 1) return false;

        std::string body = text.substr(lb + 1, rb - lb - 1);
        std::replace(body.begin(), body.end(), ',', ' ');

        std::stringstream ss(body);
        out.clear();
        double v = 0.0;
        while (ss >> v) out.push_back(v);
        return !out.empty();
    };

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
        // 普通YAML会触发FileStorage异常，这里转到文本解析回退逻辑。
    }

    return parseByTextFallback(path);
}

// 功能：从运行配置构建相机模型。
static bool buildCameraModelFromConfig(const AppConfig& cfg, CameraModel& cam) {
    if (cfg.camera_matrix_valid) {
        cam.K = (cv::Mat_<double>(3, 3) << cfg.camera_matrix[0], cfg.camera_matrix[1], cfg.camera_matrix[2],
                 cfg.camera_matrix[3], cfg.camera_matrix[4], cfg.camera_matrix[5],
                 cfg.camera_matrix[6], cfg.camera_matrix[7], cfg.camera_matrix[8]);
        cam.D = cv::Mat::zeros(1, 4, CV_64F);
        for (int i = 0; i < 4; ++i) cam.D.at<double>(0, i) = cfg.fish_distortion[static_cast<size_t>(i)];
        return true;
    }
    if (cfg.fx <= 0.0 || cfg.fy <= 0.0) return false;
    cam.K = (cv::Mat_<double>(3, 3) << cfg.fx, 0.0, cfg.cx,
             0.0, cfg.fy, cfg.cy,
             0.0, 0.0, 1.0);
    cam.D = cv::Mat::zeros(1, 4, CV_64F);
    for (int i = 0; i < 4; ++i) cam.D.at<double>(0, i) = cfg.fish_distortion[static_cast<size_t>(i)];
    return true;
}

// 功能：把 9 维数组转成 OpenCV `Mat` 矩阵。
static cv::Mat matFromArray9(const std::array<double, 9>& arr) {
    return (cv::Mat_<double>(3, 3) << arr[0], arr[1], arr[2],
            arr[3], arr[4], arr[5],
            arr[6], arr[7], arr[8]);
}

// 功能：把 9 维数组转成 `cv::Matx33d`。
static cv::Matx33d matx33dFromArray9(const std::array<double, 9>& arr) {
    return cv::Matx33d(
        arr[0], arr[1], arr[2],
        arr[3], arr[4], arr[5],
        arr[6], arr[7], arr[8]);
}

// 功能：实现 `transformPointHomogeneous` 对应的功能。
static bool transformPointHomogeneous(const cv::Matx33d& H,
                                      const cv::Point2f& in,
                                      cv::Point2f& out) {
    const cv::Vec3d p(static_cast<double>(in.x), static_cast<double>(in.y), 1.0);
    const cv::Vec3d q = H * p;
    if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2])) return false;
    if (std::abs(q[2]) < 1e-12) return false;
    out.x = static_cast<float>(q[0] / q[2]);
    out.y = static_cast<float>(q[1] / q[2]);
    return std::isfinite(out.x) && std::isfinite(out.y);
}

// 功能：实现 `undistortFisheye` 对应的功能。
static bool undistortFisheye(const cv::Mat& src,
                             const CameraModel& cam,
                             double balance,
                             double fov_scale,
                             cv::Mat& undist,
                             cv::Mat& newK) {
    if (src.empty() || cam.K.empty() || cam.D.empty()) return false;

    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        cam.K, cam.D, src.size(), R, newK, balance, src.size(), fov_scale);

    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(
        cam.K, cam.D, R, newK, src.size(), CV_32FC1, map1, map2);

    cv::remap(src, undist, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return !undist.empty();
}

// 功能：实现 `pixelToLonLat` 对应的功能。
static bool pixelToLonLat(const std::array<double, 6>& gt,
                          double px,
                          double py,
                          double& lon,
                          double& lat) {
    lon = gt[0] + gt[1] * px + gt[2] * py;
    lat = gt[3] + gt[4] * px + gt[5] * py;
    return true;
}

// 功能：实现 `lonLatToPixel` 对应的功能。
static bool lonLatToPixel(const std::array<double, 6>& gt,
                          double lon,
                          double lat,
                          double& px,
                          double& py) {
    const double a = gt[1];
    const double b = gt[2];
    const double c = gt[4];
    const double d = gt[5];
    const double det = a * d - b * c;
    if (std::abs(det) < 1e-15) return false;

    const double u = lon - gt[0];
    const double v = lat - gt[3];
    px = (u * d - v * b) / det;
    py = (v * a - u * c) / det;
    return true;
}

// 功能：把 DEM 读取成浮点型高程图。
static cv::Mat readDemAsFloat(const std::string& path) {
    cv::Mat dem = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (dem.empty()) dem = cv::imread(path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
    if (dem.empty()) throw std::runtime_error("Failed to read DEM: " + path);

    cv::Mat dem1;
    if (dem.channels() == 1) {
        dem1 = dem;
    } else if (dem.channels() == 3) {
        cv::cvtColor(dem, dem1, cv::COLOR_BGR2GRAY);
    } else if (dem.channels() == 4) {
        cv::cvtColor(dem, dem1, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported DEM channels: " + std::to_string(dem.channels()));
    }

    cv::Mat demf;
    dem1.convertTo(demf, CV_32F);
    std::cout << "[DEM] " << path << " shape=" << demf.rows << "x" << demf.cols << " depth=CV_32F\n";
    return demf;
}

// 功能：实现 `sampleDemBilinear` 对应的功能。
static bool sampleDemBilinear(const cv::Mat& demf,
                              double x,
                              double y,
                              float nodata,
                              float& alt) {
    if (demf.empty()) return false;
    if (x < 0.0 || y < 0.0 || x > static_cast<double>(demf.cols - 1) || y > static_cast<double>(demf.rows - 1)) return false;

    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, demf.cols - 1);
    int y1 = std::min(y0 + 1, demf.rows - 1);

    float v00 = demf.at<float>(y0, x0);
    float v10 = demf.at<float>(y0, x1);
    float v01 = demf.at<float>(y1, x0);
    float v11 = demf.at<float>(y1, x1);

    auto is_bad = [&](float v) {
        if (!std::isfinite(v)) return true;
        if (std::isfinite(nodata) && std::abs(v - nodata) < 1e-5f) return true;
        return false;
    };

    if (is_bad(v00) || is_bad(v10) || is_bad(v01) || is_bad(v11)) return false;

    double tx = x - static_cast<double>(x0);
    double ty = y - static_cast<double>(y0);
    double v0 = static_cast<double>(v00) * (1.0 - tx) + static_cast<double>(v10) * tx;
    double v1 = static_cast<double>(v01) * (1.0 - tx) + static_cast<double>(v11) * tx;
    alt = static_cast<float>(v0 * (1.0 - ty) + v1 * ty);
    return true;
}

// 功能：把二维地理对应关系写成 CSV。
static bool writeGeo2DCsv(const std::string& path, const std::vector<Geo2DRecord>& rows) {
    ensureParentDir(path);
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    ofs << "cam_x,cam_y,map_x,map_y,lon,lat,score\n";
    ofs << std::fixed << std::setprecision(8);
    for (const auto& r : rows) {
        ofs << r.cam_x << ',' << r.cam_y << ',' << r.map_x << ',' << r.map_y << ','
            << r.lon << ',' << r.lat << ',' << r.score << '\n';
    }
    return true;
}

// 功能：把三维地理对应关系写成 CSV。
static bool writeGeo3DCsv(const std::string& path, const std::vector<Geo3DRecord>& rows) {
    ensureParentDir(path);
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    ofs << "cam_x,cam_y,map_x,map_y,lon,lat,alt,score\n";
    ofs << std::fixed << std::setprecision(8);
    for (const auto& r : rows) {
        ofs << r.m.cam_x << ',' << r.m.cam_y << ',' << r.m.map_x << ',' << r.m.map_y << ','
            << r.m.lon << ',' << r.m.lat << ',' << r.alt << ',' << r.m.score << '\n';
    }
    return true;
}

// 功能：实现 `geodeticToECEF` 对应的功能。
static cv::Vec3d geodeticToECEF(double lon_deg, double lat_deg, double h) {
    const double a = 6378137.0;
    const double f = 1.0 / 298.257223563;
    const double e2 = f * (2.0 - f);

    const double lon = lon_deg * CV_PI / 180.0;
    const double lat = lat_deg * CV_PI / 180.0;
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);

    const double N = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    const double x = (N + h) * cos_lat * cos_lon;
    const double y = (N + h) * cos_lat * sin_lon;
    const double z = (N * (1.0 - e2) + h) * sin_lat;
    return cv::Vec3d(x, y, z);
}

// 功能：把 ECEF 坐标转换回经纬高。
static void ecefToGeodetic(const cv::Vec3d& ecef, double& lon_deg, double& lat_deg, double& h) {
    const double a = 6378137.0;
    const double f = 1.0 / 298.257223563;
    const double b = a * (1.0 - f);
    const double e2 = f * (2.0 - f);
    const double ep2 = (a * a - b * b) / (b * b);

    const double x = ecef[0];
    const double y = ecef[1];
    const double z = ecef[2];
    const double lon = std::atan2(y, x);
    const double p = std::sqrt(x * x + y * y);

    const double theta = std::atan2(z * a, p * b);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double lat = std::atan2(z + ep2 * b * st * st * st, p - e2 * a * ct * ct * ct);

    const double sin_lat = std::sin(lat);
    const double N = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    h = p / std::cos(lat) - N;

    lon_deg = lon * 180.0 / CV_PI;
    lat_deg = lat * 180.0 / CV_PI;
}

// 功能：在指定经纬度处构造 ENU 三个坐标轴。
static void enuAxes(double lon_deg, double lat_deg, cv::Vec3d& east, cv::Vec3d& north, cv::Vec3d& up) {
    const double lon = lon_deg * CV_PI / 180.0;
    const double lat = lat_deg * CV_PI / 180.0;

    east = cv::Vec3d(-std::sin(lon), std::cos(lon), 0.0);
    north = cv::Vec3d(-std::sin(lat) * std::cos(lon), -std::sin(lat) * std::sin(lon), std::cos(lat));
    up = cv::Vec3d(std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat));
}

// 功能：实现 `ecefToEnu` 对应的功能。
static cv::Vec3d ecefToEnu(const cv::Vec3d& p,
                           const cv::Vec3d& p0,
                           const cv::Vec3d& east,
                           const cv::Vec3d& north,
                           const cv::Vec3d& up) {
    const cv::Vec3d d = p - p0;
    return cv::Vec3d(d.dot(east), d.dot(north), d.dot(up));
}

// 功能：实现 `enuToEcef` 对应的功能。
static cv::Vec3d enuToEcef(const cv::Vec3d& enu,
                           const cv::Vec3d& p0,
                           const cv::Vec3d& east,
                           const cv::Vec3d& north,
                           const cv::Vec3d& up) {
    return p0 + east * enu[0] + north * enu[1] + up * enu[2];
}

// 功能：计算一组数的中位数。
static double computeMedian(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double median = values[mid];
    if ((values.size() % 2) == 0) {
        const double upper = median;
        std::nth_element(values.begin(),
                         values.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                         values.begin() + static_cast<std::ptrdiff_t>(mid));
        median = 0.5 * (values[mid - 1] + upper);
    }
    return median;
}

// 功能：计算一组数的分位数。
static double computeQuantile(std::vector<double> values, double q) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double q_clamped = std::min(1.0, std::max(0.0, q));
    const size_t idx = static_cast<size_t>(std::floor(q_clamped * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx), values.end());
    return values[idx];
}

// 功能：求解 2x2 矩阵的逆。
static bool invertMatx22(const cv::Matx22d& M, cv::Matx22d& inv) {
    const double det = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
    if (!std::isfinite(det) || std::abs(det) < 1e-12) return false;
    inv = cv::Matx22d(
        M(1, 1) / det, -M(0, 1) / det,
        -M(1, 0) / det, M(0, 0) / det);
    return true;
}

// 功能：估计 2x2 矩阵的条件数。
static double computeConditionNumber2x2(const cv::Matx22d& J) {
    cv::Mat Jmat = (cv::Mat_<double>(2, 2) << J(0, 0), J(0, 1), J(1, 0), J(1, 1));
    cv::Mat w;
    cv::SVD::compute(Jmat, w);
    if (w.rows < 2) return std::numeric_limits<double>::quiet_NaN();
    const double s_max = w.at<double>(0, 0);
    const double s_min = w.at<double>(1, 0);
    if (!std::isfinite(s_max) || !std::isfinite(s_min) || s_min <= 1e-12) {
        return std::numeric_limits<double>::infinity();
    }
    return s_max / s_min;
}

// 功能：计算 `computeResidualCovariance` 对应的结果。
static bool computeResidualCovariance(const std::vector<cv::Vec2d>& residuals,
                                      const std::vector<double>& weights,
                                      cv::Matx22d& cov,
                                      double& rmse_px) {
    cov = cv::Matx22d::zeros();
    rmse_px = std::numeric_limits<double>::quiet_NaN();
    if (residuals.size() < 2 || residuals.size() != weights.size()) return false;

    std::vector<size_t> valid_idx;
    std::vector<double> norms;
    valid_idx.reserve(residuals.size());
    norms.reserve(residuals.size());
    for (size_t i = 0; i < residuals.size(); ++i) {
        const cv::Vec2d& r = residuals[i];
        const double w = weights[i];
        if (!std::isfinite(r[0]) || !std::isfinite(r[1]) || !std::isfinite(w) || w <= 0.0) continue;
        valid_idx.push_back(i);
        norms.push_back(std::sqrt(r[0] * r[0] + r[1] * r[1]));
    }
    if (valid_idx.size() < 2) return false;

    const double clip_thresh = computeQuantile(norms, 0.9);
    std::vector<size_t> kept_idx;
    kept_idx.reserve(valid_idx.size());
    for (size_t idx : valid_idx) {
        const cv::Vec2d& r = residuals[idx];
        const double norm = std::sqrt(r[0] * r[0] + r[1] * r[1]);
        if (!std::isfinite(clip_thresh) || norm <= clip_thresh + 1e-12) {
            kept_idx.push_back(idx);
        }
    }
    if (kept_idx.size() < 2) kept_idx = valid_idx;

    std::vector<double> us;
    std::vector<double> vs;
    us.reserve(kept_idx.size());
    vs.reserve(kept_idx.size());
    for (size_t idx : kept_idx) {
        us.push_back(residuals[idx][0]);
        vs.push_back(residuals[idx][1]);
    }
    const double median_u = computeMedian(us);
    const double median_v = computeMedian(vs);
    if (!std::isfinite(median_u) || !std::isfinite(median_v)) return false;

    double weight_sum = 0.0;
    double sq_sum = 0.0;
    cv::Matx22d accum = cv::Matx22d::zeros();
    for (size_t idx : kept_idx) {
        const double w = weights[idx];
        const double du = residuals[idx][0] - median_u;
        const double dv = residuals[idx][1] - median_v;
        weight_sum += w;
        sq_sum += w * (residuals[idx][0] * residuals[idx][0] + residuals[idx][1] * residuals[idx][1]);
        accum(0, 0) += w * du * du;
        accum(0, 1) += w * du * dv;
        accum(1, 0) += w * dv * du;
        accum(1, 1) += w * dv * dv;
    }
    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) return false;
    cov = accum * (1.0 / weight_sum);
    rmse_px = std::sqrt(std::max(sq_sum / weight_sum, 0.0));
    return std::isfinite(cov(0, 0)) && std::isfinite(cov(0, 1)) &&
           std::isfinite(cov(1, 0)) && std::isfinite(cov(1, 1)) &&
           std::isfinite(rmse_px);
}

// 功能：实现 `projectEnuPoint` 对应的功能。
static bool projectEnuPoint(const cv::Point3f& obj_pt,
                            const cv::Vec3d& rvec,
                            const cv::Vec3d& tvec,
                            const cv::Mat& camera_K,
                            cv::Point2f& image_pt) {
    std::vector<cv::Point3f> obj{obj_pt};
    std::vector<cv::Point2f> img;
    const cv::Mat dist_zero = cv::Mat::zeros(1, 4, CV_64F);
    cv::projectPoints(obj, rvec, tvec, camera_K, dist_zero, img);
    if (img.empty()) return false;
    image_pt = img.front();
    return std::isfinite(image_pt.x) && std::isfinite(image_pt.y);
}

// 功能：计算 `computeImageToGroundJacobian` 对应的结果。
static bool computeImageToGroundJacobian(const cv::Point3f& ref_obj_pt,
                                         const cv::Vec3d& rvec,
                                         const cv::Vec3d& tvec,
                                         const cv::Mat& camera_K,
                                         cv::Matx22d& J) {
    constexpr double delta_m = 1.0;
    cv::Point2f plus_e;
    cv::Point2f minus_e;
    cv::Point2f plus_n;
    cv::Point2f minus_n;
    if (!projectEnuPoint(cv::Point3f(ref_obj_pt.x + static_cast<float>(delta_m), ref_obj_pt.y, ref_obj_pt.z),
                         rvec, tvec, camera_K, plus_e)) return false;
    if (!projectEnuPoint(cv::Point3f(ref_obj_pt.x - static_cast<float>(delta_m), ref_obj_pt.y, ref_obj_pt.z),
                         rvec, tvec, camera_K, minus_e)) return false;
    if (!projectEnuPoint(cv::Point3f(ref_obj_pt.x, ref_obj_pt.y + static_cast<float>(delta_m), ref_obj_pt.z),
                         rvec, tvec, camera_K, plus_n)) return false;
    if (!projectEnuPoint(cv::Point3f(ref_obj_pt.x, ref_obj_pt.y - static_cast<float>(delta_m), ref_obj_pt.z),
                         rvec, tvec, camera_K, minus_n)) return false;

    const double inv_span = 1.0 / (2.0 * delta_m);
    J = cv::Matx22d(
        (static_cast<double>(plus_e.x) - static_cast<double>(minus_e.x)) * inv_span,
        (static_cast<double>(plus_n.x) - static_cast<double>(minus_n.x)) * inv_span,
        (static_cast<double>(plus_e.y) - static_cast<double>(minus_e.y)) * inv_span,
        (static_cast<double>(plus_n.y) - static_cast<double>(minus_n.y)) * inv_span);
    return std::isfinite(J(0, 0)) && std::isfinite(J(0, 1)) &&
           std::isfinite(J(1, 0)) && std::isfinite(J(1, 1));
}

static DualGeometryStats computeDualGeometryStats(const cv::Matx33d& H01,
                                                  const std::vector<Geo3DRecord>& geo3d,
                                                  const std::vector<cv::Point3f>& obj_pts,
                                                  const std::vector<cv::Point2f>& img_pts,
                                                  const std::vector<int>& pnp_inlier_idx,
                                                  const cv::Vec3d& rvec,
                                                  const cv::Vec3d& tvec,
                                                  const cv::Mat& pnp_camera_K,
                                                  const cv::Matx33d& image1_inverse_affine,
                                                  bool image1_inverse_affine_valid,
                                                  double map_scale_x,
                                                  double map_scale_y,
                                                  double cam_scale_x,
                                                  double cam_scale_y) {
    DualGeometryStats stats;
    stats.shared_count = static_cast<int>(pnp_inlier_idx.size());
    if (pnp_inlier_idx.size() < 4 ||
        geo3d.size() != obj_pts.size() ||
        obj_pts.size() != img_pts.size() ||
        !std::isfinite(map_scale_x) || !std::isfinite(map_scale_y) ||
        !std::isfinite(cam_scale_x) || !std::isfinite(cam_scale_y) ||
        map_scale_x <= 0.0 || map_scale_y <= 0.0 ||
        cam_scale_x <= 0.0 || cam_scale_y <= 0.0) {
        return stats;
    }

    std::vector<cv::Point3f> shared_obj_pts;
    std::vector<cv::Point2f> shared_img_pts;
    std::vector<cv::Point2f> homography_pred_pts;
    std::vector<double> weights;
    shared_obj_pts.reserve(pnp_inlier_idx.size());
    shared_img_pts.reserve(pnp_inlier_idx.size());
    homography_pred_pts.reserve(pnp_inlier_idx.size());
    weights.reserve(pnp_inlier_idx.size());

    for (int idx : pnp_inlier_idx) {
        if (idx < 0) continue;
        const size_t uidx = static_cast<size_t>(idx);
        if (uidx >= geo3d.size() || uidx >= obj_pts.size() || uidx >= img_pts.size()) continue;

        const Geo3DRecord& row = geo3d[uidx];
        cv::Point2f map_match_pt(
            static_cast<float>(static_cast<double>(row.m.map_x) / map_scale_x),
            static_cast<float>(static_cast<double>(row.m.map_y) / map_scale_y));
        cv::Point2f cam_match_pred;
        if (!transformPointHomogeneous(H01, map_match_pt, cam_match_pred)) continue;

        cv::Point2f cam_raw_pred(
            static_cast<float>(static_cast<double>(cam_match_pred.x) * cam_scale_x),
            static_cast<float>(static_cast<double>(cam_match_pred.y) * cam_scale_y));
        cv::Point2f cam_pnp_pred = cam_raw_pred;
        if (image1_inverse_affine_valid &&
            !transformPointHomogeneous(image1_inverse_affine, cam_raw_pred, cam_pnp_pred)) {
            continue;
        }

        shared_obj_pts.push_back(obj_pts[uidx]);
        shared_img_pts.push_back(img_pts[uidx]);
        homography_pred_pts.push_back(cam_pnp_pred);
        weights.push_back(std::max(static_cast<double>(row.m.score), 1e-3));
    }

    stats.shared_count = static_cast<int>(shared_obj_pts.size());
    if (shared_obj_pts.size() < 4) return stats;

    std::vector<cv::Point2f> pnp_pred_pts;
    const cv::Mat dist_zero = cv::Mat::zeros(1, 4, CV_64F);
    cv::projectPoints(shared_obj_pts, rvec, tvec, pnp_camera_K, dist_zero, pnp_pred_pts);
    if (pnp_pred_pts.size() != shared_obj_pts.size()) return stats;

    std::vector<cv::Vec2d> homography_residuals;
    std::vector<cv::Vec2d> pnp_residuals;
    std::vector<cv::Vec2d> disagree_residuals;
    std::vector<double> residual_weights;
    homography_residuals.reserve(shared_obj_pts.size());
    pnp_residuals.reserve(shared_obj_pts.size());
    disagree_residuals.reserve(shared_obj_pts.size());
    residual_weights.reserve(shared_obj_pts.size());

    cv::Point3d ref_sum(0.0, 0.0, 0.0);
    double weight_sum = 0.0;
    for (size_t i = 0; i < shared_obj_pts.size(); ++i) {
        const cv::Point2f& actual = shared_img_pts[i];
        const cv::Point2f& h_pred = homography_pred_pts[i];
        const cv::Point2f& p_pred = pnp_pred_pts[i];
        if (!std::isfinite(actual.x) || !std::isfinite(actual.y) ||
            !std::isfinite(h_pred.x) || !std::isfinite(h_pred.y) ||
            !std::isfinite(p_pred.x) || !std::isfinite(p_pred.y)) {
            continue;
        }

        homography_residuals.emplace_back(
            static_cast<double>(h_pred.x) - static_cast<double>(actual.x),
            static_cast<double>(h_pred.y) - static_cast<double>(actual.y));
        pnp_residuals.emplace_back(
            static_cast<double>(p_pred.x) - static_cast<double>(actual.x),
            static_cast<double>(p_pred.y) - static_cast<double>(actual.y));
        disagree_residuals.emplace_back(
            static_cast<double>(p_pred.x) - static_cast<double>(h_pred.x),
            static_cast<double>(p_pred.y) - static_cast<double>(h_pred.y));

        const double w = weights[i];
        residual_weights.push_back(w);
        ref_sum.x += w * static_cast<double>(shared_obj_pts[i].x);
        ref_sum.y += w * static_cast<double>(shared_obj_pts[i].y);
        ref_sum.z += w * static_cast<double>(shared_obj_pts[i].z);
        weight_sum += w;
    }

    if (homography_residuals.size() < 4 || !(weight_sum > 0.0)) {
        stats.shared_count = static_cast<int>(homography_residuals.size());
        return stats;
    }
    stats.shared_count = static_cast<int>(homography_residuals.size());

    cv::Matx22d sigma_h = cv::Matx22d::zeros();
    cv::Matx22d sigma_p = cv::Matx22d::zeros();
    cv::Matx22d sigma_d = cv::Matx22d::zeros();
    if (!computeResidualCovariance(homography_residuals, residual_weights, sigma_h, stats.homography_rmse_px)) return stats;
    if (!computeResidualCovariance(pnp_residuals, residual_weights, sigma_p, stats.pnp_rmse_px)) return stats;
    if (!computeResidualCovariance(disagree_residuals, residual_weights, sigma_d, stats.disagree_rmse_px)) return stats;

    const cv::Matx22d sigma_px =
        sigma_h * 0.25 +
        sigma_p * 0.25 +
        sigma_d * 0.50 +
        cv::Matx22d(1e-4, 0.0, 0.0, 1e-4);
    stats.px_var_u = sigma_px(0, 0);
    stats.px_cov_uv = sigma_px(0, 1);
    stats.px_var_v = sigma_px(1, 1);

    const cv::Point3f ref_obj_pt(
        static_cast<float>(ref_sum.x / weight_sum),
        static_cast<float>(ref_sum.y / weight_sum),
        static_cast<float>(ref_sum.z / weight_sum));
    cv::Matx22d J = cv::Matx22d::zeros();
    if (!computeImageToGroundJacobian(ref_obj_pt, rvec, tvec, pnp_camera_K, J)) return stats;

    stats.jacobian_11 = J(0, 0);
    stats.jacobian_12 = J(0, 1);
    stats.jacobian_21 = J(1, 0);
    stats.jacobian_22 = J(1, 1);
    stats.jacobian_det = J(0, 0) * J(1, 1) - J(0, 1) * J(1, 0);
    stats.jacobian_cond = computeConditionNumber2x2(J);

    cv::Matx22d pixel_to_ground = cv::Matx22d::zeros();
    const bool well_conditioned =
        std::isfinite(stats.jacobian_cond) && stats.jacobian_cond <= 1e4 &&
        invertMatx22(J, pixel_to_ground);
    if (well_conditioned) {
    } else {
        const cv::Matx22d jtj = J.t() * J + cv::Matx22d(1e-6, 0.0, 0.0, 1e-6);
        cv::Matx22d inv_jtj = cv::Matx22d::zeros();
        if (!invertMatx22(jtj, inv_jtj)) return stats;
        const cv::Matx22d j_plus = inv_jtj * J.t();
        pixel_to_ground = j_plus;
    }

    const cv::Vec2d ground_per_u(pixel_to_ground(0, 0), pixel_to_ground(1, 0));
    const cv::Vec2d ground_per_v(pixel_to_ground(0, 1), pixel_to_ground(1, 1));
    stats.local_scale_u_m_per_px = std::sqrt(ground_per_u.dot(ground_per_u));
    stats.local_scale_v_m_per_px = std::sqrt(ground_per_v.dot(ground_per_v));
    if (std::isfinite(stats.local_scale_u_m_per_px) &&
        std::isfinite(stats.local_scale_v_m_per_px) &&
        stats.local_scale_u_m_per_px > 0.0 &&
        stats.local_scale_v_m_per_px > 0.0) {
        stats.local_scale_mean_m_per_px =
            0.5 * (stats.local_scale_u_m_per_px + stats.local_scale_v_m_per_px);
    }

    const cv::Vec2d x_from_pixel(pixel_to_ground(0, 0), pixel_to_ground(0, 1));
    const cv::Vec2d y_from_pixel(pixel_to_ground(1, 0), pixel_to_ground(1, 1));
    const cv::Vec2d sigma_px_x(
        sigma_px(0, 0) * x_from_pixel[0] + sigma_px(0, 1) * x_from_pixel[1],
        sigma_px(1, 0) * x_from_pixel[0] + sigma_px(1, 1) * x_from_pixel[1]);
    const cv::Vec2d sigma_px_y(
        sigma_px(0, 0) * y_from_pixel[0] + sigma_px(0, 1) * y_from_pixel[1],
        sigma_px(1, 0) * y_from_pixel[0] + sigma_px(1, 1) * y_from_pixel[1]);
    stats.var_x_m2 = x_from_pixel.dot(sigma_px_x);
    stats.var_y_m2 = y_from_pixel.dot(sigma_px_y);
    stats.cov_xy_m2 = std::numeric_limits<double>::quiet_NaN();
    stats.valid =
        std::isfinite(stats.var_x_m2) && std::isfinite(stats.var_y_m2) &&
        stats.var_x_m2 >= 0.0 && stats.var_y_m2 >= 0.0 &&
        std::isfinite(stats.local_scale_mean_m_per_px) && stats.local_scale_mean_m_per_px > 0.0;
    return stats;
}

// 功能：写出 `writePnpReport` 相关结果。
static bool writePnpReport(const std::string& path,
                           bool ok,
                           int map_feature_points,
                           int aerial_feature_points,
                           int lightglue_match_pairs,
                           int homography_used_pairs,
                           int homography_inlier_count,
                           int pnp_input_points,
                           int used_points,
                           int inlier_points,
                           double reproj,
                           double terrain_relief_m,
                           double terrain_alt_std_m,
                           double terrain_alt_median_m,
                           double pnp_f_eff_px,
                           double scale_h_m_per_px,
                           double point_spread_img_ratio,
                           double point_spread_map_xy_m2,
                           double depth_spread_m,
                           double obliqueness_deg,
                           double model_load_ms,
                           double feature_ms,
                           double matching_ms,
                           double pair_match_ms,
                           double homography_ms,
                           double match_vis_ms,
                           double geo2d_ms,
                           double dem_sample_ms,
                           double pnp_ms,
                           const DualGeometryStats& dual_geom_stats,
                           const cv::Vec3d& rvec,
                           const cv::Vec3d& tvec,
                           double cam_lon,
                           double cam_lat,
                           double cam_alt) {
    ensureParentDir(path);
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    ofs << std::fixed << std::setprecision(8);
    ofs << "map_feature_points=" << map_feature_points << "\n";
    ofs << "aerial_feature_points=" << aerial_feature_points << "\n";
    ofs << "lightglue_match_pairs=" << lightglue_match_pairs << "\n";
    ofs << "homography_used_pairs=" << homography_used_pairs << "\n";
    ofs << "homography_inlier_count=" << homography_inlier_count << "\n";
    ofs << "lightglue_geom_inlier_rate=" << safeRatio(homography_inlier_count, homography_used_pairs) << "\n";
    ofs << "pnp_input_points=" << pnp_input_points << "\n";
    ofs << "success=" << (ok ? 1 : 0) << "\n";
    ofs << "used_points=" << used_points << "\n";
    ofs << "inlier_points=" << inlier_points << "\n";
    ofs << "pnp_inlier_rate=" << safeRatio(inlier_points, pnp_input_points) << "\n";
    ofs << "reproj_error=" << reproj << "\n";
    ofs << "terrain_relief_m=" << terrain_relief_m << "\n";
    ofs << "terrain_alt_std_m=" << terrain_alt_std_m << "\n";
    ofs << "terrain_alt_median_m=" << terrain_alt_median_m << "\n";
    ofs << "pnp_f_eff_px=" << pnp_f_eff_px << "\n";
    ofs << "scale_h_m_per_px=" << scale_h_m_per_px << "\n";
    ofs << "point_spread_img_ratio=" << point_spread_img_ratio << "\n";
    ofs << "point_spread_map_xy_m2=" << point_spread_map_xy_m2 << "\n";
    ofs << "depth_spread_m=" << depth_spread_m << "\n";
    ofs << "obliqueness_deg=" << obliqueness_deg << "\n";
    ofs << "model_load_ms=" << model_load_ms << "\n";
    ofs << "feature_ms=" << feature_ms << "\n";
    ofs << "matching_ms=" << matching_ms << "\n";
    ofs << "pair_match_ms=" << pair_match_ms << "\n";
    ofs << "homography_ms=" << homography_ms << "\n";
    ofs << "match_vis_ms=" << match_vis_ms << "\n";
    ofs << "geo2d_ms=" << geo2d_ms << "\n";
    ofs << "dem_sample_ms=" << dem_sample_ms << "\n";
    ofs << "pnp_ms=" << pnp_ms << "\n";
    ofs << "dual_geom_valid=" << (dual_geom_stats.valid ? 1 : 0) << "\n";
    ofs << "dual_geom_shared_count=" << dual_geom_stats.shared_count << "\n";
    ofs << "dual_geom_h_rmse_px=" << dual_geom_stats.homography_rmse_px << "\n";
    ofs << "dual_geom_pnp_rmse_px=" << dual_geom_stats.pnp_rmse_px << "\n";
    ofs << "dual_geom_disagree_rmse_px=" << dual_geom_stats.disagree_rmse_px << "\n";
    ofs << "dual_geom_px_var_u=" << dual_geom_stats.px_var_u << "\n";
    ofs << "dual_geom_px_cov_uv=" << dual_geom_stats.px_cov_uv << "\n";
    ofs << "dual_geom_px_var_v=" << dual_geom_stats.px_var_v << "\n";
    ofs << "dual_geom_j11=" << dual_geom_stats.jacobian_11 << "\n";
    ofs << "dual_geom_j12=" << dual_geom_stats.jacobian_12 << "\n";
    ofs << "dual_geom_j21=" << dual_geom_stats.jacobian_21 << "\n";
    ofs << "dual_geom_j22=" << dual_geom_stats.jacobian_22 << "\n";
    ofs << "dual_geom_j_det=" << dual_geom_stats.jacobian_det << "\n";
    ofs << "dual_geom_j_cond=" << dual_geom_stats.jacobian_cond << "\n";
    ofs << "dual_geom_local_scale_u_m_per_px=" << dual_geom_stats.local_scale_u_m_per_px << "\n";
    ofs << "dual_geom_local_scale_v_m_per_px=" << dual_geom_stats.local_scale_v_m_per_px << "\n";
    ofs << "dual_geom_local_scale_mean_m_per_px=" << dual_geom_stats.local_scale_mean_m_per_px << "\n";
    ofs << "dual_geom_var_x_m2=" << dual_geom_stats.var_x_m2 << "\n";
    ofs << "dual_geom_cov_xy_m2=" << dual_geom_stats.cov_xy_m2 << "\n";
    ofs << "dual_geom_var_y_m2=" << dual_geom_stats.var_y_m2 << "\n";
    if (ok) {
        ofs << "rvec=" << rvec[0] << "," << rvec[1] << "," << rvec[2] << "\n";
        ofs << "tvec=" << tvec[0] << "," << tvec[1] << "," << tvec[2] << "\n";
        ofs << "camera_lon=" << cam_lon << "\n";
        ofs << "camera_lat=" << cam_lat << "\n";
        ofs << "camera_alt=" << cam_alt << "\n";
    }
    return true;
}

// 功能：读取输入图像并做基础合法性检查。
static cv::Mat readInputImage(const std::string& path) {
    cv::Mat img;

    if (hasTiffExt(path)) {
        // TIFF 先按 UNCHANGED 读取，优先保留原始位深与通道，不做隐式量化。
        img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            img = cv::imread(path, cv::IMREAD_ANYDEPTH | cv::IMREAD_ANYCOLOR);
        }
    } else {
        img = cv::imread(path, cv::IMREAD_UNCHANGED);
    }

    if (img.empty()) {
        img = cv::imread(path, cv::IMREAD_UNCHANGED);
    }

    if (img.empty()) {
        throw std::runtime_error("Failed to read image: " + path);
    }

    std::cout << "[Image] " << path << " shape=" << img.rows << "x" << img.cols
              << " channels=" << img.channels() << " depth=" << img.depth() << "\n";
    return img;
}

// 功能：把灰度图稳健归一化到 8 位范围。
static cv::Mat normalizeGrayToU8Robust(const cv::Mat& gray_src) {
    cv::Mat gray32;
    gray_src.convertTo(gray32, CV_32F);

    std::vector<float> vals;
    vals.reserve(gray32.total());

    for (int y = 0; y < gray32.rows; ++y) {
        const float* row = gray32.ptr<float>(y);
        for (int x = 0; x < gray32.cols; ++x) {
            float v = row[x];
            if (std::isfinite(v)) {
                vals.push_back(v);
            }
        }
    }

    if (vals.empty()) {
        return cv::Mat::zeros(gray32.size(), CV_8U);
    }

    std::sort(vals.begin(), vals.end());
    const size_t n = vals.size();
    const size_t id_lo = static_cast<size_t>(0.01 * static_cast<double>(n - 1));
    const size_t id_hi = static_cast<size_t>(0.99 * static_cast<double>(n - 1));

    float lo = vals[id_lo];
    float hi = vals[id_hi];

    if (!(hi > lo)) {
        lo = vals.front();
        hi = vals.back();
    }

    if (!(hi > lo)) {
        return cv::Mat::zeros(gray32.size(), CV_8U);
    }

    const float scale = 255.0f / (hi - lo);
    cv::Mat out8(gray32.size(), CV_8U);

    for (int y = 0; y < gray32.rows; ++y) {
        const float* src_row = gray32.ptr<float>(y);
        uchar* dst_row = out8.ptr<uchar>(y);
        for (int x = 0; x < gray32.cols; ++x) {
            float v = src_row[x];
            if (!std::isfinite(v)) v = lo;
            v = std::max(lo, std::min(hi, v));
            dst_row[x] = static_cast<uchar>(std::round((v - lo) * scale));
        }
    }

    return out8;
}

// 功能：把输入图转成模型需要的 8 位灰度图。
static cv::Mat toGray8ForModel(const cv::Mat& src) {
    cv::Mat gray;
    if (src.channels() == 1) {
        gray = src;
    } else if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported channel count for model input.");
    }

    if (gray.depth() == CV_8U) {
        return gray.clone();
    }

    cv::Mat out8 = normalizeGrayToU8Robust(gray);

    // 遥感影像常出现低对比度区域，做一次轻量 CLAHE 增强可提升特征可检测性。
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(out8, out8);

    return out8;
}

// 功能：把图像转成便于可视化保存的 BGR 8 位图。
static cv::Mat toVisBgr8(const cv::Mat& src) {
    cv::Mat src8;
    if (src.depth() == CV_8U) {
        src8 = src;
    } else {
        cv::Mat gray;
        if (src.channels() == 1) {
            gray = src;
        } else if (src.channels() == 3) {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        } else if (src.channels() == 4) {
            cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
        } else {
            throw std::runtime_error("Unsupported channel count for visualization.");
        }
        src8 = normalizeGrayToU8Robust(gray);
    }

    cv::Mat bgr;
    if (src8.channels() == 1) {
        cv::cvtColor(src8, bgr, cv::COLOR_GRAY2BGR);
    } else if (src8.channels() == 3) {
        bgr = src8.clone();
    } else if (src8.channels() == 4) {
        cv::cvtColor(src8, bgr, cv::COLOR_BGRA2BGR);
    } else {
        throw std::runtime_error("Unsupported channel count for visualization.");
    }
    return bgr;
}

// 功能：把灰度图转换成模型输入需要的 CHW 浮点数组。
static std::vector<float> toFloatCHW01(const cv::Mat& gray8, int H, int W) {
    cv::Mat r, f;
    int interp = cv::INTER_AREA;
    if (gray8.cols < W || gray8.rows < H) {
        interp = cv::INTER_CUBIC;
    }
    cv::resize(gray8, r, cv::Size(W, H), 0, 0, interp);
    r.convertTo(f, CV_32F, 1.0 / 255.0);

    std::vector<float> out(static_cast<size_t>(H) * static_cast<size_t>(W));
    std::memcpy(out.data(), f.ptr<float>(), out.size() * sizeof(float));
    return out;
}

// 功能：把灰度图缩放到特征提取尺寸。
static cv::Mat resizeGrayForFeatures(const cv::Mat& gray8, int H, int W) {
    cv::Mat r;
    int interp = cv::INTER_AREA;
    if (gray8.cols < W || gray8.rows < H) interp = cv::INTER_CUBIC;
    cv::resize(gray8, r, cv::Size(W, H), 0, 0, interp);
    return r;
}

// 功能：按响应值保留最强的一批关键点。
static void trimByTopResponse(std::vector<cv::KeyPoint>& kpts, int max_features) {
    if (max_features <= 0) return;
    if (static_cast<int>(kpts.size()) <= max_features) return;
    std::nth_element(kpts.begin(), kpts.begin() + max_features, kpts.end(),
                     [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                         return a.response > b.response;
                     });
    kpts.resize(static_cast<size_t>(max_features));
}

static void filterClassicalFeaturesByMinDistance(LocalFeatures& f, float min_dist) {
    if (min_dist <= 0.f || f.kpts.size() <= 1) return;

    const bool has_desc = !f.desc.empty() && f.desc.rows == static_cast<int>(f.kpts.size());
    const bool has_score = f.scores.size() == f.kpts.size();
    const float cell = min_dist;
    const float min_d2 = min_dist * min_dist;

    auto cellKey = [](int cx, int cy) -> int64_t {
        return (static_cast<int64_t>(cx) << 32) ^ static_cast<uint32_t>(cy);
    };

    std::vector<int> order(f.kpts.size());
    for (int i = 0; i < static_cast<int>(f.kpts.size()); ++i) order[static_cast<size_t>(i)] = i;

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const float sa = has_score ? f.scores[static_cast<size_t>(a)] : f.kpts[static_cast<size_t>(a)].response;
        const float sb = has_score ? f.scores[static_cast<size_t>(b)] : f.kpts[static_cast<size_t>(b)].response;
        if (sa == sb) return a < b;
        return sa > sb;
    });

    std::unordered_map<int64_t, std::vector<int>> grid;
    std::vector<int> kept;
    kept.reserve(f.kpts.size());

    for (int idx : order) {
        const auto& kp = f.kpts[static_cast<size_t>(idx)];
        const int cx = static_cast<int>(std::floor(kp.pt.x / cell));
        const int cy = static_cast<int>(std::floor(kp.pt.y / cell));

        bool reject = false;
        for (int dx = -1; dx <= 1 && !reject; ++dx) {
            for (int dy = -1; dy <= 1 && !reject; ++dy) {
                auto it = grid.find(cellKey(cx + dx, cy + dy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    const auto& kk = f.kpts[static_cast<size_t>(j)];
                    const float ddx = kp.pt.x - kk.pt.x;
                    const float ddy = kp.pt.y - kk.pt.y;
                    if (ddx * ddx + ddy * ddy < min_d2) {
                        reject = true;
                        break;
                    }
                }
            }
        }

        if (!reject) {
            kept.push_back(idx);
            grid[cellKey(cx, cy)].push_back(idx);
        }
    }

    if (kept.size() == f.kpts.size()) return;

    std::vector<cv::KeyPoint> new_kpts;
    std::vector<float> new_scores;
    cv::Mat new_desc;
    new_kpts.reserve(kept.size());
    new_scores.reserve(kept.size());
    if (has_desc) new_desc.create(static_cast<int>(kept.size()), f.desc.cols, f.desc.type());

    for (size_t i = 0; i < kept.size(); ++i) {
        int idx = kept[i];
        new_kpts.push_back(f.kpts[static_cast<size_t>(idx)]);
        if (has_score) new_scores.push_back(f.scores[static_cast<size_t>(idx)]);
        if (has_desc) {
            f.desc.row(idx).copyTo(new_desc.row(static_cast<int>(i)));
        }
    }

    f.kpts = std::move(new_kpts);
    f.scores = std::move(new_scores);
    if (has_desc) f.desc = std::move(new_desc);
}

// 功能：实现 `runClassicalFeature` 对应的功能。
static LocalFeatures runClassicalFeature(const cv::Mat& gray8,
                                         int H,
                                         int W,
                                         FeatureMethod method,
                                         int max_features,
                                         int fast_threshold,
                                         bool fast_nonmax,
                                         const std::string& tag) {
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat img = resizeGrayForFeatures(gray8, H, W);

    LocalFeatures f;
    if (method == FeatureMethod::SIFT) {
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 4)
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(std::max(0, max_features));
        sift->detectAndCompute(img, cv::noArray(), f.kpts, f.desc, false);
#elif HAS_OPENCV_XFEATURES2D
        cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(std::max(0, max_features));
        sift->detectAndCompute(img, cv::noArray(), f.kpts, f.desc, false);
#else
        throw std::runtime_error("feature.method=sift is not available in current OpenCV build.");
#endif
        f.binary_desc = false;
    } else if (method == FeatureMethod::SURF) {
#if HAS_OPENCV_XFEATURES2D
        cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create(400.0, 4, 3, false, false);
        surf->detectAndCompute(img, cv::noArray(), f.kpts, f.desc, false);
        trimByTopResponse(f.kpts, max_features);
        if (!f.desc.empty() && f.desc.rows != static_cast<int>(f.kpts.size())) {
            cv::Ptr<cv::xfeatures2d::SURF> surf2 = cv::xfeatures2d::SURF::create(400.0, 4, 3, false, false);
            surf2->compute(img, f.kpts, f.desc);
        }
        f.binary_desc = false;
#else
        throw std::runtime_error("feature.method=surf is not available in current OpenCV build (xfeatures2d missing).");
#endif
    } else if (method == FeatureMethod::ORB) {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(std::max(0, max_features));
        orb->detectAndCompute(img, cv::noArray(), f.kpts, f.desc, false);
        f.binary_desc = true;
    } else if (method == FeatureMethod::FAST) {
        cv::FAST(img, f.kpts, fast_threshold, fast_nonmax);
        trimByTopResponse(f.kpts, max_features);
        cv::Ptr<cv::ORB> orb_desc = cv::ORB::create(std::max(500, max_features));
        orb_desc->compute(img, f.kpts, f.desc);
        f.binary_desc = true;
    } else {
        throw std::runtime_error("runClassicalFeature called with unsupported method.");
    }

    f.scores.resize(f.kpts.size());
    for (size_t i = 0; i < f.kpts.size(); ++i) {
        f.scores[i] = f.kpts[i].response;
    }

    if (!f.desc.empty()) {
        if (f.desc.rows != static_cast<int>(f.kpts.size())) {
            throw std::runtime_error("Descriptor rows mismatch keypoints count.");
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[Feature " << tag << "] method=" << featureMethodName(method)
              << " kpts=" << f.kpts.size()
              << " desc=" << (f.desc.empty() ? 0 : f.desc.cols)
              << " type=" << (f.binary_desc ? "binary" : "float")
              << " time=" << ms << " ms\n";
    return f;
}

// 功能：把 SuperPoint 描述子转换成 OpenCV 矩阵。
static cv::Mat superPointDescToMat(const SPFeatures& f) {
    cv::Mat m(static_cast<int>(f.N), static_cast<int>(f.D), CV_32F);
    if (!f.desc.empty() && m.total() == f.desc.size()) {
        std::memcpy(m.ptr<float>(), f.desc.data(), f.desc.size() * sizeof(float));
    }
    return m;
}

// 功能：实现 `runDescriptorMatcher` 对应的功能。
static std::vector<MatchWithScore> runDescriptorMatcher(const cv::Mat& desc0,
                                                        const cv::Mat& desc1,
                                                        MatcherMethod matcher,
                                                        DistanceType dist,
                                                        bool binary_desc,
                                                        bool cross_check,
                                                        float ratio,
                                                        int max_keep) {
    auto t0 = std::chrono::steady_clock::now();
    std::vector<MatchWithScore> out;
    if (desc0.empty() || desc1.empty()) return out;

    DistanceType d = dist;
    if (d == DistanceType::Auto) d = binary_desc ? DistanceType::Hamming : DistanceType::L2;

    cv::Mat a = desc0;
    cv::Mat b = desc1;

    std::vector<cv::DMatch> good;
    if (matcher == MatcherMethod::BF) {
        const int norm_type = (d == DistanceType::Hamming) ? cv::NORM_HAMMING : cv::NORM_L2;
        if (cross_check) {
            cv::BFMatcher bf(norm_type, true);
            bf.match(a, b, good);
        } else {
            cv::BFMatcher bf(norm_type, false);
            std::vector<std::vector<cv::DMatch>> knn;
            bf.knnMatch(a, b, knn, 2);
            for (const auto& v : knn) {
                if (v.size() < 1) continue;
                if (v.size() == 1) {
                    good.push_back(v[0]);
                } else if (v[0].distance < ratio * v[1].distance) {
                    good.push_back(v[0]);
                }
            }
        }
    } else if (matcher == MatcherMethod::FLANN) {
        if (d == DistanceType::L2) {
            if (a.type() != CV_32F) a.convertTo(a, CV_32F);
            if (b.type() != CV_32F) b.convertTo(b, CV_32F);
            cv::FlannBasedMatcher flann(
                cv::makePtr<cv::flann::KDTreeIndexParams>(5),
                cv::makePtr<cv::flann::SearchParams>(64));
            std::vector<std::vector<cv::DMatch>> knn;
            flann.knnMatch(a, b, knn, 2);
            for (const auto& v : knn) {
                if (v.size() < 1) continue;
                if (v.size() == 1) {
                    good.push_back(v[0]);
                } else if (v[0].distance < ratio * v[1].distance) {
                    good.push_back(v[0]);
                }
            }
        } else {
            if (a.type() != CV_8U || b.type() != CV_8U) {
                throw std::runtime_error("FLANN+hamming requires CV_8U binary descriptors.");
            }
            cv::FlannBasedMatcher flann(
                cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2),
                cv::makePtr<cv::flann::SearchParams>(64));
            std::vector<std::vector<cv::DMatch>> knn;
            flann.knnMatch(a, b, knn, 2);
            for (const auto& v : knn) {
                if (v.size() < 1) continue;
                if (v.size() == 1) {
                    good.push_back(v[0]);
                } else if (v[0].distance < ratio * v[1].distance) {
                    good.push_back(v[0]);
                }
            }
        }
    } else {
        throw std::runtime_error("runDescriptorMatcher called with unsupported matcher.");
    }

    std::sort(good.begin(), good.end(), [](const cv::DMatch& a0, const cv::DMatch& b0) {
        return a0.distance < b0.distance;
    });

    if (max_keep > 0 && static_cast<int>(good.size()) > max_keep) {
        good.resize(static_cast<size_t>(max_keep));
    }

    out.reserve(good.size());
    for (const auto& m : good) {
        float s = 1.0f / (1.0f + m.distance);
        out.push_back({m.queryIdx, m.trainIdx, s});
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[Matcher] method=" << matcherMethodName(matcher)
              << " distance=" << distanceTypeName(d)
              << " matches=" << out.size()
              << " time=" << ms << " ms\n";
    return out;
}

// 功能：从 ONNX 输出张量中解析关键点。
static std::vector<cv::KeyPoint> parseKeypoints(const Ort::Value& kp_tensor) {
    auto info = kp_tensor.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape();
    auto et = info.GetElementType();

    int64_t N = 0;
    if (shp.size() == 2 && shp[1] == 2) {
        N = shp[0];
    } else if (shp.size() == 3 && shp[0] == 1 && shp[2] == 2) {
        N = shp[1];
    } else {
        throw std::runtime_error("Unexpected keypoints shape.");
    }

    std::vector<cv::KeyPoint> kpts;
    kpts.reserve(static_cast<size_t>(N));

    auto push_xy = [&](double x, double y) {
        kpts.emplace_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)), 2.f);
    };

    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* p = kp_tensor.GetTensorData<float>();
        for (int64_t i = 0; i < N; ++i) push_xy(p[static_cast<size_t>(i) * 2 + 0], p[static_cast<size_t>(i) * 2 + 1]);
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        const double* p = kp_tensor.GetTensorData<double>();
        for (int64_t i = 0; i < N; ++i) push_xy(p[static_cast<size_t>(i) * 2 + 0], p[static_cast<size_t>(i) * 2 + 1]);
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* p = kp_tensor.GetTensorData<int64_t>();
        for (int64_t i = 0; i < N; ++i) push_xy(static_cast<double>(p[static_cast<size_t>(i) * 2 + 0]), static_cast<double>(p[static_cast<size_t>(i) * 2 + 1]));
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t* p = kp_tensor.GetTensorData<int32_t>();
        for (int64_t i = 0; i < N; ++i) push_xy(static_cast<double>(p[static_cast<size_t>(i) * 2 + 0]), static_cast<double>(p[static_cast<size_t>(i) * 2 + 1]));
    } else {
        throw std::runtime_error("Unsupported keypoints element type.");
    }

    return kpts;
}

// 功能：实现 `runSuperPoint` 对应的功能。
static SPFeatures runSuperPoint(Ort::Session& sp_sess,
                                const cv::Mat& gray8,
                                int H,
                                int W,
                                const std::string& tag) {
    auto t0 = std::chrono::steady_clock::now();

    std::vector<float> input = toFloatCHW01(gray8, H, W);
    std::array<int64_t, 4> in_shape{1, 1, static_cast<int64_t>(H), static_cast<int64_t>(W)};

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), in_shape.data(), in_shape.size());

    const char* in_names[] = {"image"};
    const char* out_names[] = {"keypoints", "scores", "descriptors"};

    auto outs = sp_sess.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 3);

    std::vector<cv::KeyPoint> kpts = parseKeypoints(outs[0]);
    int64_t N = static_cast<int64_t>(kpts.size());

    const auto& desc = outs[2];
    auto d_info = desc.GetTensorTypeAndShapeInfo();
    auto d_shape = d_info.GetShape();
    if (d_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("SuperPoint descriptors must be float32.");
    if (!(d_shape.size() == 3 && d_shape[0] == 1))
        throw std::runtime_error("Unexpected descriptors shape.");

    int64_t Nd = d_shape[1];
    int64_t D = d_shape[2];
    const float* d_ptr = desc.GetTensorData<float>();

    SPFeatures f;
    f.kpts = std::move(kpts);

    const auto& score = outs[1];
    auto s_info = score.GetTensorTypeAndShapeInfo();
    if (s_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const float* s_ptr = score.GetTensorData<float>();
        int64_t Ns = static_cast<int64_t>(s_info.GetElementCount());
        int64_t useN = std::min<int64_t>(N, Ns);
        f.scores.assign(s_ptr, s_ptr + useN);
        if (useN < N) f.scores.resize(static_cast<size_t>(N), 0.f);
    } else {
        f.scores.assign(static_cast<size_t>(N), 0.f);
    }

    f.N = N;
    f.D = D;
    f.desc.assign(d_ptr, d_ptr + static_cast<size_t>(Nd) * static_cast<size_t>(D));

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[SuperPoint " << tag << "] N=" << f.N << " D=" << f.D << " time=" << ms << " ms\n";

    return f;
}

// 功能：按最小距离约束过滤 SuperPoint 特征点。
static void filterFeaturesByMinDistance(SPFeatures& f, float min_dist) {
    if (min_dist <= 0.f || f.N <= 1 || f.D <= 0) return;
    if (f.desc.size() < static_cast<size_t>(f.N) * static_cast<size_t>(f.D)) return;

    const float cell = min_dist;
    const float min_dist2 = min_dist * min_dist;

    auto cellKey = [](int cx, int cy) -> int64_t {
        return (static_cast<int64_t>(cx) << 32) ^ static_cast<uint32_t>(cy);
    };

    std::vector<int> order(static_cast<size_t>(f.N));
    for (int i = 0; i < static_cast<int>(f.N); ++i) order[static_cast<size_t>(i)] = i;

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        float sa = (static_cast<size_t>(a) < f.scores.size()) ? f.scores[static_cast<size_t>(a)] : 0.f;
        float sb = (static_cast<size_t>(b) < f.scores.size()) ? f.scores[static_cast<size_t>(b)] : 0.f;
        if (sa == sb) return a < b;
        return sa > sb;
    });

    std::unordered_map<int64_t, std::vector<int>> grid;
    std::vector<int> kept;
    kept.reserve(static_cast<size_t>(f.N));

    for (int idx : order) {
        const auto& kp = f.kpts[static_cast<size_t>(idx)];
        int cx = static_cast<int>(std::floor(kp.pt.x / cell));
        int cy = static_cast<int>(std::floor(kp.pt.y / cell));

        bool reject = false;
        for (int dx = -1; dx <= 1 && !reject; ++dx) {
            for (int dy = -1; dy <= 1 && !reject; ++dy) {
                auto it = grid.find(cellKey(cx + dx, cy + dy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    const auto& pk = f.kpts[static_cast<size_t>(j)];
                    float ddx = kp.pt.x - pk.pt.x;
                    float ddy = kp.pt.y - pk.pt.y;
                    if (ddx * ddx + ddy * ddy < min_dist2) {
                        reject = true;
                        break;
                    }
                }
            }
        }

        if (!reject) {
            kept.push_back(idx);
            grid[cellKey(cx, cy)].push_back(idx);
        }
    }

    if (kept.size() == static_cast<size_t>(f.N)) return;

    std::vector<cv::KeyPoint> new_kpts;
    std::vector<float> new_scores;
    std::vector<float> new_desc;

    new_kpts.reserve(kept.size());
    new_scores.reserve(kept.size());
    new_desc.reserve(kept.size() * static_cast<size_t>(f.D));

    for (int idx : kept) {
        new_kpts.push_back(f.kpts[static_cast<size_t>(idx)]);
        new_scores.push_back((static_cast<size_t>(idx) < f.scores.size()) ? f.scores[static_cast<size_t>(idx)] : 0.f);

        const float* dptr = f.desc.data() + static_cast<size_t>(idx) * static_cast<size_t>(f.D);
        new_desc.insert(new_desc.end(), dptr, dptr + static_cast<size_t>(f.D));
    }

    int64_t oldN = f.N;
    f.kpts = std::move(new_kpts);
    f.scores = std::move(new_scores);
    f.desc = std::move(new_desc);
    f.N = static_cast<int64_t>(f.kpts.size());

    std::cout << "[Filter] min_kpt_dist=" << min_dist << " keep " << f.N << "/" << oldN << "\n";
}

enum class KptCoordMode { Pixel = 0, ZeroOne = 1, MinusOneOne = 2 };

// 功能：把关键点坐标模式枚举转成字符串。
static const char* coordModeName(KptCoordMode mode) {
    switch (mode) {
        case KptCoordMode::Pixel: return "pixel";
        case KptCoordMode::ZeroOne: return "zero_one";
        case KptCoordMode::MinusOneOne: return "minus_one_one";
    }
    return "unknown";
}

// 功能：实现 `fillKeypointTensor` 对应的功能。
static void fillKeypointTensor(const std::vector<cv::KeyPoint>& kpts,
                               int64_t N,
                               int W,
                               int H,
                               KptCoordMode mode,
                               std::vector<float>& out) {
    out.resize(static_cast<size_t>(N) * 2);

    const float w = std::max(1, W - 1);
    const float h = std::max(1, H - 1);

    for (int64_t i = 0; i < N; ++i) {
        float x = kpts[static_cast<size_t>(i)].pt.x;
        float y = kpts[static_cast<size_t>(i)].pt.y;

        if (mode == KptCoordMode::ZeroOne) {
            x = x / w;
            y = y / h;
        } else if (mode == KptCoordMode::MinusOneOne) {
            x = x / w * 2.f - 1.f;
            y = y / h * 2.f - 1.f;
        }

        out[static_cast<size_t>(i) * 2 + 0] = x;
        out[static_cast<size_t>(i) * 2 + 1] = y;
    }
}

// 功能：实现 `runLightGlueOnce` 对应的功能。
static std::vector<MatchWithScore> runLightGlueOnce(Ort::Session& lg_sess,
                                                     const SPFeatures& f0,
                                                     const SPFeatures& f1,
                                                     float mscore_thresh,
                                                     int W,
                                                     int H,
                                                     KptCoordMode mode) {
    std::vector<float> k0, k1;
    fillKeypointTensor(f0.kpts, f0.N, W, H, mode, k0);
    fillKeypointTensor(f1.kpts, f1.N, W, H, mode, k1);

    std::array<int64_t, 3> k0_shape{1, f0.N, 2};
    std::array<int64_t, 3> k1_shape{1, f1.N, 2};
    std::array<int64_t, 3> d0_shape{1, f0.N, f0.D};
    std::array<int64_t, 3> d1_shape{1, f1.N, f1.D};

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value k0_t = Ort::Value::CreateTensor<float>(mem, k0.data(), k0.size(), k0_shape.data(), k0_shape.size());
    Ort::Value k1_t = Ort::Value::CreateTensor<float>(mem, k1.data(), k1.size(), k1_shape.data(), k1_shape.size());
    Ort::Value d0_t = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(f0.desc.data()), f0.desc.size(), d0_shape.data(), d0_shape.size());
    Ort::Value d1_t = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(f1.desc.data()), f1.desc.size(), d1_shape.data(), d1_shape.size());

    const char* in_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
    const char* out_names[] = {"matches0", "mscores0"};

    std::array<Ort::Value, 4> in_tensors = {std::move(k0_t), std::move(k1_t), std::move(d0_t), std::move(d1_t)};
    auto outs = lg_sess.Run(Ort::RunOptions{nullptr}, in_names, in_tensors.data(), in_tensors.size(), out_names, 2);

    const auto& m0 = outs[0];
    auto m0_info = m0.GetTensorTypeAndShapeInfo();
    auto m0_shape = m0_info.GetShape();
    if (m0_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        throw std::runtime_error("LightGlue matches0 must be int64.");
    if (!(m0_shape.size() == 2 && m0_shape[1] == 2))
        throw std::runtime_error("Unexpected matches0 shape.");

    int64_t M = m0_shape[0];
    const int64_t* mptr = m0.GetTensorData<int64_t>();

    const auto& sc0 = outs[1];
    auto sc_info = sc0.GetTensorTypeAndShapeInfo();
    auto sc_shape = sc_info.GetShape();
    if (sc_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("LightGlue mscores0 must be float32.");
    if (!((sc_shape.size() == 1 && sc_shape[0] == M) || (sc_shape.size() == 2 && sc_shape[0] == 1 && sc_shape[1] == M)))
        throw std::runtime_error("Unexpected mscores0 shape.");

    const float* sptr = sc0.GetTensorData<float>();

    std::vector<MatchWithScore> pairs;
    pairs.reserve(static_cast<size_t>(M));
    float smin = std::numeric_limits<float>::infinity();
    float smax = -std::numeric_limits<float>::infinity();
    int64_t valid_pairs = 0;

    for (int64_t i = 0; i < M; ++i) {
        int64_t a = mptr[static_cast<size_t>(i) * 2 + 0];
        int64_t b = mptr[static_cast<size_t>(i) * 2 + 1];
        if (a < 0 || b < 0 || a >= f0.N || b >= f1.N) continue;

        float s = sptr[i];
        smin = std::min(smin, s);
        smax = std::max(smax, s);
        ++valid_pairs;

        if (mscore_thresh >= 0.f && s < mscore_thresh) continue;
        pairs.push_back({static_cast<int>(a), static_cast<int>(b), s});
    }

    if (valid_pairs == 0) {
        smin = 0.f;
        smax = 0.f;
    }

    std::cout << "[LightGlue attempt] mode=" << coordModeName(mode)
              << " raw_valid=" << valid_pairs
              << " keep=" << pairs.size()
              << " smin=" << smin
              << " smax=" << smax
              << " thresh=" << mscore_thresh << "\n";

    return pairs;
}

// 功能：实现 `runLightGlue` 对应的功能。
static std::vector<MatchWithScore> runLightGlue(Ort::Session& lg_sess,
                                                 const SPFeatures& f0,
                                                 const SPFeatures& f1,
                                                 float mscore_thresh,
                                                 int W,
                                                 int H) {
    auto t0 = std::chrono::steady_clock::now();
    const int min_pairs_for_early_accept = 100;

    std::vector<MatchWithScore> best = runLightGlueOnce(
        lg_sess, f0, f1, mscore_thresh, W, H, KptCoordMode::Pixel);

    if (static_cast<int>(best.size()) >= min_pairs_for_early_accept) {
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[LightGlue] mode=pixel pairs=" << best.size() << " time=" << ms << " ms\n";
        return best;
    }

    if (!best.empty()) {
        std::cout << "[LightGlue] pixel mode returned only " << best.size()
                  << " pairs, continue fallback trials.\n";
    }

    // 当默认坐标约定无匹配时，自动尝试常见坐标归一化约定并放宽阈值。
    std::vector<std::pair<KptCoordMode, std::vector<MatchWithScore>>> tries;
    tries.push_back({KptCoordMode::Pixel, runLightGlueOnce(lg_sess, f0, f1, -1.f, W, H, KptCoordMode::Pixel)});
    tries.push_back({KptCoordMode::ZeroOne, runLightGlueOnce(lg_sess, f0, f1, -1.f, W, H, KptCoordMode::ZeroOne)});
    tries.push_back({KptCoordMode::MinusOneOne, runLightGlueOnce(lg_sess, f0, f1, -1.f, W, H, KptCoordMode::MinusOneOne)});

    auto best_it = std::max_element(tries.begin(), tries.end(), [](const auto& a, const auto& b) {
        if (a.second.size() != b.second.size()) return a.second.size() < b.second.size();
        float as = a.second.empty() ? -1e9f : a.second.front().s;
        float bs = b.second.empty() ? -1e9f : b.second.front().s;
        return as < bs;
    });

    best = (best_it != tries.end()) ? best_it->second : std::vector<MatchWithScore>{};

    std::sort(best.begin(), best.end(), [](const MatchWithScore& a, const MatchWithScore& b) { return a.s > b.s; });

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (best_it != tries.end()) {
        std::cout << "[LightGlue fallback] choose mode=" << coordModeName(best_it->first)
                  << " pairs=" << best.size() << " time=" << ms << " ms\n";
    } else {
        std::cout << "[LightGlue fallback] no valid mode, time=" << ms << " ms\n";
    }

    return best;
}

// 功能：实现 `drawMatchesCustom` 对应的功能。
static cv::Mat drawMatchesCustom(const cv::Mat& left,
                                 const cv::Mat& right,
                                 const std::vector<cv::KeyPoint>& k0,
                                 const std::vector<cv::KeyPoint>& k1,
                                 const std::vector<MatchWithScore>& pairs,
                                 int max_draw) {
    cv::Mat L = left.clone();
    cv::Mat R = right.clone();
    if (L.channels() == 1) cv::cvtColor(L, L, cv::COLOR_GRAY2BGR);
    if (R.channels() == 1) cv::cvtColor(R, R, cv::COLOR_GRAY2BGR);

    cv::Mat canvas;
    cv::hconcat(L, R, canvas);
    const int offset = L.cols;

    const int draw_n = std::min<int>(max_draw, static_cast<int>(pairs.size()));
    for (int i = 0; i < draw_n; ++i) {
        const auto& m = pairs[static_cast<size_t>(i)];
        cv::Point2f p0 = k0[static_cast<size_t>(m.q)].pt;
        cv::Point2f p1 = k1[static_cast<size_t>(m.t)].pt + cv::Point2f(static_cast<float>(offset), 0.f);

        cv::line(canvas, p0, p1, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::circle(canvas, p0, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::circle(canvas, p1, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    cv::putText(canvas,
                "draw " + std::to_string(draw_n) + " matches",
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 255),
                2);
    return canvas;
}

// 功能：实现 `estimateHomographyFromMatches` 对应的功能。
static bool estimateHomographyFromMatches(const std::vector<MatchWithScore>& pairs,
                                          const std::vector<cv::KeyPoint>& k0,
                                          const std::vector<cv::KeyPoint>& k1,
                                          double reproj_thresh,
                                          int max_pairs,
                                          cv::Mat& H01,
                                          std::vector<uchar>& inlier_mask,
                                          int& used_pairs,
                                          int& inlier_count) {
    H01.release();
    inlier_mask.clear();
    used_pairs = 0;
    inlier_count = 0;

    if (pairs.size() < 4) return false;

    const int capped = std::max(4, max_pairs);
    used_pairs = std::min<int>(static_cast<int>(pairs.size()), capped);

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    src_pts.reserve(static_cast<size_t>(used_pairs));
    dst_pts.reserve(static_cast<size_t>(used_pairs));

    for (int i = 0; i < used_pairs; ++i) {
        const auto& m = pairs[static_cast<size_t>(i)];
        if (m.q < 0 || m.t < 0) continue;
        if (static_cast<size_t>(m.q) >= k0.size() || static_cast<size_t>(m.t) >= k1.size()) continue;
        src_pts.push_back(k0[static_cast<size_t>(m.q)].pt);
        dst_pts.push_back(k1[static_cast<size_t>(m.t)].pt);
    }

    if (src_pts.size() < 4 || dst_pts.size() < 4) return false;

    H01 = cv::findHomography(src_pts, dst_pts, cv::RANSAC, reproj_thresh, inlier_mask, 2000, 0.995);
    if (H01.empty()) return false;

    inlier_count = static_cast<int>(std::count(inlier_mask.begin(), inlier_mask.end(), static_cast<uchar>(1)));
    if (inlier_count < 4) return false;

    used_pairs = static_cast<int>(inlier_mask.size());
    return true;
}

// 功能：生成 `makeHomographyCompareCanvas` 对应的结果。
static cv::Mat makeHomographyCompareCanvas(const cv::Mat& target_bgr,
                                           const cv::Mat& warped_bgr,
                                           int inlier_count,
                                           int used_pairs,
                                           bool homography_ok) {
    cv::Mat target = target_bgr.clone();
    cv::Mat warped = warped_bgr.clone();

    if (target.channels() == 1) cv::cvtColor(target, target, cv::COLOR_GRAY2BGR);
    if (warped.channels() == 1) cv::cvtColor(warped, warped, cv::COLOR_GRAY2BGR);

    if (warped.size() != target.size()) {
        cv::resize(warped, warped, target.size());
    }

    cv::Mat blend;
    cv::addWeighted(target, 0.5, warped, 0.5, 0.0, blend);

    cv::Mat diff;
    cv::absdiff(target, warped, diff);

    cv::Mat top_row;
    cv::Mat bottom_row;
    cv::hconcat(target, warped, top_row);
    cv::hconcat(blend, diff, bottom_row);

    cv::Mat canvas;
    cv::vconcat(top_row, bottom_row, canvas);

    std::string title = homography_ok ? "Homography OK" : "Homography Failed";
    std::string stat = "inliers=" + std::to_string(inlier_count) + "/" + std::to_string(used_pairs);
    cv::putText(canvas, title + "  " + stat, cv::Point(20, 36), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

    cv::putText(canvas, "TL: target  TR: warped", cv::Point(20, top_row.rows - 14), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    cv::putText(canvas, "BL: blend   BR: absdiff", cv::Point(20, canvas.rows - 14), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

    return canvas;
}

// 功能：尝试为 ONNX Runtime 会话启用 CUDA。
static bool tryEnableCuda(Ort::SessionOptions& opt, int device_id) {
    try {
        const OrtApi& api = Ort::GetApi();

        char** providers = nullptr;
        int len = 0;
        OrtStatus* st0 = api.GetAvailableProviders(&providers, &len);
        if (st0 != nullptr) {
            api.ReleaseStatus(st0);
            return false;
        }

        bool has_cuda = false;
        for (int i = 0; i < len; ++i) {
            if (providers[i] && std::string(providers[i]) == "CUDAExecutionProvider") {
                has_cuda = true;
                break;
            }
        }
        OrtStatus* st_release = api.ReleaseAvailableProviders(providers, len);
        if (st_release != nullptr) {
            api.ReleaseStatus(st_release);
        }

        if (!has_cuda) {
            std::cout << "[EP] CUDAExecutionProvider not available, fallback CPU.\n";
            return false;
        }

#if defined(ORT_API_VERSION) && (ORT_API_VERSION >= 14)
        {
            OrtCUDAProviderOptionsV2* cuda_options = nullptr;
            OrtStatus* st_create = api.CreateCUDAProviderOptions(&cuda_options);
            if (st_create != nullptr) {
                api.ReleaseStatus(st_create);
                cuda_options = nullptr;
            }

            if (cuda_options) {
                const char* keys[] = {"device_id"};
                std::string dev = std::to_string(device_id);
                const char* values[] = {dev.c_str()};
                OrtStatus* st_upd = api.UpdateCUDAProviderOptions(cuda_options, keys, values, 1);
                if (st_upd != nullptr) {
                    api.ReleaseStatus(st_upd);
                    api.ReleaseCUDAProviderOptions(cuda_options);
                    cuda_options = nullptr;
                }
            }

            if (cuda_options) {
                OrtStatus* st_append = api.SessionOptionsAppendExecutionProvider_CUDA_V2((OrtSessionOptions*)opt, cuda_options);
                api.ReleaseCUDAProviderOptions(cuda_options);
                if (st_append == nullptr) {
                    std::cout << "[EP] CUDA enabled by CUDA_V2, device_id=" << device_id << "\n";
                    return true;
                }
                api.ReleaseStatus(st_append);
            }
        }
#endif

        {
            OrtCUDAProviderOptions cuda_options;
            std::memset(&cuda_options, 0, sizeof(cuda_options));
            cuda_options.device_id = device_id;

            OrtStatus* st_append = api.SessionOptionsAppendExecutionProvider_CUDA((OrtSessionOptions*)opt, &cuda_options);
            if (st_append != nullptr) {
                api.ReleaseStatus(st_append);
                return false;
            }

            std::cout << "[EP] CUDA enabled by legacy options, device_id=" << device_id << "\n";
            return true;
        }
    } catch (...) {
        std::cout << "[EP] enable CUDA failed, fallback CPU.\n";
        return false;
    }
}

struct RuntimeSessions {
    Ort::Session* sp_sess = nullptr;
    Ort::Session* lg_sess = nullptr;
    bool use_cuda = false;
    bool cache_hit = false;
    double model_load_ms = 0.0;
};

// 功能：实现 `acquireRuntimeSessions` 对应的功能。
static RuntimeSessions acquireRuntimeSessions(const AppConfig& cfg,
                                              FeatureMethod feat_method,
                                              MatcherMethod matcher_method) {
    RuntimeSessions out;

    const bool need_sp = (feat_method == FeatureMethod::SuperPoint || matcher_method == MatcherMethod::LightGlue);
    const bool need_lg = (matcher_method == MatcherMethod::LightGlue);
    if (!need_sp && !need_lg) {
        return out;
    }

    static std::unique_ptr<Ort::Env> env;
    static std::unique_ptr<Ort::Session> sp_sess;
    static std::unique_ptr<Ort::Session> lg_sess;
    static std::string cache_key;
    static bool cache_use_cuda = false;

    std::ostringstream key_ss;
    key_ss << "sp=" << cfg.superpoint_onnx
           << "|lg=" << cfg.lightglue_onnx
           << "|try_cuda=" << (cfg.try_cuda ? 1 : 0)
           << "|device_id=" << cfg.device_id
           << "|need_sp=" << (need_sp ? 1 : 0)
           << "|need_lg=" << (need_lg ? 1 : 0);
    const std::string desired_key = key_ss.str();

    const bool cache_valid =
        (cache_key == desired_key) &&
        (!need_sp || sp_sess != nullptr) &&
        (!need_lg || lg_sess != nullptr);

    if (!cache_valid) {
        const auto model_load_t0 = std::chrono::steady_clock::now();

        if (!env) {
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sp_lg_cfg");
        }

        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        bool use_cuda = false;
        if (cfg.try_cuda) {
            use_cuda = tryEnableCuda(opt, cfg.device_id);
        }

        std::unique_ptr<Ort::Session> new_sp_sess;
        std::unique_ptr<Ort::Session> new_lg_sess;
        if (need_sp) {
            new_sp_sess = std::make_unique<Ort::Session>(*env, cfg.superpoint_onnx.c_str(), opt);
        }
        if (need_lg) {
            new_lg_sess = std::make_unique<Ort::Session>(*env, cfg.lightglue_onnx.c_str(), opt);
        }

        sp_sess = std::move(new_sp_sess);
        lg_sess = std::move(new_lg_sess);
        cache_key = desired_key;
        cache_use_cuda = use_cuda;

        const auto model_load_t1 = std::chrono::steady_clock::now();
        out.model_load_ms =
            std::chrono::duration<double, std::milli>(model_load_t1 - model_load_t0).count();
        out.cache_hit = false;
    } else {
        out.model_load_ms = 0.0;
        out.cache_hit = true;
    }

    out.sp_sess = sp_sess.get();
    out.lg_sess = lg_sess.get();
    out.use_cuda = cache_use_cuda;
    return out;
}

// 功能：从配置节点读取必填字符串。
static std::string readRequiredString(const cv::FileNode& node, const char* key) {
    cv::FileNode v = node[key];
    if (v.empty()) {
        throw std::runtime_error(std::string("Missing config key: ") + key);
    }
    std::string s = static_cast<std::string>(v);
    if (s.empty()) {
        throw std::runtime_error(std::string("Empty config value: ") + key);
    }
    return s;
}

// 功能：读取单帧 `sp_lg` 运行配置。
static AppConfig loadConfig(const std::string& cfg_path_input) {
    std::filesystem::path cfg_path = std::filesystem::absolute(std::filesystem::path(cfg_path_input));
    cv::FileStorage fs(cfg_path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open config file: " + cfg_path.string());
    }

    cv::FileNode in_node = fs["input"];
    cv::FileNode model_node = fs["model"];
    cv::FileNode out_node = fs["output"];
    cv::FileNode runtime_node = fs["runtime"];
    cv::FileNode camera_node = fs["camera"];
    cv::FileNode geo_node = fs["geo"];
    cv::FileNode feature_node = fs["feature"];
    cv::FileNode matching_node = fs["matching"];

    if (in_node.empty() || model_node.empty() || out_node.empty()) {
        throw std::runtime_error("Config must contain input/model/output sections.");
    }

    AppConfig cfg;

    cfg.image0 = readRequiredString(in_node, "image0");
    cfg.image1 = readRequiredString(in_node, "image1");

    cfg.superpoint_onnx = readRequiredString(model_node, "superpoint");
    cfg.lightglue_onnx = readRequiredString(model_node, "lightglue");

    if (!out_node["sp0_kpts"].empty()) cfg.sp0_kpts_out = static_cast<std::string>(out_node["sp0_kpts"]);
    if (!out_node["sp1_kpts"].empty()) cfg.sp1_kpts_out = static_cast<std::string>(out_node["sp1_kpts"]);
    if (!out_node["matches"].empty()) cfg.matches_out = static_cast<std::string>(out_node["matches"]);
    if (!out_node["matches_inlier"].empty()) cfg.matches_inlier_out = static_cast<std::string>(out_node["matches_inlier"]);
    if (!out_node["warp"].empty()) cfg.warp_out = static_cast<std::string>(out_node["warp"]);
    if (!out_node["compare"].empty()) cfg.compare_out = static_cast<std::string>(out_node["compare"]);
    if (!out_node["undistort"].empty()) cfg.undistort_out = static_cast<std::string>(out_node["undistort"]);
    if (!out_node["map2d_csv"].empty()) cfg.map2d_csv_out = static_cast<std::string>(out_node["map2d_csv"]);
    if (!out_node["map3d_csv"].empty()) cfg.map3d_csv_out = static_cast<std::string>(out_node["map3d_csv"]);
    if (!out_node["pnp_report"].empty()) cfg.pnp_report_out = static_cast<std::string>(out_node["pnp_report"]);

    if (!in_node["dem_tif"].empty()) cfg.dem_tif = static_cast<std::string>(in_node["dem_tif"]);

    if (!runtime_node.empty()) {
        if (!runtime_node["H"].empty()) cfg.H = static_cast<int>(runtime_node["H"]);
        if (!runtime_node["W"].empty()) cfg.W = static_cast<int>(runtime_node["W"]);
        if (!runtime_node["mscore_thresh"].empty()) cfg.mscore_thresh = static_cast<float>(runtime_node["mscore_thresh"]);
        if (!runtime_node["min_kpt_dist"].empty()) cfg.min_kpt_dist = static_cast<float>(runtime_node["min_kpt_dist"]);
        if (!runtime_node["max_draw"].empty()) cfg.max_draw = static_cast<int>(runtime_node["max_draw"]);
        if (!runtime_node["homography_ransac_reproj"].empty()) cfg.homography_ransac_reproj = static_cast<double>(runtime_node["homography_ransac_reproj"]);
        if (!runtime_node["homography_max_pairs"].empty()) cfg.homography_max_pairs = static_cast<int>(runtime_node["homography_max_pairs"]);
        if (!runtime_node["device_id"].empty()) cfg.device_id = static_cast<int>(runtime_node["device_id"]);
        if (!runtime_node["try_cuda"].empty()) cfg.try_cuda = static_cast<int>(runtime_node["try_cuda"]) != 0;
    }

    if (!feature_node.empty()) {
        if (!feature_node["method"].empty()) cfg.feature_method = static_cast<std::string>(feature_node["method"]);
        if (!feature_node["max_features"].empty()) cfg.max_features = static_cast<int>(feature_node["max_features"]);
        if (!feature_node["fast_threshold"].empty()) cfg.fast_threshold = static_cast<int>(feature_node["fast_threshold"]);
        if (!feature_node["fast_nonmax"].empty()) cfg.fast_nonmax = static_cast<int>(feature_node["fast_nonmax"]) != 0;
    }

    if (!matching_node.empty()) {
        if (!matching_node["method"].empty()) cfg.matcher_method = static_cast<std::string>(matching_node["method"]);
        if (!matching_node["distance"].empty()) cfg.matcher_distance = static_cast<std::string>(matching_node["distance"]);
        if (!matching_node["ratio"].empty()) cfg.matcher_ratio = static_cast<float>(matching_node["ratio"]);
        if (!matching_node["cross_check"].empty()) cfg.matcher_cross_check = static_cast<int>(matching_node["cross_check"]) != 0;
        if (!matching_node["max_keep"].empty()) cfg.matcher_max_keep = static_cast<int>(matching_node["max_keep"]);
    }

    if (!camera_node.empty()) {
        if (!camera_node["use_fisheye_undistort"].empty()) cfg.use_fisheye_undistort = static_cast<int>(camera_node["use_fisheye_undistort"]) != 0;
        if (!camera_node["undistort_balance"].empty()) cfg.undistort_balance = static_cast<double>(camera_node["undistort_balance"]);
        if (!camera_node["undistort_fov_scale"].empty()) cfg.undistort_fov_scale = static_cast<double>(camera_node["undistort_fov_scale"]);
        if (!camera_node["intrinsics_file"].empty()) cfg.camera_intrinsics_file = static_cast<std::string>(camera_node["intrinsics_file"]);

        if (!camera_node["fx"].empty()) cfg.fx = static_cast<double>(camera_node["fx"]);
        if (!camera_node["fy"].empty()) cfg.fy = static_cast<double>(camera_node["fy"]);
        if (!camera_node["cx"].empty()) cfg.cx = static_cast<double>(camera_node["cx"]);
        if (!camera_node["cy"].empty()) cfg.cy = static_cast<double>(camera_node["cy"]);

        std::array<double, 4> d4;
        if (readArrayFromNode(camera_node["distortion_coeffs"], d4)) cfg.fish_distortion = d4;
        std::array<double, 9> k9;
        if (readArrayFromNode(camera_node["camera_matrix"], k9)) {
            cfg.camera_matrix = k9;
            cfg.camera_matrix_valid = true;
        }
        if (readArrayFromNode(camera_node["pnp_camera_matrix"], k9)) {
            cfg.pnp_camera_matrix = k9;
            cfg.pnp_camera_matrix_valid = true;
        }
        if (readArrayFromNode(camera_node["image1_inverse_affine"], k9)) {
            cfg.image1_inverse_affine = k9;
            cfg.image1_inverse_affine_valid = true;
        }
    }

    if (!geo_node.empty()) {
        bool map_valid_from_cfg = true;
        bool dem_valid_from_cfg = true;
        if (!geo_node["map_affine_valid"].empty()) map_valid_from_cfg = static_cast<int>(geo_node["map_affine_valid"]) != 0;
        if (!geo_node["dem_affine_valid"].empty()) dem_valid_from_cfg = static_cast<int>(geo_node["dem_affine_valid"]) != 0;

        std::array<double, 6> a6;
        if (readArrayFromNode(geo_node["map_wgs84_affine"], a6)) {
            cfg.map_wgs84_affine = a6;
            cfg.map_affine_valid = map_valid_from_cfg;
        }
        if (readArrayFromNode(geo_node["dem_wgs84_affine"], a6)) {
            cfg.dem_wgs84_affine = a6;
            cfg.dem_affine_valid = dem_valid_from_cfg;
        }
        if (!geo_node["dem_nodata"].empty()) cfg.dem_nodata = static_cast<float>(geo_node["dem_nodata"]);
    }

    if (cfg.H <= 0 || cfg.W <= 0) {
        throw std::runtime_error("H/W must be positive.");
    }
    if (cfg.max_draw <= 0) {
        cfg.max_draw = 200;
    }

    std::filesystem::path base = cfg_path.parent_path();
    cfg.image0 = resolvePath(base, cfg.image0).string();
    cfg.image1 = resolvePath(base, cfg.image1).string();
    cfg.superpoint_onnx = resolvePath(base, cfg.superpoint_onnx).string();
    cfg.lightglue_onnx = resolvePath(base, cfg.lightglue_onnx).string();
    if (!cfg.sp0_kpts_out.empty()) cfg.sp0_kpts_out = resolvePath(base, cfg.sp0_kpts_out).string();
    if (!cfg.sp1_kpts_out.empty()) cfg.sp1_kpts_out = resolvePath(base, cfg.sp1_kpts_out).string();
    if (!cfg.matches_out.empty()) cfg.matches_out = resolvePath(base, cfg.matches_out).string();
    if (!cfg.matches_inlier_out.empty()) cfg.matches_inlier_out = resolvePath(base, cfg.matches_inlier_out).string();
    if (!cfg.warp_out.empty()) cfg.warp_out = resolvePath(base, cfg.warp_out).string();
    if (!cfg.compare_out.empty()) cfg.compare_out = resolvePath(base, cfg.compare_out).string();
    if (!cfg.undistort_out.empty()) cfg.undistort_out = resolvePath(base, cfg.undistort_out).string();
    if (!cfg.map2d_csv_out.empty()) cfg.map2d_csv_out = resolvePath(base, cfg.map2d_csv_out).string();
    if (!cfg.map3d_csv_out.empty()) cfg.map3d_csv_out = resolvePath(base, cfg.map3d_csv_out).string();
    if (!cfg.pnp_report_out.empty()) cfg.pnp_report_out = resolvePath(base, cfg.pnp_report_out).string();
    if (!cfg.dem_tif.empty()) cfg.dem_tif = resolvePath(base, cfg.dem_tif).string();
    if (!cfg.camera_intrinsics_file.empty()) cfg.camera_intrinsics_file = resolvePath(base, cfg.camera_intrinsics_file).string();

    int cam_w = 0;
    int cam_h = 0;
    if (!cfg.camera_intrinsics_file.empty()) {
        if (!loadCameraIntrinsicsYaml(cfg.camera_intrinsics_file, cfg.fx, cfg.fy, cfg.cx, cfg.cy, cfg.fish_distortion, cam_w, cam_h)) {
            std::cout << "[Camera] Failed to parse intrinsics file: " << cfg.camera_intrinsics_file << "\n";
        }
    }

    if (!cfg.matches_out.empty()) {
        if (cfg.matches_inlier_out.empty()) cfg.matches_inlier_out = makeSiblingPath(cfg.matches_out, "_inliers");
        if (cfg.warp_out.empty()) cfg.warp_out = makeSiblingPath(cfg.matches_out, "_warp");
        if (cfg.compare_out.empty()) cfg.compare_out = makeSiblingPath(cfg.matches_out, "_compare");
        if (cfg.map2d_csv_out.empty()) cfg.map2d_csv_out = makeSiblingPathWithExt(cfg.matches_out, "_map2d", ".csv");
        if (cfg.map3d_csv_out.empty()) cfg.map3d_csv_out = makeSiblingPathWithExt(cfg.matches_out, "_map3d", ".csv");
        if (cfg.pnp_report_out.empty()) cfg.pnp_report_out = makeSiblingPathWithExt(cfg.matches_out, "_pnp", ".txt");
    }
    if (!cfg.sp1_kpts_out.empty() && cfg.undistort_out.empty()) {
        cfg.undistort_out = makeSiblingPath(cfg.sp1_kpts_out, "_undistort");
    }
    if (cfg.dem_tif.empty()) cfg.dem_tif = cfg.image0;

    if (cfg.homography_ransac_reproj <= 0.0) cfg.homography_ransac_reproj = 3.0;
    if (cfg.homography_max_pairs < 4) cfg.homography_max_pairs = 300;
    if (cfg.max_features <= 0) cfg.max_features = 4000;
    if (cfg.fast_threshold <= 0) cfg.fast_threshold = 20;
    if (cfg.matcher_ratio <= 0.f || cfg.matcher_ratio >= 1.0f) cfg.matcher_ratio = 0.8f;
    if (cfg.matcher_max_keep <= 0) cfg.matcher_max_keep = 3000;

    return cfg;
}

// 功能：打印程序用法说明。
static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [config.yaml]\n"
              << "       " << prog << " --worker\n"
              << "Default config path: ./config/config.yaml\n";
}

// 功能：根据命令行参数选择配置文件路径。
static std::string pickConfigPath(int argc, char** argv) {
    if (argc == 2) {
        return argv[1];
    }

    std::vector<std::filesystem::path> candidates;
    candidates.emplace_back(std::filesystem::current_path() / "config" / "config.yaml");

    std::filesystem::path exe_path;
    try {
        exe_path = std::filesystem::absolute(std::filesystem::path(argv[0]));
    } catch (...) {
        exe_path.clear();
    }

    if (!exe_path.empty()) {
        const std::filesystem::path exe_dir = exe_path.parent_path();
        candidates.emplace_back(exe_dir / "config" / "config.yaml");
        candidates.emplace_back(exe_dir / ".." / "config" / "config.yaml");
        candidates.emplace_back(exe_dir / ".." / ".." / "config" / "config.yaml");
    }

    for (const auto& c : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(c, ec) && !ec) {
            return c.lexically_normal().string();
        }
    }

    std::string msg = "Failed to locate default config.yaml. Tried:\n";
    for (const auto& c : candidates) {
        msg += "  - " + c.lexically_normal().string() + "\n";
    }
    msg += "Please pass config path explicitly, e.g.: ./sp_lg ../config/config.yaml";
    throw std::runtime_error(msg);
}

static int runSingleConfig(const std::string& cfg_path) {
        AppConfig cfg = loadConfig(cfg_path);

        FeatureMethod feat_method = parseFeatureMethod(cfg.feature_method);
        MatcherMethod matcher_method = parseMatcherMethod(cfg.matcher_method);
        DistanceType dist_type = parseDistanceType(cfg.matcher_distance);

        // 支持 euclidean/hamming 作为匹配方法简写（等价于 BF + 对应距离）。
        const std::string matcher_lc = toLowerCopy(sanitizeConfigToken(cfg.matcher_method));
        if (matcher_lc == "euclidean" || matcher_lc == "l2") {
            matcher_method = MatcherMethod::BF;
            dist_type = DistanceType::L2;
        } else if (matcher_lc == "hamming") {
            matcher_method = MatcherMethod::BF;
            dist_type = DistanceType::Hamming;
        }

        validateCombo(feat_method, matcher_method, dist_type);

        std::cout << "[Config] image0=" << cfg.image0 << "\n"
                  << "[Config] image1=" << cfg.image1 << "\n"
                  << "[Config] dem_tif=" << cfg.dem_tif << "\n"
                  << "[Config] superpoint=" << cfg.superpoint_onnx << "\n"
                  << "[Config] lightglue=" << cfg.lightglue_onnx << "\n"
                  << "[Config] output(sp0/sp1/matches)=" << cfg.sp0_kpts_out << " | " << cfg.sp1_kpts_out << " | " << cfg.matches_out << "\n"
                  << "[Config] output(matches_inlier)=" << cfg.matches_inlier_out << "\n"
                  << "[Config] output(warp/compare)=" << cfg.warp_out << " | " << cfg.compare_out << "\n"
                  << "[Config] output(undistort/map2d/map3d/pnp)=" << cfg.undistort_out << " | " << cfg.map2d_csv_out << " | " << cfg.map3d_csv_out << " | " << cfg.pnp_report_out << "\n"
                  << "[Config] H=" << cfg.H << " W=" << cfg.W
                  << " mscore_thresh=" << cfg.mscore_thresh
                  << " min_kpt_dist=" << cfg.min_kpt_dist
                  << " max_draw=" << cfg.max_draw
                  << " homography_ransac_reproj=" << cfg.homography_ransac_reproj
                  << " homography_max_pairs=" << cfg.homography_max_pairs
                  << " feature.method=" << featureMethodName(feat_method)
                  << " feature.max_features=" << cfg.max_features
                  << " matcher.method=" << matcherMethodName(matcher_method)
                  << " matcher.distance=" << distanceTypeName(dist_type)
                  << " matcher.ratio=" << cfg.matcher_ratio
                  << " matcher.cross_check=" << (cfg.matcher_cross_check ? 1 : 0)
                  << " matcher.max_keep=" << cfg.matcher_max_keep
                  << " use_fisheye_undistort=" << (cfg.use_fisheye_undistort ? 1 : 0)
                  << " undistort_balance=" << cfg.undistort_balance
                  << " undistort_fov_scale=" << cfg.undistort_fov_scale
                  << " try_cuda=" << (cfg.try_cuda ? 1 : 0)
                  << " device_id=" << cfg.device_id << "\n";

        CameraModel cam_model;
        bool camera_ok = buildCameraModelFromConfig(cfg, cam_model);
        if (!camera_ok) {
            std::cout << "[Camera] fx/fy not ready. camera_intrinsics_file=" << cfg.camera_intrinsics_file << "\n";
        }

        cv::Mat img0_raw = readInputImage(cfg.image0);
        cv::Mat img1_raw_input = readInputImage(cfg.image1);

        RasterGeoRef map_geo_ref;
        RasterGeoRef dem_geo_ref;
        std::string map_geo_reason;
        std::string dem_geo_reason;
        const bool map_geo_ok = loadRasterGeoRef(cfg.image0, map_geo_ref, map_geo_reason);
        const bool dem_geo_ok = loadRasterGeoRef(cfg.dem_tif, dem_geo_ref, dem_geo_reason);

        if (map_geo_ok) {
            std::cout << "[GeoTIFF] map georef loaded from " << cfg.image0 << "\n";
        } else {
            std::cout << "[GeoTIFF] map georef unavailable: " << map_geo_reason << "\n";
        }

        if (dem_geo_ok) {
            std::cout << "[GeoTIFF] dem georef loaded from " << cfg.dem_tif << "\n";
        } else {
            std::cout << "[GeoTIFF] dem georef unavailable: " << dem_geo_reason << "\n";
        }

        cv::Mat img1_match_raw = img1_raw_input;
        cv::Mat pnp_camera_K = cfg.pnp_camera_matrix_valid
            ? matFromArray9(cfg.pnp_camera_matrix)
            : cam_model.K.clone();
        const cv::Matx33d image1_inverse_affine = cfg.image1_inverse_affine_valid
            ? matx33dFromArray9(cfg.image1_inverse_affine)
            : cv::Matx33d::eye();

        if (cfg.use_fisheye_undistort) {
            if (!camera_ok) {
                throw std::runtime_error("Fisheye undistort enabled but camera intrinsics are missing.");
            }
            cv::Mat undistorted;
            cv::Mat newK;
            if (!undistortFisheye(img1_raw_input, cam_model, cfg.undistort_balance, cfg.undistort_fov_scale, undistorted, newK)) {
                throw std::runtime_error("Fisheye undistort failed.");
            }
            img1_match_raw = undistorted;
            if (!cfg.pnp_camera_matrix_valid) {
                pnp_camera_K = newK.clone();
            }

            if (!cfg.undistort_out.empty()) {
                ensureParentDir(cfg.undistort_out);
                if (!cv::imwrite(cfg.undistort_out, img1_match_raw)) {
                    throw std::runtime_error("Failed to write undistorted image: " + cfg.undistort_out);
                }
                std::cout << "[Undistort] saved=" << cfg.undistort_out << "\n";
            }
            std::cout << "[Undistort] K_new=" << pnp_camera_K << "\n";
        }

        cv::Mat img0_gray8 = toGray8ForModel(img0_raw);
        cv::Mat img1_gray8 = toGray8ForModel(img1_match_raw);

        const bool need_keypoint_vis = !cfg.sp0_kpts_out.empty() || !cfg.sp1_kpts_out.empty();
        const bool need_match_vis = !cfg.matches_out.empty() || !cfg.matches_inlier_out.empty();
        const bool need_warp_vis = !cfg.warp_out.empty() || !cfg.compare_out.empty();
        const bool need_any_vis = need_keypoint_vis || need_match_vis || need_warp_vis;

        cv::Mat img0_vis;
        cv::Mat img1_vis;
        if (need_any_vis) {
            img0_vis = toVisBgr8(img0_raw);
            img1_vis = toVisBgr8(img1_match_raw);
            cv::resize(img0_vis, img0_vis, cv::Size(cfg.W, cfg.H));
            cv::resize(img1_vis, img1_vis, cv::Size(cfg.W, cfg.H));
        }

        RuntimeSessions runtime_sessions = acquireRuntimeSessions(cfg, feat_method, matcher_method);
        const bool need_sp = (feat_method == FeatureMethod::SuperPoint || matcher_method == MatcherMethod::LightGlue);
        const bool need_lg = (matcher_method == MatcherMethod::LightGlue);
        if (need_sp && runtime_sessions.sp_sess == nullptr) {
            throw std::runtime_error("SuperPoint session is unavailable.");
        }
        if (need_lg && runtime_sessions.lg_sess == nullptr) {
            throw std::runtime_error("LightGlue session is unavailable.");
        }

        const double model_load_ms = runtime_sessions.model_load_ms;
        std::cout << "[EP] use_cuda=" << (runtime_sessions.use_cuda ? 1 : 0) << "\n";
        std::cout << "[Timing] model_load_ms=" << model_load_ms << "\n";
        std::cout << "[ModelCache] hit=" << (runtime_sessions.cache_hit ? 1 : 0) << "\n";

        std::vector<cv::KeyPoint> kpts0;
        std::vector<cv::KeyPoint> kpts1;
        cv::Mat desc0;
        cv::Mat desc1;
        bool binary_desc = false;

        SPFeatures sp0;
        SPFeatures sp1;
        const auto feature_t0 = std::chrono::steady_clock::now();
        if (feat_method == FeatureMethod::SuperPoint) {
            sp0 = runSuperPoint(*runtime_sessions.sp_sess, img0_gray8, cfg.H, cfg.W, "img0");
            sp1 = runSuperPoint(*runtime_sessions.sp_sess, img1_gray8, cfg.H, cfg.W, "img1");

            filterFeaturesByMinDistance(sp0, cfg.min_kpt_dist);
            filterFeaturesByMinDistance(sp1, cfg.min_kpt_dist);

            kpts0 = sp0.kpts;
            kpts1 = sp1.kpts;
            desc0 = superPointDescToMat(sp0);
            desc1 = superPointDescToMat(sp1);
            binary_desc = false;
        } else {
            LocalFeatures f0 = runClassicalFeature(
                img0_gray8, cfg.H, cfg.W, feat_method,
                cfg.max_features, cfg.fast_threshold, cfg.fast_nonmax, "img0");
            LocalFeatures f1 = runClassicalFeature(
                img1_gray8, cfg.H, cfg.W, feat_method,
                cfg.max_features, cfg.fast_threshold, cfg.fast_nonmax, "img1");

            filterClassicalFeaturesByMinDistance(f0, cfg.min_kpt_dist);
            filterClassicalFeaturesByMinDistance(f1, cfg.min_kpt_dist);

            kpts0 = std::move(f0.kpts);
            kpts1 = std::move(f1.kpts);
            desc0 = std::move(f0.desc);
            desc1 = std::move(f1.desc);
            binary_desc = f0.binary_desc;
        }
        const auto feature_t1 = std::chrono::steady_clock::now();
        const double feature_ms =
            std::chrono::duration<double, std::milli>(feature_t1 - feature_t0).count();
        const int map_feature_points = static_cast<int>(kpts0.size());
        const int aerial_feature_points = static_cast<int>(kpts1.size());
        std::cout << "[Timing] feature_ms=" << feature_ms << "\n";

        cv::Mat sp0_vis;
        cv::Mat sp1_vis;
        if (need_keypoint_vis) {
            if (!cfg.sp0_kpts_out.empty()) {
                cv::drawKeypoints(img0_vis, kpts0, sp0_vis, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            }
            if (!cfg.sp1_kpts_out.empty()) {
                cv::drawKeypoints(img1_vis, kpts1, sp1_vis, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            }
        }

        const auto matching_t0 = std::chrono::steady_clock::now();
        const auto pair_match_t0 = matching_t0;
        std::vector<MatchWithScore> pairs;
        if (matcher_method == MatcherMethod::LightGlue) {
            pairs = runLightGlue(*runtime_sessions.lg_sess, sp0, sp1, cfg.mscore_thresh, cfg.W, cfg.H);
        } else {
            pairs = runDescriptorMatcher(
                desc0,
                desc1,
                matcher_method,
                dist_type,
                binary_desc,
                cfg.matcher_cross_check,
                cfg.matcher_ratio,
                cfg.matcher_max_keep);
        }
        std::sort(pairs.begin(), pairs.end(), [](const MatchWithScore& a, const MatchWithScore& b) { return a.s > b.s; });
        const int lightglue_match_pairs = static_cast<int>(pairs.size());
        const auto pair_match_t1 = std::chrono::steady_clock::now();
        const double pair_match_ms =
            std::chrono::duration<double, std::milli>(pair_match_t1 - pair_match_t0).count();

        const auto homography_t0 = std::chrono::steady_clock::now();
        cv::Mat H01;
        std::vector<uchar> inlier_mask;
        int used_pairs = 0;
        int inlier_count = 0;
        bool homography_ok = estimateHomographyFromMatches(
            pairs, kpts0, kpts1,
            cfg.homography_ransac_reproj,
            cfg.homography_max_pairs,
            H01, inlier_mask, used_pairs, inlier_count);

        std::vector<MatchWithScore> draw_pairs = pairs;
        if (homography_ok && used_pairs > 0 && static_cast<int>(inlier_mask.size()) == used_pairs) {
            draw_pairs.clear();
            draw_pairs.reserve(static_cast<size_t>(inlier_count));
            for (int i = 0; i < used_pairs; ++i) {
                if (inlier_mask[static_cast<size_t>(i)] != 0) {
                    draw_pairs.push_back(pairs[static_cast<size_t>(i)]);
                }
            }
            std::cout << "[Homography] inliers=" << inlier_count << "/" << used_pairs << "\n";
            std::cout << "[Homography] H01=" << H01 << "\n";
        } else {
            std::cout << "[Homography] failed, need >=4 consistent matches. current pairs=" << pairs.size() << "\n";
        }
        const auto homography_t1 = std::chrono::steady_clock::now();
        const double homography_ms =
            std::chrono::duration<double, std::milli>(homography_t1 - homography_t0).count();

        const auto match_vis_t0 = std::chrono::steady_clock::now();
        cv::Mat match_vis_raw;
        cv::Mat match_vis_inlier;
        if (!cfg.matches_out.empty()) {
            match_vis_raw = drawMatchesCustom(img0_vis, img1_vis, kpts0, kpts1, pairs, cfg.max_draw);
        }
        if (!cfg.matches_inlier_out.empty()) {
            match_vis_inlier = drawMatchesCustom(img0_vis, img1_vis, kpts0, kpts1, draw_pairs, cfg.max_draw);
        }
        const auto match_vis_t1 = std::chrono::steady_clock::now();
        const double match_vis_ms =
            std::chrono::duration<double, std::milli>(match_vis_t1 - match_vis_t0).count();

        const auto geo2d_t0 = std::chrono::steady_clock::now();
        std::vector<MatchWithScore> geo_pairs;
        if (homography_ok && used_pairs > 0 && static_cast<int>(inlier_mask.size()) == used_pairs) {
            for (int i = 0; i < used_pairs; ++i) {
                if (inlier_mask[static_cast<size_t>(i)] != 0) geo_pairs.push_back(pairs[static_cast<size_t>(i)]);
            }
        } else {
            geo_pairs = pairs;
        }

        if (static_cast<int>(geo_pairs.size()) > cfg.homography_max_pairs) {
            geo_pairs.resize(static_cast<size_t>(cfg.homography_max_pairs));
        }

        const double map_scale_x = static_cast<double>(img0_raw.cols) / static_cast<double>(cfg.W);
        const double map_scale_y = static_cast<double>(img0_raw.rows) / static_cast<double>(cfg.H);
        const double cam_scale_x = static_cast<double>(img1_match_raw.cols) / static_cast<double>(cfg.W);
        const double cam_scale_y = static_cast<double>(img1_match_raw.rows) / static_cast<double>(cfg.H);

        std::vector<Geo2DRecord> geo2d;
        geo2d.reserve(geo_pairs.size());
        for (const auto& m : geo_pairs) {
            if (m.q < 0 || m.t < 0) continue;
            if (static_cast<size_t>(m.q) >= kpts0.size() || static_cast<size_t>(m.t) >= kpts1.size()) continue;

            Geo2DRecord r;
            r.map_x = static_cast<float>(kpts0[static_cast<size_t>(m.q)].pt.x * map_scale_x);
            r.map_y = static_cast<float>(kpts0[static_cast<size_t>(m.q)].pt.y * map_scale_y);
            r.cam_x = static_cast<float>(kpts1[static_cast<size_t>(m.t)].pt.x * cam_scale_x);
            r.cam_y = static_cast<float>(kpts1[static_cast<size_t>(m.t)].pt.y * cam_scale_y);
            r.score = m.s;

            if (map_geo_ok) {
                if (!rasterPixelToWgs84(map_geo_ref, r.map_x, r.map_y, r.lon, r.lat)) {
                    r.lon = std::numeric_limits<double>::quiet_NaN();
                    r.lat = std::numeric_limits<double>::quiet_NaN();
                }
            } else if (cfg.map_affine_valid) {
                pixelToLonLat(cfg.map_wgs84_affine, r.map_x, r.map_y, r.lon, r.lat);
            } else {
                r.lon = std::numeric_limits<double>::quiet_NaN();
                r.lat = std::numeric_limits<double>::quiet_NaN();
            }
            geo2d.push_back(r);
        }

        if (!map_geo_ok && !cfg.map_affine_valid) {
            std::cout << "[Geo] no map georef available (GeoTIFF + config both unavailable), 2D-2D geographic mapping will contain NaN lon/lat.\n";
        }

        if (!cfg.map2d_csv_out.empty()) {
            if (!writeGeo2DCsv(cfg.map2d_csv_out, geo2d)) {
                throw std::runtime_error("Failed to write 2D mapping csv: " + cfg.map2d_csv_out);
            }
            std::cout << "[Geo] 2D-2D mapping rows=" << geo2d.size() << " saved=" << cfg.map2d_csv_out << "\n";
        }
        const auto geo2d_t1 = std::chrono::steady_clock::now();
        const double geo2d_ms =
            std::chrono::duration<double, std::milli>(geo2d_t1 - geo2d_t0).count();

        const auto dem_sample_t0 = std::chrono::steady_clock::now();
        std::vector<Geo3DRecord> geo3d;
        const bool can_sample_dem = !cfg.dem_tif.empty() && (dem_geo_ok || cfg.dem_affine_valid);
        if (can_sample_dem) {
            cv::Mat demf = readDemAsFloat(cfg.dem_tif);
            geo3d.reserve(geo2d.size());
            for (const auto& r : geo2d) {
                if (!std::isfinite(r.lon) || !std::isfinite(r.lat)) continue;
                double dx = 0.0;
                double dy = 0.0;
                bool ok_xy = false;
                if (dem_geo_ok) {
                    ok_xy = wgs84ToRasterPixel(dem_geo_ref, r.lon, r.lat, dx, dy);
                } else {
                    ok_xy = lonLatToPixel(cfg.dem_wgs84_affine, r.lon, r.lat, dx, dy);
                }
                if (!ok_xy) continue;
                float alt = 0.f;
                if (!sampleDemBilinear(demf, dx, dy, cfg.dem_nodata, alt)) continue;
                geo3d.push_back({r, alt});
            }
        } else {
            std::cout << "[Geo] no DEM georef available (GeoTIFF + config both unavailable), skip 2D-3D mapping.\n";
        }

        if (!cfg.map3d_csv_out.empty()) {
            if (!writeGeo3DCsv(cfg.map3d_csv_out, geo3d)) {
                throw std::runtime_error("Failed to write 3D mapping csv: " + cfg.map3d_csv_out);
            }
            std::cout << "[Geo] 2D-3D mapping rows=" << geo3d.size() << " saved=" << cfg.map3d_csv_out << "\n";
        }
        const auto dem_sample_t1 = std::chrono::steady_clock::now();
        const double dem_sample_ms =
            std::chrono::duration<double, std::milli>(dem_sample_t1 - dem_sample_t0).count();
        const TerrainStats terrain_stats = computeTerrainStats(geo3d);
        const auto matching_t1 = std::chrono::steady_clock::now();
        const double matching_ms =
            std::chrono::duration<double, std::milli>(matching_t1 - matching_t0).count();
        std::cout << "[Timing] matching_ms=" << matching_ms << "\n";
        std::cout << "[Timing] pair_match_ms=" << pair_match_ms
                  << " homography_ms=" << homography_ms
                  << " match_vis_ms=" << match_vis_ms
                  << " geo2d_ms=" << geo2d_ms
                  << " dem_sample_ms=" << dem_sample_ms << "\n";

        bool pnp_ok = false;
        cv::Vec3d pnp_rvec(0.0, 0.0, 0.0);
        cv::Vec3d pnp_tvec(0.0, 0.0, 0.0);
        int pnp_inliers = 0;
        double pnp_reproj = std::numeric_limits<double>::quiet_NaN();
        double cam_lon = std::numeric_limits<double>::quiet_NaN();
        double cam_lat = std::numeric_limits<double>::quiet_NaN();
        double cam_alt = std::numeric_limits<double>::quiet_NaN();
        GeometryFeatureStats geom_stats;
        DualGeometryStats dual_geom_stats;
        int pnp_input_points = 0;
        const double pnp_f_eff_px =
            pnp_camera_K.empty()
                ? std::numeric_limits<double>::quiet_NaN()
                : std::sqrt(pnp_camera_K.at<double>(0, 0) * pnp_camera_K.at<double>(1, 1));
        const auto pnp_t0 = std::chrono::steady_clock::now();

        if (geo3d.size() >= 6 && !pnp_camera_K.empty()) {
            const double lon0 = geo3d.front().m.lon;
            const double lat0 = geo3d.front().m.lat;
            const double alt0 = geo3d.front().alt;
            const cv::Vec3d ecef0 = geodeticToECEF(lon0, lat0, alt0);

            cv::Vec3d east, north, up;
            enuAxes(lon0, lat0, east, north, up);

            std::vector<cv::Point3f> obj_pts;
            std::vector<cv::Point2f> img_pts;
            obj_pts.reserve(geo3d.size());
            img_pts.reserve(geo3d.size());

            for (const auto& r : geo3d) {
                cv::Vec3d ecef = geodeticToECEF(r.m.lon, r.m.lat, r.alt);
                cv::Vec3d enu = ecefToEnu(ecef, ecef0, east, north, up);
                cv::Point2f pnp_cam_pt(r.m.cam_x, r.m.cam_y);
                if (cfg.image1_inverse_affine_valid) {
                    if (!transformPointHomogeneous(image1_inverse_affine, pnp_cam_pt, pnp_cam_pt)) {
                        continue;
                    }
                }
                obj_pts.emplace_back(static_cast<float>(enu[0]), static_cast<float>(enu[1]), static_cast<float>(enu[2]));
                img_pts.emplace_back(pnp_cam_pt);
            }
            pnp_input_points = static_cast<int>(obj_pts.size());

            std::vector<int> pnp_inlier_idx;
            cv::Mat rvec, tvec;
            cv::Mat dist_zero = cv::Mat::zeros(1, 4, CV_64F);

            bool ok = cv::solvePnPRansac(obj_pts,
                                         img_pts,
                                         pnp_camera_K,
                                         dist_zero,
                                         rvec,
                                         tvec,
                                         false,
                                         300,
                                         6.0,
                                         0.99,
                                         pnp_inlier_idx,
                                         cv::SOLVEPNP_EPNP);

            if (ok && pnp_inlier_idx.size() >= 4) {
                std::vector<cv::Point3f> obj_in;
                std::vector<cv::Point2f> img_in;
                obj_in.reserve(pnp_inlier_idx.size());
                img_in.reserve(pnp_inlier_idx.size());
                for (int idx : pnp_inlier_idx) {
                    obj_in.push_back(obj_pts[static_cast<size_t>(idx)]);
                    img_in.push_back(img_pts[static_cast<size_t>(idx)]);
                }

                cv::solvePnP(obj_in, img_in, pnp_camera_K, dist_zero, rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

                std::vector<cv::Point2f> reproj_pts;
                cv::projectPoints(obj_in, rvec, tvec, pnp_camera_K, dist_zero, reproj_pts);
                double err_sum = 0.0;
                for (size_t i = 0; i < reproj_pts.size(); ++i) {
                    cv::Point2f d = reproj_pts[i] - img_in[i];
                    err_sum += std::sqrt(static_cast<double>(d.x * d.x + d.y * d.y));
                }
                pnp_reproj = reproj_pts.empty() ? 0.0 : err_sum / static_cast<double>(reproj_pts.size());

                pnp_rvec = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
                pnp_tvec = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                pnp_inliers = static_cast<int>(pnp_inlier_idx.size());
                pnp_ok = true;

                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::Mat cam_center = -R.t() * tvec;
                cv::Vec3d cam_enu(cam_center.at<double>(0), cam_center.at<double>(1), cam_center.at<double>(2));
                cv::Vec3d cam_ecef = enuToEcef(cam_enu, ecef0, east, north, up);
                ecefToGeodetic(cam_ecef, cam_lon, cam_lat, cam_alt);
                geom_stats.point_spread_img_ratio =
                    computePointSpreadImgRatio(img_in,
                                               static_cast<double>(img1_match_raw.cols),
                                               static_cast<double>(img1_match_raw.rows));
                geom_stats.point_spread_map_xy_m2 = computePointSpreadMapXYM2(obj_in);
                geom_stats.depth_spread_m = computeDepthSpreadM(obj_in, cam_enu);
                geom_stats.obliqueness_deg = computeObliquenessDeg(R);
                if (homography_ok && !H01.empty()) {
                    cv::Mat H01_64f;
                    H01.convertTo(H01_64f, CV_64F);
                    const cv::Matx33d H01_matx(
                        H01_64f.at<double>(0, 0), H01_64f.at<double>(0, 1), H01_64f.at<double>(0, 2),
                        H01_64f.at<double>(1, 0), H01_64f.at<double>(1, 1), H01_64f.at<double>(1, 2),
                        H01_64f.at<double>(2, 0), H01_64f.at<double>(2, 1), H01_64f.at<double>(2, 2));
                    dual_geom_stats = computeDualGeometryStats(
                        H01_matx,
                        geo3d,
                        obj_pts,
                        img_pts,
                        pnp_inlier_idx,
                        pnp_rvec,
                        pnp_tvec,
                        pnp_camera_K,
                        image1_inverse_affine,
                        cfg.image1_inverse_affine_valid,
                        map_scale_x,
                        map_scale_y,
                        cam_scale_x,
                        cam_scale_y);
                }

                std::cout << "[PnP] success inliers=" << pnp_inliers << "/" << obj_pts.size()
                          << " reproj=" << pnp_reproj << " px\n";
                std::cout << "[PnP] camera lon/lat/alt=" << cam_lon << ", " << cam_lat << ", " << cam_alt << "\n";
                if (dual_geom_stats.shared_count > 0) {
                    std::cout << "[DualGeom] shared=" << dual_geom_stats.shared_count
                              << " h_rmse_px=" << dual_geom_stats.homography_rmse_px
                              << " pnp_rmse_px=" << dual_geom_stats.pnp_rmse_px
                              << " disagree_rmse_px=" << dual_geom_stats.disagree_rmse_px
                              << " valid=" << (dual_geom_stats.valid ? 1 : 0) << "\n";
                    if (dual_geom_stats.valid) {
                        std::cout << "[DualGeom] var_x_m2=" << dual_geom_stats.var_x_m2
                                  << " cov_xy_m2=" << dual_geom_stats.cov_xy_m2
                                  << " var_y_m2=" << dual_geom_stats.var_y_m2 << "\n";
                    }
                }
            } else {
                std::cout << "[PnP] solvePnPRansac failed. points=" << obj_pts.size() << "\n";
            }
        } else {
            std::cout << "[PnP] skipped: require >=6 valid 2D-3D points and camera intrinsics.\n";
        }
        const auto pnp_t1 = std::chrono::steady_clock::now();
        const double pnp_ms =
            std::chrono::duration<double, std::milli>(pnp_t1 - pnp_t0).count();
        double scale_h_m_per_px = std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(cam_alt) &&
            std::isfinite(terrain_stats.alt_median_m) &&
            std::isfinite(pnp_f_eff_px) &&
            pnp_f_eff_px > 0.0) {
            scale_h_m_per_px = std::abs(cam_alt - terrain_stats.alt_median_m) / pnp_f_eff_px;
        }
        std::cout << "[Timing] pnp_ms=" << pnp_ms << "\n";

        if (!writePnpReport(cfg.pnp_report_out,
                            pnp_ok,
                            map_feature_points,
                            aerial_feature_points,
                            lightglue_match_pairs,
                            used_pairs,
                            inlier_count,
                            pnp_input_points,
                            pnp_input_points,
                            pnp_inliers,
                            pnp_reproj,
                            terrain_stats.relief_m,
                            terrain_stats.alt_std_m,
                            terrain_stats.alt_median_m,
                            pnp_f_eff_px,
                            scale_h_m_per_px,
                            geom_stats.point_spread_img_ratio,
                            geom_stats.point_spread_map_xy_m2,
                            geom_stats.depth_spread_m,
                            geom_stats.obliqueness_deg,
                            model_load_ms,
                            feature_ms,
                            matching_ms,
                            pair_match_ms,
                            homography_ms,
                            match_vis_ms,
                            geo2d_ms,
                            dem_sample_ms,
                            pnp_ms,
                            dual_geom_stats,
                            pnp_rvec,
                            pnp_tvec,
                            cam_lon,
                            cam_lat,
                            cam_alt)) {
            throw std::runtime_error("Failed to write PnP report: " + cfg.pnp_report_out);
        }

        cv::Mat warped;
        cv::Mat compare_vis;
        if (need_warp_vis) {
            if (homography_ok) {
                cv::warpPerspective(img0_vis, warped, H01, img1_vis.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            } else {
                warped = cv::Mat::zeros(img1_vis.size(), img1_vis.type());
                cv::putText(warped,
                            "homography failed",
                            cv::Point(20, 40),
                            cv::FONT_HERSHEY_SIMPLEX,
                            1.0,
                            cv::Scalar(0, 0, 255),
                            2);
            }
            if (!cfg.compare_out.empty()) {
                compare_vis = makeHomographyCompareCanvas(img1_vis, warped, inlier_count, used_pairs, homography_ok);
            }
        }

        if (!cfg.sp0_kpts_out.empty()) {
            ensureParentDir(cfg.sp0_kpts_out);
            if (!cv::imwrite(cfg.sp0_kpts_out, sp0_vis)) {
                throw std::runtime_error("Failed to write output: " + cfg.sp0_kpts_out);
            }
            std::cout << "[Output] Saved SuperPoint image0 keypoints: " << cfg.sp0_kpts_out << "\n";
        }
        if (!cfg.sp1_kpts_out.empty()) {
            ensureParentDir(cfg.sp1_kpts_out);
            if (!cv::imwrite(cfg.sp1_kpts_out, sp1_vis)) {
                throw std::runtime_error("Failed to write output: " + cfg.sp1_kpts_out);
            }
            std::cout << "[Output] Saved SuperPoint image1 keypoints: " << cfg.sp1_kpts_out << "\n";
        }
        if (!cfg.matches_out.empty()) {
            ensureParentDir(cfg.matches_out);
            if (!cv::imwrite(cfg.matches_out, match_vis_raw)) {
                throw std::runtime_error("Failed to write output: " + cfg.matches_out);
            }
            std::cout << "[Output] Saved LightGlue matches(raw): " << cfg.matches_out << "\n";
        }
        if (!cfg.matches_inlier_out.empty()) {
            ensureParentDir(cfg.matches_inlier_out);
            if (!cv::imwrite(cfg.matches_inlier_out, match_vis_inlier)) {
                throw std::runtime_error("Failed to write output: " + cfg.matches_inlier_out);
            }
            std::cout << "[Output] Saved LightGlue matches(inlier): " << cfg.matches_inlier_out << "\n";
        }
        if (!cfg.warp_out.empty()) {
            ensureParentDir(cfg.warp_out);
            if (!cv::imwrite(cfg.warp_out, warped)) {
                throw std::runtime_error("Failed to write output: " + cfg.warp_out);
            }
            std::cout << "[Output] Saved warped map: " << cfg.warp_out << "\n";
        }
        if (!cfg.compare_out.empty()) {
            ensureParentDir(cfg.compare_out);
            if (!cv::imwrite(cfg.compare_out, compare_vis)) {
                throw std::runtime_error("Failed to write output: " + cfg.compare_out);
            }
            std::cout << "[Output] Saved homography compare: " << cfg.compare_out << "\n";
        }
        if (!cfg.undistort_out.empty()) {
            std::cout << "[Output] Saved undistorted image: " << cfg.undistort_out << "\n";
        }
        if (!cfg.map2d_csv_out.empty()) {
            std::cout << "[Output] Saved map2d csv: " << cfg.map2d_csv_out << "\n";
        }
        if (!cfg.map3d_csv_out.empty()) {
            std::cout << "[Output] Saved map3d csv: " << cfg.map3d_csv_out << "\n";
        }
        std::cout << "[Output] Saved pnp report: " << cfg.pnp_report_out << "\n";

        return 0;
}

// 功能：运行 `sp_lg` worker 循环，持续处理外部请求。
static int runWorkerLoop() {
    std::cout << "[SP_LG_WORKER_READY]" << "\n";
    std::cout.flush();

    std::string line;
    while (std::getline(std::cin, line)) {
        const std::string cfg_path = trimCopy(line);
        if (cfg_path.empty()) {
            continue;
        }
        if (cfg_path == "__exit__") {
            break;
        }

        std::cout << "[SP_LG_WORKER_BEGIN] cfg=" << cfg_path << "\n";
        std::cout.flush();

        int rc = 0;
        try {
            rc = runSingleConfig(cfg_path);
        } catch (const std::exception& e) {
            std::cerr << "[Error] " << e.what() << "\n";
            rc = 1;
        }

        std::cout << "[SP_LG_WORKER_DONE] rc=" << rc << " cfg=" << cfg_path << "\n";
        std::cout.flush();
    }

    return 0;
}

// 功能：实现 `main` 对应的功能。
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--worker") {
            return runWorkerLoop();
        }

        if (argc > 2) {
            printUsage(argv[0]);
            return 1;
        }

        const std::string cfg_path = pickConfigPath(argc, argv);
        return runSingleConfig(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}
