/**
 * @file fusion_geometry.h
 * @brief 几何、坐标变换、IMU 跟踪与地图栅格工具。
 *
 * 这里放的是融合和视觉都要用到的几何基础：
 * ENU / ECEF / WGS84 转换、IMU 姿态跟踪、短时预积分、
 * 栅格地图读取与裁图、射线与地理计算等。
 */

#pragma once

#include "splg_fusion/common.h"
#include "splg_fusion/config/fusion_config_camera.h"
#include "splg_fusion/fusion/fusion_types.h"

// 功能：对向量做归一化，得到单位方向向量。
static cv::Vec3d normalizeVec(const cv::Vec3d& v) {
    const double n = cv::norm(v);
    if (n < 1e-12) return v;
    return v * (1.0 / n);
}

// 功能：判断三维向量是否都是有效有限数值。
static bool isFiniteVec(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

// 功能：根据向量构造反对称矩阵。
static cv::Matx33d skewMat(const cv::Vec3d& v) {
    return cv::Matx33d(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0);
}

// 功能：计算旋转向量到旋转矩阵的指数映射。
static cv::Matx33d so3Exp(const cv::Vec3d& rotvec) {
    const double theta = cv::norm(rotvec);
    const cv::Matx33d I = cv::Matx33d::eye();
    if (theta < 1e-12) {
        return I + skewMat(rotvec);
    }

    const cv::Matx33d K = skewMat(rotvec * (1.0 / theta));
    return I + std::sin(theta) * K + (1.0 - std::cos(theta)) * (K * K);
}

// 功能：重新正交化旋转矩阵，抑制数值漂移。
static cv::Matx33d orthonormalize(const cv::Matx33d& R) {
    cv::Vec3d x(R(0, 0), R(1, 0), R(2, 0));
    cv::Vec3d y(R(0, 1), R(1, 1), R(2, 1));
    x = normalizeVec(x);
    y = y - x * x.dot(y);
    y = normalizeVec(y);
    cv::Vec3d z = x.cross(y);
    z = normalizeVec(z);
    y = z.cross(x);
    y = normalizeVec(y);
    return cv::Matx33d(
        x[0], y[0], z[0],
        x[1], y[1], z[1],
        x[2], y[2], z[2]);
}

// 功能：构造绕 X 轴旋转的矩阵。
static cv::Matx33d rotationX(double rad) {
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return cv::Matx33d(
        1.0, 0.0, 0.0,
        0.0, c, -s,
        0.0, s, c);
}

// 功能：构造绕 Y 轴旋转的矩阵。
static cv::Matx33d rotationY(double rad) {
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return cv::Matx33d(
        c, 0.0, s,
        0.0, 1.0, 0.0,
        -s, 0.0, c);
}

// 功能：构造绕世界坐标 Z 轴旋转的矩阵。
static cv::Matx33d rotationZWorld(double rad) {
    const double c = std::cos(rad);
    const double s = std::sin(rad);
    return cv::Matx33d(
        c, -s, 0.0,
        s, c, 0.0,
        0.0, 0.0, 1.0);
}

// 功能：根据 heading、roll、pitch 组合机体系到世界系的旋转矩阵。
static cv::Matx33d worldFromImuFrd(double heading_compass_rad, double roll_rad, double pitch_rad) {
    const cv::Vec3d x_level(std::sin(heading_compass_rad), std::cos(heading_compass_rad), 0.0);
    const cv::Vec3d y_level(std::cos(heading_compass_rad), -std::sin(heading_compass_rad), 0.0);
    const cv::Vec3d z_level(0.0, 0.0, -1.0);

    const cv::Matx33d R_level(
        x_level[0], y_level[0], z_level[0],
        x_level[1], y_level[1], z_level[1],
        x_level[2], y_level[2], z_level[2]);
    return R_level * (rotationY(pitch_rad) * rotationX(roll_rad));
}

// 功能：根据机体系重力方向估计 roll 和 pitch。
static void rollPitchFromGravityFrd(const cv::Vec3d& gravity_body_down, double& roll_rad, double& pitch_rad) {
    roll_rad = std::atan2(gravity_body_down[1], gravity_body_down[2]);
    pitch_rad = std::atan2(-gravity_body_down[0],
                           std::sqrt(gravity_body_down[1] * gravity_body_down[1] +
                                     gravity_body_down[2] * gravity_body_down[2]));
}

// 功能：把角度约束到 `[0, 2pi)` 区间。
static double wrapToTwoPi(double rad) {
    double out = std::fmod(rad, 2.0 * CV_PI);
    if (out < 0.0) out += 2.0 * CV_PI;
    return out;
}

// 功能：从姿态矩阵反推出罗盘航向角。
static double headingCompassFromRwi(const cv::Matx33d& R_wi) {
    const cv::Matx33d R = orthonormalize(R_wi);
    const cv::Vec3d x_world(R(0, 0), R(1, 0), R(2, 0));
    return wrapToTwoPi(std::atan2(x_world[0], x_world[1]));
}

class GpsInterpolator {
public:
// 功能：`GpsInterpolator` 的构造函数，用于建立 GNSS 插值器。
    explicit GpsInterpolator(std::vector<FixSample> fixes) : fixes_(std::move(fixes)) {
        if (fixes_.empty()) {
            throw std::runtime_error("GPS series is empty.");
        }
    }

    bool interpolate(double t, FixSample& out) const {
        if (fixes_.empty()) return false;
        if (t < fixes_.front().t || t > fixes_.back().t) return false;
        auto it = std::lower_bound(
            fixes_.begin(), fixes_.end(), t,
            [](const FixSample& s, double value) { return s.t < value; });
        if (it == fixes_.begin()) {
            out = *it;
            return true;
        }
        if (it == fixes_.end()) {
            out = fixes_.back();
            return true;
        }
        const FixSample& b = *it;
        const FixSample& a = *(it - 1);
        const double span = b.t - a.t;
        if (span <= 1e-9) {
            out = a;
            return true;
        }
        const double w = (t - a.t) / span;
        out.t = t;
        out.lat = a.lat + (b.lat - a.lat) * w;
        out.lon = a.lon + (b.lon - a.lon) * w;
        out.alt = a.alt + (b.alt - a.alt) * w;
        return true;
    }

private:
    std::vector<FixSample> fixes_;
};

class ImuRollPitchTracker {
public:
// 功能：`ImuRollPitchTracker` 的构造函数，用于建立离线 IMU 姿态跟踪器。
    explicit ImuRollPitchTracker(
        std::vector<ImuSample> samples,
        const Options& opt,
        double init_query_t = std::numeric_limits<double>::quiet_NaN())
        : samples_(std::move(samples)),
          fusion_kp_(std::max(0.0, opt.imu_fusion_kp)),
          fusion_ki_(std::max(0.0, opt.imu_fusion_ki)),
          accel_lpf_alpha_(std::clamp(opt.imu_accel_lpf_alpha, 0.0, 1.0)),
          accel_min_norm_mps2_(std::max(0.0, opt.imu_accel_min_norm_mps2)),
          accel_max_norm_mps2_(std::max(opt.imu_accel_min_norm_mps2, opt.imu_accel_max_norm_mps2)),
          integral_limit_rad_s_(std::max(0.0, opt.imu_fusion_integral_limit_rad_s)),
          init_query_t_(init_query_t) {
        if (samples_.size() < 2) {
            throw std::runtime_error("Need at least two IMU samples.");
        }
        initialize();
    }

// 功能：按给定时间查询当前对象的插值或估计结果。
    bool query(double t, double& roll_deg, double& pitch_deg) {
        if (samples_.empty()) return false;
        if (t < query_start_t_ || t > samples_.back().t) return false;
        advanceBase(t);

        cv::Matx33d R_query = base_R_wi_;
        if (base_index_ + 1 < samples_.size() && t > samples_[base_index_].t) {
            const ImuSample& a = samples_[base_index_];
            const ImuSample& b = samples_[base_index_ + 1];
            const double span = b.t - a.t;
            if (span > 1e-9) {
                const double w = (t - a.t) / span;
                const cv::Vec3d gyro = a.gyro * (1.0 - w) + b.gyro * w;
                const cv::Vec3d accel = a.accel * (1.0 - w) + b.accel * w;
                cv::Vec3d integral_correction = integral_correction_body_;
                cv::Vec3d accel_lp = accel_lp_body_;
                bool accel_lp_ready = accel_lp_ready_;
                advanceOrientation(R_query,
                                   gyro,
                                   accel,
                                   t - a.t,
                                   integral_correction,
                                   accel_lp,
                                   accel_lp_ready,
                                   nullptr);
            }
        }

        const cv::Vec3d down_world(0.0, 0.0, -1.0);
        const cv::Vec3d down_body = orthonormalize(R_query).t() * down_world;
        double roll_rad = 0.0;
        double pitch_rad = 0.0;
        rollPitchFromGravityFrd(down_body, roll_rad, pitch_rad);
        roll_deg = roll_rad * 180.0 / CV_PI;
        pitch_deg = pitch_rad * 180.0 / CV_PI;
        return std::isfinite(roll_deg) && std::isfinite(pitch_deg);
    }

private:
// 功能：完成对象的初始化准备。
    void initialize() {
        size_t init_index = 0;
        if (std::isfinite(init_query_t_)) {
            const auto it = std::lower_bound(
                samples_.begin(),
                samples_.end(),
                init_query_t_,
                [](const ImuSample& sample, double t) { return sample.t < t; });
            if (it == samples_.end()) {
                throw std::runtime_error("No IMU sample found near requested init time.");
            }
            init_index = static_cast<size_t>(it - samples_.begin());
            if (init_index > 0 && samples_[init_index].t > init_query_t_) {
                --init_index;
            }
        }

        size_t count = std::min<size_t>(samples_.size() - init_index, 20);
        cv::Vec3d accel_sum(0.0, 0.0, 0.0);
        size_t used = 0;
        for (size_t i = 0; i < count; ++i) {
            const ImuSample& sample = samples_[init_index + i];
            if (!std::isfinite(sample.accel[0]) ||
                !std::isfinite(sample.accel[1]) ||
                !std::isfinite(sample.accel[2])) {
                continue;
            }
            accel_sum += sample.accel;
            ++used;
        }
        if (used == 0) {
            throw std::runtime_error("Failed to initialize IMU attitude from accelerometer.");
        }

        const cv::Vec3d accel_avg = accel_sum * (1.0 / static_cast<double>(used));
        // The IMU message in this bag follows the common convention where the
        // accelerometer reports a vector opposite to gravity at rest.
        const cv::Vec3d gravity_body_down = -accel_avg;
        double roll_rad = 0.0;
        double pitch_rad = 0.0;
        rollPitchFromGravityFrd(gravity_body_down, roll_rad, pitch_rad);
        base_R_wi_ = orthonormalize(worldFromImuFrd(0.0, roll_rad, pitch_rad));
        accel_lp_body_ = accel_avg;
        accel_lp_ready_ = true;
        integral_correction_body_ = cv::Vec3d(0.0, 0.0, 0.0);
        base_index_ = init_index;
        query_start_t_ = std::isfinite(init_query_t_) ? std::max(samples_.front().t, init_query_t_)
                                                      : samples_.front().t;
        step_counter_ = 0;
    }

// 功能：把内部基准状态推进到目标时刻附近。
    void advanceBase(double t) {
        while (base_index_ + 1 < samples_.size() && samples_[base_index_ + 1].t <= t) {
            const ImuSample& a = samples_[base_index_];
            const ImuSample& b = samples_[base_index_ + 1];
            const double dt = b.t - a.t;
            if (dt > 1e-9) {
                const cv::Vec3d gyro = (a.gyro + b.gyro) * 0.5;
                const cv::Vec3d accel = (a.accel + b.accel) * 0.5;
                advanceOrientation(base_R_wi_,
                                   gyro,
                                   accel,
                                   dt,
                                   integral_correction_body_,
                                   accel_lp_body_,
                                   accel_lp_ready_,
                                   &step_counter_);
            }
            ++base_index_;
        }
    }

// 功能：实现 `computeAccelError` 对应的功能。
    bool computeAccelError(const cv::Matx33d& R_wi,
                           const cv::Vec3d& accel_meas,
                           cv::Vec3d& accel_lp,
                           bool& accel_lp_ready,
                           cv::Vec3d& error_body) const {
        if (!isFiniteVec(accel_meas)) return false;

        if (!accel_lp_ready) {
            accel_lp = accel_meas;
            accel_lp_ready = true;
        } else {
            accel_lp = accel_lp * (1.0 - accel_lpf_alpha_) + accel_meas * accel_lpf_alpha_;
        }

        const double raw_norm = cv::norm(accel_meas);
        const double filt_norm = cv::norm(accel_lp);
        if (raw_norm < accel_min_norm_mps2_ || raw_norm > accel_max_norm_mps2_) return false;
        if (filt_norm < accel_min_norm_mps2_ || filt_norm > accel_max_norm_mps2_) return false;

        const cv::Vec3d measured_down_body = normalizeVec(-accel_lp);
        const cv::Vec3d predicted_down_body = normalizeVec(R_wi.t() * cv::Vec3d(0.0, 0.0, -1.0));
        error_body = measured_down_body.cross(predicted_down_body);
        return isFiniteVec(error_body);
    }

// 功能：实现 `advanceOrientation` 对应的功能。
    void advanceOrientation(cv::Matx33d& R_wi,
                            const cv::Vec3d& gyro_meas,
                            const cv::Vec3d& accel_meas,
                            double dt,
                            cv::Vec3d& integral_correction,
                            cv::Vec3d& accel_lp,
                            bool& accel_lp_ready,
                            size_t* step_counter) const {
        if (dt <= 1e-9) return;

        cv::Vec3d corrected_gyro = gyro_meas;
        cv::Vec3d accel_error_body(0.0, 0.0, 0.0);
        if (computeAccelError(R_wi, accel_meas, accel_lp, accel_lp_ready, accel_error_body)) {
            if (fusion_ki_ > 0.0) {
                integral_correction += accel_error_body * (fusion_ki_ * dt);
                const double integral_norm = cv::norm(integral_correction);
                if (integral_limit_rad_s_ > 0.0 && integral_norm > integral_limit_rad_s_) {
                    integral_correction *= integral_limit_rad_s_ / integral_norm;
                }
            }
            corrected_gyro += fusion_kp_ * accel_error_body + integral_correction;
        }

        R_wi = R_wi * so3Exp(corrected_gyro * dt);
        if (step_counter) {
            ++(*step_counter);
        }
        if (step_counter && ((*step_counter % 200) == 0)) {
            R_wi = orthonormalize(R_wi);
        }
    }

    std::vector<ImuSample> samples_;
    double fusion_kp_ = 0.0;
    double fusion_ki_ = 0.0;
    double accel_lpf_alpha_ = 0.05;
    double accel_min_norm_mps2_ = 8.8;
    double accel_max_norm_mps2_ = 10.8;
    double integral_limit_rad_s_ = 0.2;
    double init_query_t_ = std::numeric_limits<double>::quiet_NaN();
    double query_start_t_ = 0.0;
    cv::Vec3d integral_correction_body_ = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d accel_lp_body_ = cv::Vec3d(0.0, 0.0, 0.0);
    bool accel_lp_ready_ = false;
    size_t base_index_ = 0;
    size_t step_counter_ = 0;
    cv::Matx33d base_R_wi_ = cv::Matx33d::eye();
};

// 功能：在 fix 序列上按时间插值出指定时刻的位置。
static bool interpolateFixSeries(const std::deque<FixSample>& fixes, double t, FixSample& out) {
    if (fixes.empty()) return false;
    if (t < fixes.front().t || t > fixes.back().t) return false;

    auto it = std::lower_bound(
        fixes.begin(), fixes.end(), t,
        [](const FixSample& s, double value) { return s.t < value; });
    if (it == fixes.begin()) {
        out = *it;
        return true;
    }
    if (it == fixes.end()) {
        out = fixes.back();
        return true;
    }

    const FixSample& b = *it;
    const FixSample& a = *(it - 1);
    const double span = b.t - a.t;
    if (span <= 1e-9) {
        out = a;
        return true;
    }

    const double w = (t - a.t) / span;
    out.t = t;
    out.lat = a.lat + (b.lat - a.lat) * w;
    out.lon = a.lon + (b.lon - a.lon) * w;
    out.alt = a.alt + (b.alt - a.alt) * w;
    return true;
}

class OnlineImuRollPitchTracker {
public:
    struct PreintegrationResult {
        cv::Vec3d delta_enu = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d delta_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        double mean_speed_mps = 0.0;
        double final_speed_mps = 0.0;
    };

// 功能：`OnlineImuRollPitchTracker` 的构造函数，用于建立在线 IMU 跟踪器。
    explicit OnlineImuRollPitchTracker(const Options& opt)
        : fusion_kp_(std::max(0.0, opt.imu_fusion_kp)),
          fusion_ki_(std::max(0.0, opt.imu_fusion_ki)),
          accel_lpf_alpha_(std::clamp(opt.imu_accel_lpf_alpha, 0.0, 1.0)),
          accel_min_norm_mps2_(std::max(0.0, opt.imu_accel_min_norm_mps2)),
          accel_max_norm_mps2_(std::max(opt.imu_accel_min_norm_mps2, opt.imu_accel_max_norm_mps2)),
          max_preintegration_gap_sec_(std::max(1e-4, opt.imu_preintegration_max_gap_sec)),
          integral_limit_rad_s_(std::max(0.0, opt.imu_fusion_integral_limit_rad_s)),
          bias_lpf_alpha_(std::clamp(opt.imu_bias_lpf_alpha, 0.0, 1.0)),
          bias_gyro_threshold_rad_s_(std::max(0.0, opt.imu_bias_gyro_threshold_rad_s)),
          bias_residual_threshold_mps2_(std::max(0.0, opt.imu_bias_residual_threshold_mps2)),
          bias_max_abs_mps2_(std::max(0.0, opt.imu_bias_max_abs_mps2)),
          velocity_damping_per_sec_(std::max(0.0, opt.imu_velocity_damping_per_sec)),
          vertical_velocity_damping_per_sec_(std::max(0.0, opt.imu_vertical_velocity_damping_per_sec)),
          max_speed_mps_(std::max(0.0, opt.imu_max_speed_mps)),
          max_vertical_speed_mps_(std::max(0.0, opt.imu_max_vertical_speed_mps)),
          history_keep_sec_(opt.input_mode == "offline"
                                ? std::numeric_limits<double>::infinity()
                                : std::max(30.0, opt.window_sec * 4.0 + 10.0)) {}

// 功能：向在线 IMU 跟踪器追加一条新的 IMU 样本。
    void addSample(const ImuSample& sample) {
        if (!std::isfinite(sample.t) || !isFiniteVec(sample.gyro) || !isFiniteVec(sample.accel)) return;

        if (!initialized_) {
            if (!init_samples_.empty() && sample.t <= init_samples_.back().t) return;
            init_samples_.push_back(sample);
            if (init_samples_.size() >= kInitSampleCount) initializeFromInitBuffer();
            return;
        }

        if (!states_.empty() && sample.t <= states_.back().sample.t) return;

        State next = states_.back();
        const ImuSample& a = next.sample;
        const double dt = sample.t - a.t;
        if (dt <= 1e-9) return;

        const cv::Vec3d gyro = (a.gyro + sample.gyro) * 0.5;
        const cv::Vec3d accel = (a.accel + sample.accel) * 0.5;
        advanceOrientation(next.R_wi,
                           gyro,
                           accel,
                           dt,
                           next.integral_correction_body,
                           next.accel_lp_body,
                           next.accel_lp_ready,
                           &next.step_counter);
        updateAccelBiasEstimate(next.R_wi, gyro, accel, next.accel_bias_body);
        next.sample = sample;
        states_.push_back(next);
        trimHistory(sample.t);
    }

// 功能：按给定时间查询当前对象的插值或估计结果。
    bool query(double t, double& roll_deg, double& pitch_deg) const {
        cv::Matx33d R_query = cv::Matx33d::eye();
        if (!queryOrientationAtTime(t, R_query)) return false;
        return extractRollPitch(R_query, roll_deg, pitch_deg);
    }

// 功能：实现 `queryHeadingCompassRelativeRad` 对应的功能。
    bool queryHeadingCompassRelativeRad(double t, double& heading_rad) const {
        cv::Matx33d R_query = cv::Matx33d::eye();
        if (!queryOrientationAtTime(t, R_query)) return false;
        heading_rad = headingCompassFromRwi(R_query);
        return std::isfinite(heading_rad);
    }

// 功能：仅使用不晚于目标时刻的 IMU 历史查询相对航向，避免使用未来样本插值。
    bool queryHeadingCompassRelativeRadCausal(double t, double& heading_rad) const {
        cv::Matx33d R_query = cv::Matx33d::eye();
        if (!queryOrientationAtTimeCausal(t, R_query)) return false;
        heading_rad = headingCompassFromRwi(R_query);
        return std::isfinite(heading_rad);
    }

// 功能：实现 `queryOrientationMatrixAtTime` 对应的功能。
    bool queryOrientationMatrixAtTime(double t, cv::Matx33d& R_query) const {
        return queryOrientationAtTime(t, R_query);
    }

// 功能：对指定时间段内的 IMU 数据做相对 ENU 预积分。
    bool preintegrateRelativeEnu(double t_from,
                                 double t_to,
                                 double anchor_heading_compass_rad,
                                 double anchor_heading_rel_rad,
                                 const cv::Vec3d& initial_velocity_enu,
                                 PreintegrationResult& result) const {
        result = PreintegrationResult{};
        if (!(t_to >= t_from)) return false;
        if (t_to - t_from <= 1e-9) return true;
        if (!std::isfinite(anchor_heading_compass_rad) || !std::isfinite(anchor_heading_rel_rad)) {
            return false;
        }

        State prev_state;
        if (!queryStateAtTime(t_from, prev_state)) return false;

        const double yaw_offset_rad = global_heading::wrapToPi(anchor_heading_rel_rad -
                                                             anchor_heading_compass_rad);
        const cv::Matx33d R_enu_tracker = rotationZWorld(yaw_offset_rad);
        const cv::Vec3d gravity_enu(0.0, 0.0, -9.80665);
        const cv::Vec3d initial_vel = initial_velocity_enu;
        cv::Vec3d velocity_enu = initial_vel;
        cv::Vec3d position_enu(0.0, 0.0, 0.0);

        auto it = std::lower_bound(
            states_.begin(), states_.end(), t_from,
            [](const State& state, double value) { return state.sample.t < value; });
        if (it != states_.begin()) --it;

        double current_t = t_from;
        while (current_t < t_to - 1e-9) {
            const State curr_state = prev_state;
            State next_state;

            auto next_it = std::upper_bound(
                states_.begin(), states_.end(), current_t,
                [](double value, const State& state) { return value < state.sample.t; });
            if (next_it == states_.end() || next_it->sample.t >= t_to) {
                if (!queryStateAtTime(t_to, next_state)) return false;
            } else {
                next_state = *next_it;
            }

            const double next_t = std::min(t_to, next_state.sample.t);
            const double dt = next_t - current_t;
            if (dt <= 1e-9) break;
            if (dt > max_preintegration_gap_sec_) return false;

            const cv::Vec3d accel_body = (curr_state.sample.accel + next_state.sample.accel) * 0.5;
            const cv::Vec3d accel_bias_body =
                (curr_state.accel_bias_body + next_state.accel_bias_body) * 0.5;
            const cv::Vec3d accel_body_unbiased = accel_body - accel_bias_body;
            const cv::Matx33d R_mid = orthonormalize((curr_state.R_wi + next_state.R_wi) * 0.5);
            const cv::Vec3d accel_enu = R_enu_tracker * (R_mid * accel_body_unbiased) + gravity_enu;
            position_enu += velocity_enu * dt + accel_enu * (0.5 * dt * dt);
            velocity_enu += accel_enu * dt;
            applyVelocityConstraints(dt, velocity_enu);

            current_t = next_t;
            prev_state = next_state;
        }

        result.delta_enu = position_enu;
        result.delta_vel_enu = velocity_enu - initial_vel;
        const double total_dt = t_to - t_from;
        if (total_dt > 1e-9) {
            result.mean_speed_mps = cv::norm(position_enu) / total_dt;
        }
        result.final_speed_mps = cv::norm(velocity_enu);
        return isFiniteVec(result.delta_enu) && isFiniteVec(result.delta_vel_enu) &&
               std::isfinite(result.mean_speed_mps) &&
               std::isfinite(result.final_speed_mps);
    }

// 功能：仅使用目标时刻之前的 IMU 历史做相对 ENU 预积分，避免使用未来样本。
    bool preintegrateRelativeEnuCausal(double t_from,
                                       double t_to,
                                       double anchor_heading_compass_rad,
                                       double anchor_heading_rel_rad,
                                       const cv::Vec3d& initial_velocity_enu,
                                       PreintegrationResult& result) const {
        result = PreintegrationResult{};
        if (!(t_to >= t_from)) return false;
        if (t_to - t_from <= 1e-9) return true;
        if (!std::isfinite(anchor_heading_compass_rad) || !std::isfinite(anchor_heading_rel_rad)) {
            return false;
        }

        State prev_state;
        if (!queryStateAtTimeCausal(t_from, prev_state)) return false;

        const double yaw_offset_rad = global_heading::wrapToPi(anchor_heading_rel_rad -
                                                             anchor_heading_compass_rad);
        const cv::Matx33d R_enu_tracker = rotationZWorld(yaw_offset_rad);
        const cv::Vec3d gravity_enu(0.0, 0.0, -9.80665);
        const cv::Vec3d initial_vel = initial_velocity_enu;
        cv::Vec3d velocity_enu = initial_vel;
        cv::Vec3d position_enu(0.0, 0.0, 0.0);

        double current_t = t_from;
        while (current_t < t_to - 1e-9) {
            const State curr_state = prev_state;
            State next_state;

            auto next_it = std::upper_bound(
                states_.begin(), states_.end(), current_t,
                [](double value, const State& state) { return value < state.sample.t; });
            if (next_it == states_.end() || next_it->sample.t > t_to) {
                if (!queryStateAtTimeCausal(t_to, next_state)) return false;
            } else {
                next_state = *next_it;
            }

            const double next_t = std::min(t_to, next_state.sample.t);
            const double dt = next_t - current_t;
            if (dt <= 1e-9) break;
            if (dt > max_preintegration_gap_sec_) return false;

            const cv::Vec3d accel_body = (curr_state.sample.accel + next_state.sample.accel) * 0.5;
            const cv::Vec3d accel_bias_body =
                (curr_state.accel_bias_body + next_state.accel_bias_body) * 0.5;
            const cv::Vec3d accel_body_unbiased = accel_body - accel_bias_body;
            const cv::Matx33d R_mid = orthonormalize((curr_state.R_wi + next_state.R_wi) * 0.5);
            const cv::Vec3d accel_enu = R_enu_tracker * (R_mid * accel_body_unbiased) + gravity_enu;
            position_enu += velocity_enu * dt + accel_enu * (0.5 * dt * dt);
            velocity_enu += accel_enu * dt;
            applyVelocityConstraints(dt, velocity_enu);

            current_t = next_t;
            prev_state = next_state;
        }

        result.delta_enu = position_enu;
        result.delta_vel_enu = velocity_enu - initial_vel;
        const double total_dt = t_to - t_from;
        if (total_dt > 1e-9) {
            result.mean_speed_mps = cv::norm(position_enu) / total_dt;
        }
        result.final_speed_mps = cv::norm(velocity_enu);
        return isFiniteVec(result.delta_enu) && isFiniteVec(result.delta_vel_enu) &&
               std::isfinite(result.mean_speed_mps) &&
               std::isfinite(result.final_speed_mps);
    }

// 功能：判断当前跟踪器是否已经完成初始化。
    bool initialized() const { return initialized_ && !states_.empty(); }
    double startTime() const {
        if (states_.empty()) return std::numeric_limits<double>::quiet_NaN();
        return states_.front().sample.t;
    }
// 功能：实现 `latestTime` 对应的功能。
    double latestTime() const {
        if (states_.empty()) return std::numeric_limits<double>::quiet_NaN();
        return states_.back().sample.t;
    }

private:
    struct State {
        ImuSample sample;
        cv::Matx33d R_wi = cv::Matx33d::eye();
        cv::Vec3d integral_correction_body = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d accel_lp_body = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d accel_bias_body = cv::Vec3d(0.0, 0.0, 0.0);
        bool accel_lp_ready = false;
        size_t step_counter = 0;
    };

// 功能：实现 `queryStateAtTime` 对应的功能。
    bool queryStateAtTime(double t, State& out) const {
        if (!initialized_ || states_.empty()) return false;
        if (t < states_.front().sample.t || t > states_.back().sample.t) return false;

        auto it = std::lower_bound(
            states_.begin(), states_.end(), t,
            [](const State& state, double value) { return state.sample.t < value; });
        if (it == states_.begin()) {
            out = *it;
            return true;
        }
        if (it == states_.end()) {
            out = states_.back();
            return true;
        }
        if (std::abs(it->sample.t - t) <= 1e-9) {
            out = *it;
            return true;
        }

        const State& a = *(it - 1);
        const State& b = *it;
        const double span = b.sample.t - a.sample.t;
        if (span <= 1e-9) {
            out = a;
            return true;
        }

        const double w = (t - a.sample.t) / span;
        out = a;
        out.sample.t = t;
        out.sample.gyro = a.sample.gyro * (1.0 - w) + b.sample.gyro * w;
        out.sample.accel = a.sample.accel * (1.0 - w) + b.sample.accel * w;
        out.accel_bias_body = a.accel_bias_body * (1.0 - w) + b.accel_bias_body * w;
        advanceOrientation(out.R_wi,
                           out.sample.gyro,
                           out.sample.accel,
                           t - a.sample.t,
                           out.integral_correction_body,
                           out.accel_lp_body,
                           out.accel_lp_ready,
                           nullptr);
        return true;
    }

// 功能：仅使用目标时刻及之前的 IMU 样本查询状态，末段采用零阶保持积分。
    bool queryStateAtTimeCausal(double t, State& out) const {
        if (!initialized_ || states_.empty()) return false;
        if (t < states_.front().sample.t) return false;

        auto it = std::upper_bound(
            states_.begin(), states_.end(), t,
            [](double value, const State& state) { return value < state.sample.t; });
        if (it == states_.begin()) return false;

        if (it == states_.end()) {
            const State& last = states_.back();
            out = last;
            if (std::abs(last.sample.t - t) <= 1e-9) {
                return true;
            }
            const double dt = t - last.sample.t;
            if (dt <= 1e-9) return true;
            if (dt > max_preintegration_gap_sec_) return false;
            out.sample.t = t;
            advanceOrientation(out.R_wi,
                               last.sample.gyro,
                               last.sample.accel,
                               dt,
                               out.integral_correction_body,
                               out.accel_lp_body,
                               out.accel_lp_ready,
                               nullptr);
            return true;
        }

        const State& a = *(it - 1);
        out = a;
        if (std::abs(a.sample.t - t) <= 1e-9) {
            return true;
        }

        const double dt = t - a.sample.t;
        if (dt <= 1e-9) return true;
        out.sample.t = t;
        advanceOrientation(out.R_wi,
                           a.sample.gyro,
                           a.sample.accel,
                           dt,
                           out.integral_correction_body,
                           out.accel_lp_body,
                           out.accel_lp_ready,
                           nullptr);
        return true;
    }

// 功能：实现 `queryOrientationAtTime` 对应的功能。
    bool queryOrientationAtTime(double t, cv::Matx33d& R_query) const {
        State query_state;
        if (!queryStateAtTime(t, query_state)) return false;
        R_query = query_state.R_wi;
        return true;
    }

// 功能：仅使用目标时刻之前的 IMU 历史查询姿态矩阵。
    bool queryOrientationAtTimeCausal(double t, cv::Matx33d& R_query) const {
        State query_state;
        if (!queryStateAtTimeCausal(t, query_state)) return false;
        R_query = query_state.R_wi;
        return true;
    }

// 功能：利用初始化缓冲区完成初始姿态估计。
    void initializeFromInitBuffer() {
        cv::Vec3d accel_sum(0.0, 0.0, 0.0);
        size_t used = 0;
        for (const ImuSample& sample : init_samples_) {
            if (!isFiniteVec(sample.accel)) continue;
            accel_sum += sample.accel;
            ++used;
        }
        if (used == 0) return;

        const cv::Vec3d accel_avg = accel_sum * (1.0 / static_cast<double>(used));
        const cv::Vec3d gravity_body_down = -accel_avg;
        double roll_rad = 0.0;
        double pitch_rad = 0.0;
        rollPitchFromGravityFrd(gravity_body_down, roll_rad, pitch_rad);

        State init_state;
        init_state.sample = init_samples_.back();
        init_state.R_wi = orthonormalize(worldFromImuFrd(0.0, roll_rad, pitch_rad));
        init_state.accel_lp_body = accel_avg;
        init_state.accel_bias_body = cv::Vec3d(0.0, 0.0, 0.0);
        init_state.accel_lp_ready = true;
        states_.clear();
        states_.push_back(init_state);
        initialized_ = true;
        init_samples_.clear();
    }

    void trimHistory(double newest_t) {
        if (!std::isfinite(history_keep_sec_)) return;
        while (states_.size() > 2 && newest_t - states_[1].sample.t > history_keep_sec_) {
            states_.pop_front();
        }
    }

    void updateAccelBiasEstimate(const cv::Matx33d& R_wi,
                                 const cv::Vec3d& gyro_meas,
                                 const cv::Vec3d& accel_meas,
                                 cv::Vec3d& accel_bias_body) const {
        if (bias_lpf_alpha_ <= 0.0) return;
        if (!isFiniteVec(gyro_meas) || !isFiniteVec(accel_meas) || !isFiniteVec(accel_bias_body)) return;
        if (cv::norm(gyro_meas) > bias_gyro_threshold_rad_s_) return;

        const cv::Vec3d expected_accel_body = orthonormalize(R_wi).t() * cv::Vec3d(0.0, 0.0, 9.80665);
        const cv::Vec3d residual = accel_meas - expected_accel_body;
        if (!isFiniteVec(residual)) return;
        if (cv::norm(residual) > bias_residual_threshold_mps2_) return;

        accel_bias_body = accel_bias_body * (1.0 - bias_lpf_alpha_) + residual * bias_lpf_alpha_;
        for (int i = 0; i < 3; ++i) {
            accel_bias_body[i] = std::clamp(accel_bias_body[i], -bias_max_abs_mps2_, bias_max_abs_mps2_);
        }
    }

// 功能：实现 `applyVelocityConstraints` 对应的功能。
    void applyVelocityConstraints(double dt, cv::Vec3d& velocity_enu) const {
        if (!(dt > 1e-9) || !isFiniteVec(velocity_enu)) return;

        const double horiz_decay = std::exp(-velocity_damping_per_sec_ * dt);
        const double vert_decay = std::exp(-vertical_velocity_damping_per_sec_ * dt);
        velocity_enu[0] *= horiz_decay;
        velocity_enu[1] *= horiz_decay;
        velocity_enu[2] *= vert_decay;

        const double horiz_speed = std::hypot(velocity_enu[0], velocity_enu[1]);
        if (max_speed_mps_ > 0.0 && horiz_speed > max_speed_mps_) {
            const double scale = max_speed_mps_ / std::max(1e-9, horiz_speed);
            velocity_enu[0] *= scale;
            velocity_enu[1] *= scale;
        }
        if (max_vertical_speed_mps_ > 0.0) {
            velocity_enu[2] = std::clamp(
                velocity_enu[2], -max_vertical_speed_mps_, max_vertical_speed_mps_);
        }
    }

// 功能：实现 `computeAccelError` 对应的功能。
    bool computeAccelError(const cv::Matx33d& R_wi,
                           const cv::Vec3d& accel_meas,
                           cv::Vec3d& accel_lp,
                           bool& accel_lp_ready,
                           cv::Vec3d& error_body) const {
        if (!isFiniteVec(accel_meas)) return false;

        if (!accel_lp_ready) {
            accel_lp = accel_meas;
            accel_lp_ready = true;
        } else {
            accel_lp = accel_lp * (1.0 - accel_lpf_alpha_) + accel_meas * accel_lpf_alpha_;
        }

        const double raw_norm = cv::norm(accel_meas);
        const double filt_norm = cv::norm(accel_lp);
        if (raw_norm < accel_min_norm_mps2_ || raw_norm > accel_max_norm_mps2_) return false;
        if (filt_norm < accel_min_norm_mps2_ || filt_norm > accel_max_norm_mps2_) return false;

        const cv::Vec3d measured_down_body = normalizeVec(-accel_lp);
        const cv::Vec3d predicted_down_body = normalizeVec(R_wi.t() * cv::Vec3d(0.0, 0.0, -1.0));
        error_body = measured_down_body.cross(predicted_down_body);
        return isFiniteVec(error_body);
    }

// 功能：实现 `advanceOrientation` 对应的功能。
    void advanceOrientation(cv::Matx33d& R_wi,
                            const cv::Vec3d& gyro_meas,
                            const cv::Vec3d& accel_meas,
                            double dt,
                            cv::Vec3d& integral_correction,
                            cv::Vec3d& accel_lp,
                            bool& accel_lp_ready,
                            size_t* step_counter) const {
        if (dt <= 1e-9) return;

        cv::Vec3d corrected_gyro = gyro_meas;
        cv::Vec3d accel_error_body(0.0, 0.0, 0.0);
        if (computeAccelError(R_wi, accel_meas, accel_lp, accel_lp_ready, accel_error_body)) {
            if (fusion_ki_ > 0.0) {
                integral_correction += accel_error_body * (fusion_ki_ * dt);
                const double integral_norm = cv::norm(integral_correction);
                if (integral_limit_rad_s_ > 0.0 && integral_norm > integral_limit_rad_s_) {
                    integral_correction *= integral_limit_rad_s_ / integral_norm;
                }
            }
            corrected_gyro += fusion_kp_ * accel_error_body + integral_correction;
        }

        R_wi = R_wi * so3Exp(corrected_gyro * dt);
        if (step_counter) {
            ++(*step_counter);
            if (((*step_counter) % 200) == 0) R_wi = orthonormalize(R_wi);
        }
    }

// 功能：从姿态矩阵中提取 roll 和 pitch。
    static bool extractRollPitch(const cv::Matx33d& R_wi, double& roll_deg, double& pitch_deg) {
        const cv::Vec3d down_world(0.0, 0.0, -1.0);
        const cv::Vec3d down_body = orthonormalize(R_wi).t() * down_world;
        double roll_rad = 0.0;
        double pitch_rad = 0.0;
        rollPitchFromGravityFrd(down_body, roll_rad, pitch_rad);
        roll_deg = roll_rad * 180.0 / CV_PI;
        pitch_deg = pitch_rad * 180.0 / CV_PI;
        return std::isfinite(roll_deg) && std::isfinite(pitch_deg);
    }

    static constexpr size_t kInitSampleCount = 20;

    double fusion_kp_ = 0.0;
    double fusion_ki_ = 0.0;
    double accel_lpf_alpha_ = 0.05;
    double accel_min_norm_mps2_ = 8.8;
    double accel_max_norm_mps2_ = 10.8;
    double max_preintegration_gap_sec_ = 0.05;
    double integral_limit_rad_s_ = 0.2;
    double bias_lpf_alpha_ = 0.01;
    double bias_gyro_threshold_rad_s_ = 0.35;
    double bias_residual_threshold_mps2_ = 0.50;
    double bias_max_abs_mps2_ = 0.80;
    double velocity_damping_per_sec_ = 0.30;
    double vertical_velocity_damping_per_sec_ = 0.60;
    double max_speed_mps_ = 18.0;
    double max_vertical_speed_mps_ = 6.0;
    double history_keep_sec_ = 30.0;
    bool initialized_ = false;
    std::deque<ImuSample> init_samples_;
    std::deque<State> states_;
};

// 功能：实现 `ensureGdalRegistered` 对应的功能。
static void ensureGdalRegistered() {
    static bool registered = false;
    if (!registered) {
        GDALAllRegister();
        registered = true;
    }
}

class GeoRaster {
public:
// 功能：`GeoRaster` 的构造函数，负责打开并初始化地图或 DEM 栅格。
    GeoRaster(const std::string& path, bool load_band1_float)
        : path_(path) {
        ensureGdalRegistered();
        ds_ = static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly));
        if (!ds_) {
            throw std::runtime_error("GDALOpen failed: " + path);
        }
        width_ = ds_->GetRasterXSize();
        height_ = ds_->GetRasterYSize();
        band_count_ = ds_->GetRasterCount();
        if (band_count_ <= 0) {
            throw std::runtime_error("Raster has no bands: " + path);
        }

        if (ds_->GetGeoTransform(gt_) != CE_None) {
            throw std::runtime_error("Raster has no geotransform: " + path);
        }
        if (!GDALInvGeoTransform(gt_, inv_gt_)) {
            throw std::runtime_error("Raster geotransform is not invertible: " + path);
        }

        const char* proj = ds_->GetProjectionRef();
        if (proj && proj[0] != '\0') {
            projection_wkt_ = proj;
            if (raster_srs_.importFromWkt(projection_wkt_.c_str()) != OGRERR_NONE) {
                throw std::runtime_error("Failed to parse raster projection: " + path);
            }
        } else {
            if (raster_srs_.importFromEPSG(4326) != OGRERR_NONE) {
                throw std::runtime_error("Failed to initialize EPSG:4326.");
            }
        }
        if (wgs84_srs_.importFromEPSG(4326) != OGRERR_NONE) {
            throw std::runtime_error("Failed to initialize WGS84 SRS.");
        }

#if GDAL_VERSION_NUM >= 3000000
        raster_srs_.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        wgs84_srs_.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

        to_raster_ = OGRCreateCoordinateTransformation(&wgs84_srs_, &raster_srs_);
        to_wgs84_ = OGRCreateCoordinateTransformation(&raster_srs_, &wgs84_srs_);
        if (!to_raster_ || !to_wgs84_) {
            throw std::runtime_error("Failed to create raster coordinate transforms: " + path);
        }

        GDALRasterBand* band1 = ds_->GetRasterBand(1);
        int has_success = 0;
        nodata_ = band1->GetNoDataValue(&has_success);
        has_nodata_ = (has_success != 0);

        if (load_band1_float) {
            band1_float_ = cv::Mat(height_, width_, CV_32F);
            CPLErr err = band1->RasterIO(
                GF_Read, 0, 0, width_, height_,
                band1_float_.data, width_, height_, GDT_Float32,
                0, 0, nullptr);
            if (err != CE_None) {
                throw std::runtime_error("Failed to read DEM band: " + path);
            }
        }
    }

    ~GeoRaster() {
        if (to_raster_) OCTDestroyCoordinateTransformation(to_raster_);
        if (to_wgs84_) OCTDestroyCoordinateTransformation(to_wgs84_);
        if (ds_) GDALClose(ds_);
    }

// 功能：`GeoRaster` 的构造函数，负责打开并初始化地图或 DEM 栅格。
    GeoRaster(const GeoRaster&) = delete;
    GeoRaster& operator=(const GeoRaster&) = delete;

// 功能：实现 `lonLatToPixel` 对应的功能。
    bool lonLatToPixel(double lon, double lat, double& px, double& py) const {
        double x = lon;
        double y = lat;
        double z = 0.0;
        if (!to_raster_ || !to_raster_->Transform(1, &x, &y, &z)) return false;
        double inv_gt_local[6];
        std::copy(std::begin(inv_gt_), std::end(inv_gt_), inv_gt_local);
        GDALApplyGeoTransform(inv_gt_local, x, y, &px, &py);
        return true;
    }

// 功能：实现 `pixelToLonLat` 对应的功能。
    bool pixelToLonLat(double px, double py, double& lon, double& lat) const {
        double x = 0.0;
        double y = 0.0;
        double gt_local[6];
        std::copy(std::begin(gt_), std::end(gt_), gt_local);
        GDALApplyGeoTransform(gt_local, px, py, &x, &y);
        double z = 0.0;
        if (!to_wgs84_ || !to_wgs84_->Transform(1, &x, &y, &z)) return false;
        lon = x;
        lat = y;
        return true;
    }

// 功能：实现 `sampleBand1BilinearLonLat` 对应的功能。
    bool sampleBand1BilinearLonLat(double lon, double lat, double& value) const {
        if (band1_float_.empty()) return false;
        double px = 0.0;
        double py = 0.0;
        if (!lonLatToPixel(lon, lat, px, py)) return false;
        if (px < 0.0 || py < 0.0 ||
            px > static_cast<double>(width_ - 1) ||
            py > static_cast<double>(height_ - 1)) {
            return false;
        }

        const int x0 = static_cast<int>(std::floor(px));
        const int y0 = static_cast<int>(std::floor(py));
        const int x1 = std::min(x0 + 1, width_ - 1);
        const int y1 = std::min(y0 + 1, height_ - 1);
        const float v00 = band1_float_.at<float>(y0, x0);
        const float v10 = band1_float_.at<float>(y0, x1);
        const float v01 = band1_float_.at<float>(y1, x0);
        const float v11 = band1_float_.at<float>(y1, x1);

        auto invalid = [&](float v) {
            if (!std::isfinite(v)) return true;
            if (has_nodata_ && std::abs(static_cast<double>(v) - nodata_) < 1e-5) return true;
            return false;
        };
        if (invalid(v00) || invalid(v10) || invalid(v01) || invalid(v11)) return false;

        const double tx = px - static_cast<double>(x0);
        const double ty = py - static_cast<double>(y0);
        const double v0 = static_cast<double>(v00) * (1.0 - tx) + static_cast<double>(v10) * tx;
        const double v1 = static_cast<double>(v01) * (1.0 - tx) + static_cast<double>(v11) * tx;
        value = v0 * (1.0 - ty) + v1 * ty;
        return true;
    }

// 功能：实现 `clampWindow` 对应的功能。
    CropWindow clampWindow(const CropWindow& win) const {
        CropWindow out = win;
        out.x0 = std::max(0, std::min(out.x0, width_ - 1));
        out.y0 = std::max(0, std::min(out.y0, height_ - 1));
        out.x1 = std::max(out.x0 + 1, std::min(out.x1, width_));
        out.y1 = std::max(out.y0 + 1, std::min(out.y1, height_));
        return out;
    }

// 功能：按给定窗口从栅格中裁出子图并写到文件。
    void cropToFile(const CropWindow& raw_win, const std::string& out_path) const {
        const CropWindow win = clampWindow(raw_win);
        const int w = win.x1 - win.x0;
        const int h = win.y1 - win.y0;
        if (w <= 0 || h <= 0) {
            throw std::runtime_error("Invalid crop window for raster: " + path_);
        }

        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!driver) {
            throw std::runtime_error("GTiff driver is unavailable.");
        }

        ensureParentDir(out_path);
        GDALDataset* out_ds = driver->Create(
            out_path.c_str(), w, h, band_count_,
            ds_->GetRasterBand(1)->GetRasterDataType(),
            const_cast<char**>(kCreateOptions));
        if (!out_ds) {
            throw std::runtime_error("Failed to create output raster: " + out_path);
        }

        try {
            for (int band_idx = 1; band_idx <= band_count_; ++band_idx) {
                GDALRasterBand* in_band = ds_->GetRasterBand(band_idx);
                GDALRasterBand* out_band = out_ds->GetRasterBand(band_idx);
                const GDALDataType dt = in_band->GetRasterDataType();
                const size_t bytes = static_cast<size_t>(GDALGetDataTypeSizeBytes(dt)) *
                                     static_cast<size_t>(w) * static_cast<size_t>(h);
                std::vector<unsigned char> buffer(bytes);
                if (in_band->RasterIO(GF_Read, win.x0, win.y0, w, h,
                                      buffer.data(), w, h, dt, 0, 0, nullptr) != CE_None) {
                    throw std::runtime_error("Failed to read crop band from raster: " + path_);
                }
                if (out_band->RasterIO(GF_Write, 0, 0, w, h,
                                       buffer.data(), w, h, dt, 0, 0, nullptr) != CE_None) {
                    throw std::runtime_error("Failed to write crop band: " + out_path);
                }
                int ok = 0;
                const double nd = in_band->GetNoDataValue(&ok);
                if (ok != 0) out_band->SetNoDataValue(nd);
            }

            double new_gt[6] = {
                gt_[0] + win.x0 * gt_[1] + win.y0 * gt_[2],
                gt_[1],
                gt_[2],
                gt_[3] + win.x0 * gt_[4] + win.y0 * gt_[5],
                gt_[4],
                gt_[5]
            };
            out_ds->SetGeoTransform(new_gt);
            out_ds->SetProjection(ds_->GetProjectionRef());
            out_ds->FlushCache();
            GDALClose(out_ds);
        } catch (...) {
            GDALClose(out_ds);
            throw;
        }
    }

// 功能：实现 `width` 对应的功能。
    int width() const { return width_; }
    int height() const { return height_; }

// 功能：输出当前栅格覆盖范围的经纬度描述。
    std::string describeExtentLonLat() const {
        double lon0 = 0.0;
        double lat0 = 0.0;
        double lon1 = 0.0;
        double lat1 = 0.0;
        if (!pixelToLonLat(0.0, 0.0, lon0, lat0)) {
            return "corner transform failed";
        }
        if (!pixelToLonLat(static_cast<double>(width_ - 1),
                           static_cast<double>(height_ - 1),
                           lon1, lat1)) {
            return "corner transform failed";
        }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(9)
            << "tl=(" << lon0 << "," << lat0 << ") br=(" << lon1 << "," << lat1 << ")";
        return oss.str();
    }

private:
    static constexpr const char* kCreateOptions[2] = {"COMPRESS=LZW", nullptr};

    std::string path_;
    GDALDataset* ds_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int band_count_ = 0;
    double gt_[6]{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double inv_gt_[6]{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::string projection_wkt_;
    OGRSpatialReference raster_srs_;
    OGRSpatialReference wgs84_srs_;
    OGRCoordinateTransformation* to_raster_ = nullptr;
    OGRCoordinateTransformation* to_wgs84_ = nullptr;
    double nodata_ = std::numeric_limits<double>::quiet_NaN();
    bool has_nodata_ = false;
    cv::Mat band1_float_;
};

// 功能：把经纬高转换到 ECEF 坐标。
static cv::Vec3d wgs84ToEcef(double lon_deg, double lat_deg, double h) {
    const double a = 6378137.0;
    const double f = 1.0 / 298.257223563;
    const double e2 = f * (2.0 - f);
    const double lon = lon_deg * CV_PI / 180.0;
    const double lat = lat_deg * CV_PI / 180.0;
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);
    const double n = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    return cv::Vec3d(
        (n + h) * cos_lat * cos_lon,
        (n + h) * cos_lat * sin_lon,
        (n * (1.0 - e2) + h) * sin_lat);
}

// 功能：把 ECEF 坐标转换回经纬高。
static GeoPoint ecefToGeodetic(const cv::Vec3d& ecef) {
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
    const double lat = std::atan2(z + ep2 * b * st * st * st,
                                  p - e2 * a * ct * ct * ct);
    const double sin_lat = std::sin(lat);
    const double n = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    const double h = p / std::cos(lat) - n;
    return GeoPoint{lon * 180.0 / CV_PI, lat * 180.0 / CV_PI, h};
}

// 功能：在指定经纬度处构造 ENU 三个坐标轴。
static void enuAxes(double lon_deg,
                    double lat_deg,
                    cv::Vec3d& east,
                    cv::Vec3d& north,
                    cv::Vec3d& up) {
    const double lon = lon_deg * CV_PI / 180.0;
    const double lat = lat_deg * CV_PI / 180.0;
    east = cv::Vec3d(-std::sin(lon), std::cos(lon), 0.0);
    north = cv::Vec3d(-std::sin(lat) * std::cos(lon),
                      -std::sin(lat) * std::sin(lon),
                      std::cos(lat));
    up = cv::Vec3d(std::cos(lat) * std::cos(lon),
                   std::cos(lat) * std::sin(lon),
                   std::sin(lat));
}

// 功能：把 ECEF 坐标差转换到 ENU。
static cv::Vec3d ecefToEnu(const cv::Vec3d& p,
                           const cv::Vec3d& p0,
                           const cv::Vec3d& east,
                           const cv::Vec3d& north,
                           const cv::Vec3d& up) {
    const cv::Vec3d d = p - p0;
    return cv::Vec3d(d.dot(east), d.dot(north), d.dot(up));
}

// 功能：把 ENU 坐标差转换到 ECEF。
static cv::Vec3d enuToEcef(const cv::Vec3d& enu,
                           const cv::Vec3d& p0,
                           const cv::Vec3d& east,
                           const cv::Vec3d& north,
                           const cv::Vec3d& up) {
    return p0 + east * enu[0] + north * enu[1] + up * enu[2];
}

// 功能：把像素坐标投影为相机归一化视线方向。
static cv::Vec3d cameraRayFromPixel(const cv::Mat& K, double u, double v) {
    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);
    return normalizeVec(cv::Vec3d((u - cx) / fx, (v - cy) / fy, 1.0));
}

// 功能：根据图像对角视场角和宽高比估计水平/垂直视场角。
static bool computeImageFieldOfViewFromDiagonal(double dfov_deg,
                                                const cv::Size& image_size,
                                                double& hfov_rad,
                                                double& vfov_rad) {
    hfov_rad = std::numeric_limits<double>::quiet_NaN();
    vfov_rad = std::numeric_limits<double>::quiet_NaN();
    if (!(std::isfinite(dfov_deg) && dfov_deg > 0.0 && dfov_deg < 180.0)) return false;
    if (image_size.width <= 0 || image_size.height <= 0) return false;

    const double width = static_cast<double>(image_size.width);
    const double height = static_cast<double>(image_size.height);
    const double diagonal = std::sqrt(width * width + height * height);
    if (!(std::isfinite(diagonal) && diagonal > 1e-9)) return false;

    const double dfov_rad = dfov_deg * global_heading::DEG2RAD;
    const double tan_half_dfov = std::tan(dfov_rad * 0.5);
    if (!(std::isfinite(tan_half_dfov) && tan_half_dfov > 0.0)) return false;

    hfov_rad = 2.0 * std::atan((width / diagonal) * tan_half_dfov);
    vfov_rad = 2.0 * std::atan((height / diagonal) * tan_half_dfov);
    return std::isfinite(hfov_rad) && hfov_rad > 0.0 &&
           std::isfinite(vfov_rad) && vfov_rad > 0.0;
}

// 功能：实现 `intersectRayWithDem` 对应的功能。
static bool intersectRayWithDem(const FixSample& gps,
                                const cv::Vec3d& cam_offset_imu,
                                const cv::Matx33d& R_wi,
                                const cv::Vec3d& ray_cam,
                                const CameraExtrinsics& ext,
                                const GeoRaster& dem,
                                const Options& opt,
                                GeoPoint& hit) {
    const cv::Vec3d p0_ecef = wgs84ToEcef(gps.lon, gps.lat, 0.0);
    cv::Vec3d east;
    cv::Vec3d north;
    cv::Vec3d up;
    enuAxes(gps.lon, gps.lat, east, north, up);

    const cv::Vec3d gps_to_imu_imu(opt.gps_to_imu_x_m, opt.gps_to_imu_y_m, opt.gps_to_imu_z_m);
    const cv::Vec3d imu_origin_enu = cv::Vec3d(0.0, 0.0, gps.alt) + R_wi * gps_to_imu_imu;
    const cv::Vec3d cam_origin_enu = imu_origin_enu + cam_offset_imu;
    const cv::Vec3d ray_imu = ext.R_ic * ray_cam;
    const cv::Vec3d ray_enu = normalizeVec(R_wi * ray_imu);
    if (ray_enu[2] >= -1e-6) return false;

    auto heightDiff = [&](double distance_m, double& diff, GeoPoint& point) -> bool {
        const cv::Vec3d p_enu = cam_origin_enu + ray_enu * distance_m;
        const cv::Vec3d p_ecef = enuToEcef(p_enu, p0_ecef, east, north, up);
        point = ecefToGeodetic(p_ecef);
        double dem_alt = 0.0;
        if (!dem.sampleBand1BilinearLonLat(point.lon, point.lat, dem_alt)) return false;
        diff = point.alt - dem_alt;
        point.alt = dem_alt;
        return true;
    };

    double prev_d = 0.0;
    double prev_diff = 0.0;
    GeoPoint prev_point;
    if (!heightDiff(0.0, prev_diff, prev_point)) return false;

    for (double cur_d = opt.ray_step_m; cur_d <= opt.max_ray_distance_m; cur_d += opt.ray_step_m) {
        double cur_diff = 0.0;
        GeoPoint cur_point;
        if (!heightDiff(cur_d, cur_diff, cur_point)) continue;
        if (cur_diff <= 0.0) {
            double lo = prev_d;
            double hi = cur_d;
            GeoPoint best = cur_point;
            for (int iter = 0; iter < opt.binary_refine_iters; ++iter) {
                const double mid = 0.5 * (lo + hi);
                double mid_diff = 0.0;
                GeoPoint mid_point;
                if (!heightDiff(mid, mid_diff, mid_point)) {
                    lo = mid;
                    continue;
                }
                best = mid_point;
                if (mid_diff > 0.0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            hit = best;
            return true;
        }
        prev_d = cur_d;
        prev_diff = cur_diff;
        prev_point = cur_point;
        (void)prev_diff;
        (void)prev_point;
    }
    return false;
}

// 功能：实现 `worldRayFromCameraRay` 对应的功能。
static cv::Vec3d worldRayFromCameraRay(const cv::Vec3d& ray_cam,
                                       const cv::Matx33d& R_wi,
                                       const CameraExtrinsics& ext) {
    const cv::Vec3d ray_imu = ext.R_ic * ray_cam;
    return normalizeVec(R_wi * ray_imu);
}

// 功能：实现 `cropFromLonLatBox` 对应的功能。
static CropWindow cropFromLonLatBox(const GeoRaster& raster,
                                    const std::vector<GeoPoint>& box_corners) {
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const GeoPoint& p : box_corners) {
        double px = 0.0;
        double py = 0.0;
        if (!raster.lonLatToPixel(p.lon, p.lat, px, py)) {
            throw std::runtime_error("Failed to map lon/lat into raster pixels.");
        }
        min_x = std::min(min_x, px);
        min_y = std::min(min_y, py);
        max_x = std::max(max_x, px);
        max_y = std::max(max_y, py);
    }

    CropWindow win;
    win.x0 = static_cast<int>(std::floor(min_x));
    win.y0 = static_cast<int>(std::floor(min_y));
    win.x1 = static_cast<int>(std::ceil(max_x));
    win.y1 = static_cast<int>(std::ceil(max_y));
    return raster.clampWindow(win);
}

// 功能：实现 `makeCenteredCropBox` 对应的功能。
static std::vector<GeoPoint> makeCenteredCropBox(const GeoPoint& center,
                                                 const Options& opt,
                                                 const cv::Size& match_size,
                                                 double image_altitude_m) {
    const cv::Vec3d center_ecef = wgs84ToEcef(center.lon, center.lat, center.alt);
    cv::Vec3d east;
    cv::Vec3d north;
    cv::Vec3d up;
    enuAxes(center.lon, center.lat, east, north, up);

    const double crop_scale = std::max(1.0, opt.crop_scale_factor);
    double half_e = std::max(1.0, opt.crop_half_extent_e_m * crop_scale);
    double half_n = std::max(1.0, opt.crop_half_extent_n_m * crop_scale);

    double hfov_rad = std::numeric_limits<double>::quiet_NaN();
    double vfov_rad = std::numeric_limits<double>::quiet_NaN();
    if (computeImageFieldOfViewFromDiagonal(opt.crop_dfov_deg, match_size, hfov_rad, vfov_rad) &&
        std::isfinite(image_altitude_m) &&
        std::isfinite(opt.ground_altitude)) {
        const double relative_altitude_m = std::max(1.0, image_altitude_m - opt.ground_altitude);
        half_e = std::max(1.0, relative_altitude_m * std::tan(hfov_rad * 0.5) * crop_scale);
        half_n = std::max(1.0, relative_altitude_m * std::tan(vfov_rad * 0.5) * crop_scale);
    }

    std::vector<GeoPoint> corners;
    corners.reserve(4);
    const std::array<cv::Vec3d, 4> offsets{{
        cv::Vec3d(-half_e, -half_n, 0.0),
        cv::Vec3d(half_e, -half_n, 0.0),
        cv::Vec3d(half_e, half_n, 0.0),
        cv::Vec3d(-half_e, half_n, 0.0)
    }};
    for (const cv::Vec3d& offset : offsets) {
        const cv::Vec3d ecef = enuToEcef(offset, center_ecef, east, north, up);
        GeoPoint p = ecefToGeodetic(ecef);
        p.alt = center.alt;
        corners.push_back(p);
    }
    return corners;
}

// 功能：实现 `shiftGeoPointByEnu` 对应的功能。
static GeoPoint shiftGeoPointByEnu(const GeoPoint& origin,
                                   double delta_e_m,
                                   double delta_n_m,
                                   double delta_u_m = 0.0) {
    const cv::Vec3d origin_ecef = wgs84ToEcef(origin.lon, origin.lat, origin.alt);
    cv::Vec3d east;
    cv::Vec3d north;
    cv::Vec3d up;
    enuAxes(origin.lon, origin.lat, east, north, up);
    const cv::Vec3d shifted_ecef =
        enuToEcef(cv::Vec3d(delta_e_m, delta_n_m, delta_u_m), origin_ecef, east, north, up);
    GeoPoint out = ecefToGeodetic(shifted_ecef);
    out.alt = origin.alt + delta_u_m;
    return out;
}

// 功能：实现 `geoHorizontalDistanceMeters` 对应的功能。
static double geoHorizontalDistanceMeters(const GeoPoint& a,
                                         const GeoPoint& b) {
    if (!std::isfinite(a.lon) || !std::isfinite(a.lat) || !std::isfinite(a.alt) ||
        !std::isfinite(b.lon) || !std::isfinite(b.lat) || !std::isfinite(b.alt)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const cv::Vec3d a_ecef = wgs84ToEcef(a.lon, a.lat, a.alt);
    const cv::Vec3d b_ecef = wgs84ToEcef(b.lon, b.lat, b.alt);
    cv::Vec3d east;
    cv::Vec3d north;
    cv::Vec3d up;
    enuAxes(a.lon, a.lat, east, north, up);
    const cv::Vec3d enu = ecefToEnu(b_ecef, a_ecef, east, north, up);
    return std::hypot(enu[0], enu[1]);
}

// 功能：实现 `blendGeoPointByEnu` 对应的功能。
static GeoPoint blendGeoPointByEnu(const GeoPoint& from,
                                   const GeoPoint& to,
                                   double alpha) {
    if (!std::isfinite(alpha)) return from;
    const double clamped_alpha = std::clamp(alpha, 0.0, 1.0);
    if (clamped_alpha <= 0.0) return from;
    if (clamped_alpha >= 1.0) return to;
    if (!std::isfinite(from.lon) || !std::isfinite(from.lat) || !std::isfinite(from.alt) ||
        !std::isfinite(to.lon) || !std::isfinite(to.lat) || !std::isfinite(to.alt)) {
        return from;
    }

    const cv::Vec3d from_ecef = wgs84ToEcef(from.lon, from.lat, from.alt);
    const cv::Vec3d to_ecef = wgs84ToEcef(to.lon, to.lat, to.alt);
    cv::Vec3d east;
    cv::Vec3d north;
    cv::Vec3d up;
    enuAxes(from.lon, from.lat, east, north, up);
    const cv::Vec3d delta_enu = ecefToEnu(to_ecef, from_ecef, east, north, up);
    return shiftGeoPointByEnu(from,
                              clamped_alpha * delta_enu[0],
                              clamped_alpha * delta_enu[1],
                              clamped_alpha * delta_enu[2]);
}

// 功能：判断一个地理点是否有效。
static bool isFiniteGeoPoint(const GeoPoint& p) {
    return std::isfinite(p.lon) && std::isfinite(p.lat) && std::isfinite(p.alt);
}
