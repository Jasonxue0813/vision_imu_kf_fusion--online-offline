/**
 * @file online_ypr_crop_node_fusion.cpp
 * @brief `OnlineYprCropNode` 的视觉-IMU 融合实现。
 *
 * 这个文件负责：
 * 1. 建立融合参考系；
 * 2. 基于视觉结果生成锚点；
 * 3. 利用 IMU 进行短时传播；
 * 4. 组织连续融合与 fix 时刻融合结果输出。
 */

#include "splg_fusion/app/online_ypr_crop_node.h"

// 功能：仅用第一条可用的 fix 建立全局参考系，不用后续 fix 重建融合锚点。
void OnlineYprCropNode::bootstrapFusionAnchorFromFixLocked(const FixSample& fix) {
    if (!std::isfinite(fix.t) ||
        !std::isfinite(fix.lon) ||
        !std::isfinite(fix.lat) ||
        !std::isfinite(fix.alt)) {
        return;
    }

    const GeoPoint fix_geo{fix.lon, fix.lat, fix.alt};
    if (!ensureContinuousFusionReferenceLocked(fix_geo)) return;
}

// 功能：确保连续融合已经建立参考地理坐标系。
bool OnlineYprCropNode::ensureContinuousFusionReferenceLocked(const GeoPoint& geo) {
    if (continuous_filter_.reference_ready) return true;
    if (!isFiniteGeoPoint(geo)) return false;
    continuous_filter_.reference_ready = true;
    continuous_filter_.reference_geo = geo;
    continuous_filter_.reference_ecef = wgs84ToEcef(geo.lon, geo.lat, geo.alt);
    enuAxes(geo.lon, geo.lat,
            continuous_filter_.east,
            continuous_filter_.north,
            continuous_filter_.up);
    return true;
}

// 功能：把经纬高点转换到当前参考 ENU 坐标系。
bool OnlineYprCropNode::geoPointToReferenceEnuLocked(const GeoPoint& geo, cv::Vec3d& enu) const {
    if (!continuous_filter_.reference_ready || !isFiniteGeoPoint(geo)) return false;
    const cv::Vec3d ecef = wgs84ToEcef(geo.lon, geo.lat, geo.alt);
    enu = ecefToEnu(ecef,
                    continuous_filter_.reference_ecef,
                    continuous_filter_.east,
                    continuous_filter_.north,
                    continuous_filter_.up);
    return isFiniteVec(enu);
}

// 功能：把参考 ENU 坐标转换回经纬高。
GeoPoint OnlineYprCropNode::referenceEnuToGeoLocked(const cv::Vec3d& enu) const {
    const cv::Vec3d ecef = enuToEcef(enu,
                                     continuous_filter_.reference_ecef,
                                     continuous_filter_.east,
                                     continuous_filter_.north,
                                     continuous_filter_.up);
    return ecefToGeodetic(ecef);
}

void OnlineYprCropNode::appendContinuousFilterSnapshotLocked() {
    if (!continuous_filter_.initialized || !std::isfinite(continuous_filter_.state_t)) return;
    if (!isFiniteVec(continuous_filter_.state_enu) ||
        !isFiniteVec(continuous_filter_.state_vel_enu) ||
        !isFiniteMatx33d(continuous_filter_.cov_enu)) {
        return;
    }

    const OnlineYprCropNode::ContinuousFilterSnapshot snapshot{
        continuous_filter_.state_t,
        continuous_filter_.state_enu,
        continuous_filter_.state_vel_enu,
        continuous_filter_.cov_enu,
        continuous_filter_.last_heading_used_rad,
        continuous_filter_.last_speed_mps,
        continuous_filter_.anchor_heading_compass_rad,
        continuous_filter_.anchor_imu_heading_rel_rad,
        continuous_filter_.anchor_imu_valid,
        continuous_filter_.last_visual_update_t,
        continuous_filter_.visual_cov_scale,
    };

    if (!continuous_filter_history_.empty() &&
        snapshot.state_t + 1e-6 < continuous_filter_history_.back().state_t) {
        continuous_filter_history_.clear();
    }
    if (!continuous_filter_history_.empty() &&
        std::abs(snapshot.state_t - continuous_filter_history_.back().state_t) <= 1e-6) {
        continuous_filter_history_.back() = snapshot;
    } else {
        continuous_filter_history_.push_back(snapshot);
    }

    const double history_keep_sec =
        std::max(5.0, std::max(opt_.fusion_anchor_max_age_sec, opt_.fusion_imu_only_max_duration_sec) + 2.0);
    while (continuous_filter_history_.size() > 2 &&
           snapshot.state_t - continuous_filter_history_[1].state_t > history_keep_sec) {
        continuous_filter_history_.pop_front();
    }
}

bool OnlineYprCropNode::tryRestoreContinuousFilterSnapshotAtOrBeforeLocked(double target_t) {
    if (!std::isfinite(target_t) || continuous_filter_history_.empty()) return false;

    for (auto it = continuous_filter_history_.rbegin(); it != continuous_filter_history_.rend(); ++it) {
        if (!std::isfinite(it->state_t) || it->state_t > target_t + 1e-6) continue;
        continuous_filter_.initialized = true;
        continuous_filter_.state_t = it->state_t;
        continuous_filter_.state_enu = it->state_enu;
        continuous_filter_.state_vel_enu = it->state_vel_enu;
        continuous_filter_.cov_enu = it->cov_enu;
        continuous_filter_.last_heading_used_rad = it->last_heading_used_rad;
        continuous_filter_.last_speed_mps = it->last_speed_mps;
        continuous_filter_.anchor_heading_compass_rad = it->anchor_heading_compass_rad;
        continuous_filter_.anchor_imu_heading_rel_rad = it->anchor_imu_heading_rel_rad;
        continuous_filter_.anchor_imu_valid = it->anchor_imu_valid;
        continuous_filter_.last_visual_update_t = it->last_visual_update_t;
        continuous_filter_.visual_cov_scale = it->visual_cov_scale;
        return true;
    }
    return false;
}

// 功能：构建从当前状态到目标时刻的连续传播结果。
OnlineYprCropNode::ContinuousPropagation OnlineYprCropNode::buildContinuousPropagationLocked(double t_from,
                                                       double t_to,
                                                       double anchor_heading_compass_rad,
                                                       double anchor_imu_heading_rel_rad,
                                                       bool anchor_imu_valid,
                                                       const cv::Vec3d& initial_velocity_enu,
                                                       double fallback_heading_rad,
                                                       double fallback_speed_mps) const {
    OnlineYprCropNode::ContinuousPropagation prop;
    const double dt = t_to - t_from;
    if (!(dt > 1e-9)) {
        prop.valid = true;
        prop.speed_mps = cv::norm(initial_velocity_enu);
        if (!(prop.speed_mps > 0.0)) prop.speed_mps = std::max(0.0, fallback_speed_mps);
        if (std::isfinite(fallback_heading_rad)) prop.heading_used_rad = fallback_heading_rad;
        return prop;
    }

    double effective_t_from = t_from;
    if (imu_tracker_.initialized()) {
        const double imu_start_t = imu_tracker_.startTime();
        if (std::isfinite(imu_start_t)) {
            effective_t_from = std::max(effective_t_from, imu_start_t);
        }
    }
    if (!(t_to > effective_t_from + 1e-9)) {
        prop.valid = true;
        prop.speed_mps = cv::norm(initial_velocity_enu);
        if (!(prop.speed_mps > 0.0)) prop.speed_mps = std::max(0.0, fallback_speed_mps);
        if (std::isfinite(fallback_heading_rad)) {
            prop.heading_used_rad = fallback_heading_rad;
            prop.heading_source = "last_valid";
        }
        return prop;
    }

    double anchor_heading_rel = anchor_imu_heading_rel_rad;
    bool anchor_heading_rel_valid = anchor_imu_valid && std::isfinite(anchor_heading_rel);
    if (!anchor_heading_rel_valid) {
        anchor_heading_rel_valid =
            imu_tracker_.queryHeadingCompassRelativeRadCausal(effective_t_from, anchor_heading_rel);
    }

    OnlineImuRollPitchTracker::PreintegrationResult preint;
    if (!anchor_heading_rel_valid ||
        !imu_tracker_.preintegrateRelativeEnuCausal(
            effective_t_from, t_to, anchor_heading_compass_rad, anchor_heading_rel, initial_velocity_enu, preint)) {
        return prop;
    }

    prop.delta_enu = preint.delta_enu;
    prop.delta_vel_enu = preint.delta_vel_enu;
    prop.speed_mps = std::isfinite(preint.final_speed_mps)
        ? std::max(0.0, preint.final_speed_mps)
        : (std::isfinite(preint.mean_speed_mps) ? std::max(0.0, preint.mean_speed_mps) : 0.0);

    double heading_used_rad = std::numeric_limits<double>::quiet_NaN();
    std::string heading_source = "none";
    double imu_heading_now = 0.0;
    if (imu_tracker_.queryHeadingCompassRelativeRadCausal(t_to, imu_heading_now) &&
        std::isfinite(anchor_heading_compass_rad)) {
        const double delta = global_heading::wrapToPi(imu_heading_now - anchor_heading_rel);
        heading_used_rad = wrapToTwoPi(anchor_heading_compass_rad + delta);
        heading_source = "anchor+imu_delta";
    }
    if (!std::isfinite(heading_used_rad) && std::isfinite(fallback_heading_rad)) {
        heading_used_rad = fallback_heading_rad;
        heading_source = "last_valid";
    }
    if (!std::isfinite(heading_used_rad) && std::isfinite(anchor_heading_compass_rad)) {
        heading_used_rad = anchor_heading_compass_rad;
        heading_source = "anchor_only";
    }

    const double xy_rw_var =
        std::pow(std::max(0.0, opt_.kf_process_xy_rw_sigma_m_sqrt_s), 2.0) * dt;
    const double z_rw_var =
        std::pow(std::max(0.0, opt_.kf_process_z_rw_sigma_m_sqrt_s), 2.0) * dt;
    const double accel_sigma = std::max(0.0, opt_.imu_preintegration_accel_sigma_mps2);
    const double accel_var = 0.25 * accel_sigma * accel_sigma * dt * dt * dt * dt;
    prop.cov_enu = makeDiag3(xy_rw_var + accel_var,
                             xy_rw_var + accel_var,
                             z_rw_var + accel_var);

    prop.valid = true;
    prop.heading_used_rad = heading_used_rad;
    prop.heading_source = heading_source;
    return prop;
}

// 功能：构建一条 continuous fusion 输出记录。
bool OnlineYprCropNode::buildContinuousFusionRowLocked(const std::string& source,
                                    double stamp_sec,
                                    double heading_used_rad,
                                    double speed_mps,
                                    double propagate_dt_sec,
                                    int visual_update_applied,
                                    double visual_nis,
                                    int visual_gate_passed,
                                    const cv::Vec3d& state_enu,
                                    const cv::Matx33d& cov_enu,
                                    ContinuousFusionRow& row) {
    if (!continuous_filter_.reference_ready) return false;
    if (!isFiniteVec(state_enu) || !isFiniteMatx33d(cov_enu)) {
        return false;
    }

    const GeoPoint geo = referenceEnuToGeoLocked(state_enu);
    if (!isFiniteGeoPoint(geo)) return false;

    row.sample_index = ++continuous_filter_.sample_index;
    row.stamp_sec = stamp_sec;
    row.lon = geo.lon;
    row.lat = geo.lat;
    row.alt = geo.alt;
    row.east_m = state_enu[0];
    row.north_m = state_enu[1];
    row.up_m = state_enu[2];
    row.cov_xx_m2 = cov_enu(0, 0);
    row.cov_xy_m2 = cov_enu(0, 1);
    row.cov_xz_m2 = cov_enu(0, 2);
    row.cov_yx_m2 = cov_enu(1, 0);
    row.cov_yy_m2 = cov_enu(1, 1);
    row.cov_yz_m2 = cov_enu(1, 2);
    row.cov_zx_m2 = cov_enu(2, 0);
    row.cov_zy_m2 = cov_enu(2, 1);
    row.cov_zz_m2 = cov_enu(2, 2);
    row.heading_used_deg = std::isfinite(heading_used_rad)
        ? heading_used_rad * global_heading::RAD2DEG
        : std::numeric_limits<double>::quiet_NaN();
    row.speed_mps = speed_mps;
    row.propagate_dt_sec = propagate_dt_sec;
    row.visual_update_applied = visual_update_applied;
    row.visual_nis = visual_nis;
    row.visual_cov_scale = continuous_filter_.visual_cov_scale;
    row.visual_gate_passed = visual_gate_passed;
    row.source = source;
    appendFusionCropSeedHistoryLocked(stamp_sec, geo);
    return true;
}

// 功能：利用 IMU 和当前状态预测连续融合结果。
bool OnlineYprCropNode::predictContinuousFusionLocked(double target_t,
                                   const std::string& source,
                                   ContinuousFusionRow* out_row) {
    if (!continuous_filter_.initialized) return false;
    if (!(target_t >= continuous_filter_.state_t)) return false;
    const double dt = std::max(0.0, target_t - continuous_filter_.state_t);

    const OnlineYprCropNode::ContinuousPropagation prop = buildContinuousPropagationLocked(
        continuous_filter_.state_t,
        target_t,
        continuous_filter_.anchor_heading_compass_rad,
        continuous_filter_.anchor_imu_heading_rel_rad,
        continuous_filter_.anchor_imu_valid,
        continuous_filter_.state_vel_enu,
        continuous_filter_.last_heading_used_rad,
        continuous_filter_.last_speed_mps);
    if (!prop.valid) {
        const bool allow_gap_restart = (source == "imu_predict");
        const bool imu_gap_too_large = dt > opt_.imu_preintegration_max_gap_sec + 1e-9;
        if (!allow_gap_restart || !imu_gap_too_large) return false;

        const double accel_sigma = std::max(0.0, opt_.imu_preintegration_accel_sigma_mps2);
        const double prev_speed = std::isfinite(continuous_filter_.last_speed_mps)
            ? std::max(0.0, continuous_filter_.last_speed_mps)
            : cv::norm(continuous_filter_.state_vel_enu);
        const double xy_gap_sigma =
            std::max(5.0, prev_speed * dt + 0.5 * accel_sigma * dt * dt);
        const double z_gap_sigma =
            std::max(3.0,
                     std::abs(continuous_filter_.state_vel_enu[2]) * dt + 0.5 * accel_sigma * dt * dt);

        double current_imu_heading_rel_rad = std::numeric_limits<double>::quiet_NaN();
        const bool current_imu_heading_valid =
            imu_tracker_.queryHeadingCompassRelativeRadCausal(target_t, current_imu_heading_rel_rad);
        double restart_heading_rad = continuous_filter_.last_heading_used_rad;
        if (!std::isfinite(restart_heading_rad)) {
            restart_heading_rad = continuous_filter_.anchor_heading_compass_rad;
        }

        continuous_filter_.state_t = target_t;
        continuous_filter_.state_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        continuous_filter_.cov_enu += makeDiag3(xy_gap_sigma * xy_gap_sigma,
                                                xy_gap_sigma * xy_gap_sigma,
                                                z_gap_sigma * z_gap_sigma);
        continuous_filter_.anchor_heading_compass_rad = restart_heading_rad;
        continuous_filter_.anchor_imu_heading_rel_rad = current_imu_heading_rel_rad;
        continuous_filter_.anchor_imu_valid = current_imu_heading_valid;
        if (std::isfinite(restart_heading_rad)) {
            continuous_filter_.last_heading_used_rad = restart_heading_rad;
        }
        continuous_filter_.last_speed_mps = 0.0;
        appendContinuousFilterSnapshotLocked();

        if (out_row != nullptr) {
            return buildContinuousFusionRowLocked(
                "imu_gap_restart",
                target_t,
                restart_heading_rad,
                0.0,
                dt,
                0,
                std::numeric_limits<double>::quiet_NaN(),
                -1,
                continuous_filter_.state_enu,
                continuous_filter_.cov_enu,
                *out_row);
        }
        return true;
    }

    continuous_filter_.state_enu += prop.delta_enu;
    continuous_filter_.state_vel_enu += prop.delta_vel_enu;
    continuous_filter_.cov_enu += prop.cov_enu;
    continuous_filter_.state_t = target_t;
    if (std::isfinite(prop.heading_used_rad)) {
        continuous_filter_.last_heading_used_rad = prop.heading_used_rad;
    }
    continuous_filter_.last_speed_mps = prop.speed_mps;
    appendContinuousFilterSnapshotLocked();

    if (out_row != nullptr) {
        return buildContinuousFusionRowLocked(
            source,
            target_t,
            prop.heading_used_rad,
            prop.speed_mps,
            dt,
            0,
            std::numeric_limits<double>::quiet_NaN(),
            -1,
            continuous_filter_.state_enu,
            continuous_filter_.cov_enu,
            *out_row);
    }
    return true;
}

// 功能：计算三维视觉观测对应的 NIS 指标。
double OnlineYprCropNode::computeNis3d(const cv::Vec3d& innovation, const cv::Matx33d& covariance) const {
    if (!isFiniteVec(innovation) || !isFiniteMatx33d(covariance)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const cv::Matx33d inv = covariance.inv(cv::DECOMP_SVD);
    const cv::Vec3d weighted = inv * innovation;
    const double nis = innovation.dot(weighted);
    return std::isfinite(nis) && nis >= 0.0
        ? nis
        : std::numeric_limits<double>::quiet_NaN();
}

void OnlineYprCropNode::adaptVisualCovarianceScaleLocked(double nis) {
    if (!opt_.kf_visual_adaptive_enable || !std::isfinite(nis)) return;

    const double min_scale = std::max(1e-6, opt_.kf_visual_adaptive_min_scale);
    const double max_scale = std::max(min_scale, opt_.kf_visual_adaptive_max_scale);
    const double target = std::max(1e-6, opt_.kf_visual_adaptive_target_nis);
    const double alpha = std::clamp(opt_.kf_visual_adaptive_alpha, 0.0, 1.0);
    const double ratio = std::clamp(nis / target, 0.25, 4.0);
    const double next_scale =
        continuous_filter_.visual_cov_scale * std::exp(alpha * std::log(ratio));
    continuous_filter_.visual_cov_scale = std::clamp(next_scale, min_scale, max_scale);
}

// 功能：使用视觉观测更新连续融合状态。
bool OnlineYprCropNode::updateContinuousFusionFromVisualLocked(const FrameRecord& rec,
                                            ContinuousFusionRow& out_row) {
    if (!rec.localization_success) return false;

    OnlineYprCropNode::VisualAnchor visual_anchor_candidate;
    std::string visual_quality_reject_reason;
    bool visual_quality_gate_passed = makeVisualAnchorFromRecord(rec, visual_anchor_candidate);
    if (!visual_quality_gate_passed) {
        visual_quality_reject_reason = "visual_reject_invalid_pose";
    } else {
        visual_quality_gate_passed =
            passesVisualAnchorQualityGate(visual_anchor_candidate, visual_quality_reject_reason);
    }

    const GeoPoint visual_geo{rec.localization_lon, rec.localization_lat, rec.localization_alt};
    const GeoPoint ref_geo =
        (std::isfinite(rec.gps.lon) && std::isfinite(rec.gps.lat) && std::isfinite(rec.gps.alt))
            ? GeoPoint{rec.gps.lon, rec.gps.lat, rec.gps.alt}
            : visual_geo;
    if (!ensureContinuousFusionReferenceLocked(ref_geo)) {
        return false;
    }

    cv::Vec3d visual_enu(0.0, 0.0, 0.0);
    if (!geoPointToReferenceEnuLocked(visual_geo, visual_enu)) return false;

    double target_t = rec.image_t;
    bool replayed_from_history = false;
    double replay_target_t = std::numeric_limits<double>::quiet_NaN();
    double rec_imu_heading_rel_rad = std::numeric_limits<double>::quiet_NaN();
    const bool rec_imu_heading_valid =
        imu_tracker_.queryHeadingCompassRelativeRadCausal(rec.image_t, rec_imu_heading_rel_rad);
    const double rec_heading_compass_rad = std::isfinite(rec.yaw_compass_deg)
        ? wrapToTwoPi(rec.yaw_compass_deg * global_heading::DEG2RAD)
        : std::numeric_limits<double>::quiet_NaN();
    const bool was_initialized_before = continuous_filter_.initialized;

    cv::Vec3d z_enu = visual_enu;
    cv::Matx33d R_visual = computeVisualMeasurementCovariance(rec, opt_);
    if (!isFiniteMatx33d(R_visual)) return false;

    // 为保证严格的时间顺序，晚到视觉先恢复到观测时刻之前的历史状态，
    // 在视觉时刻完成更新，再把这段 IMU 重新正向回放到当前状态时刻。
    if (continuous_filter_.initialized && target_t + 1e-9 < continuous_filter_.state_t) {
        const double state_t_before_visual = continuous_filter_.state_t;
        const double delayed_visual_lag_sec = state_t_before_visual - target_t;
        const double max_delayed_visual_lag_sec = std::max(2.0, opt_.fusion_anchor_max_age_sec);
        if (delayed_visual_lag_sec > max_delayed_visual_lag_sec) {
            return false;
        }
        if (!tryRestoreContinuousFilterSnapshotAtOrBeforeLocked(target_t)) {
            return false;
        }
        if (continuous_filter_.state_t + 1e-9 < target_t &&
            !predictContinuousFusionLocked(target_t, "replay_to_visual_time", nullptr)) {
            return false;
        }
        replayed_from_history = true;
        replay_target_t = state_t_before_visual;
    }

    const double visual_propagate_dt_sec = continuous_filter_.initialized
        ? std::max(0.0, target_t - continuous_filter_.state_t)
        : 0.0;
    const double visual_gap_sec =
        (continuous_filter_.initialized && std::isfinite(continuous_filter_.last_visual_update_t))
            ? std::max(0.0, target_t - continuous_filter_.last_visual_update_t)
            : 0.0;
    const bool force_visual_reset_after_gap =
        continuous_filter_.initialized &&
        opt_.fusion_imu_only_max_duration_sec > 0.0 &&
        std::isfinite(continuous_filter_.last_visual_update_t) &&
        visual_gap_sec > opt_.fusion_imu_only_max_duration_sec;

    cv::Vec3d predicted_enu = visual_enu;
    cv::Vec3d predicted_vel_enu(0.0, 0.0, 0.0);
    cv::Matx33d predicted_cov = R_visual;
    OnlineYprCropNode::ContinuousPropagation pred_prop;

    if (continuous_filter_.initialized) {
        predicted_enu = continuous_filter_.state_enu;
        predicted_vel_enu = continuous_filter_.state_vel_enu;
        predicted_cov = continuous_filter_.cov_enu;
        if (target_t > continuous_filter_.state_t + 1e-9) {
            pred_prop = buildContinuousPropagationLocked(
                continuous_filter_.state_t,
                target_t,
                continuous_filter_.anchor_heading_compass_rad,
                continuous_filter_.anchor_imu_heading_rel_rad,
                continuous_filter_.anchor_imu_valid,
                continuous_filter_.state_vel_enu,
                continuous_filter_.last_heading_used_rad,
                continuous_filter_.last_speed_mps);
            if (!pred_prop.valid) return false;
            predicted_enu += pred_prop.delta_enu;
            predicted_vel_enu += pred_prop.delta_vel_enu;
            predicted_cov += pred_prop.cov_enu;
        } else {
            pred_prop.valid = true;
            pred_prop.heading_used_rad = continuous_filter_.last_heading_used_rad;
            pred_prop.speed_mps = cv::norm(predicted_vel_enu);
        }
    } else {
        pred_prop.valid = true;
        pred_prop.heading_used_rad = rec_heading_compass_rad;
        pred_prop.speed_mps = 0.0;
    }

    const cv::Matx33d R = scaleMatx33d(R_visual, continuous_filter_.visual_cov_scale);
    if (!isFiniteMatx33d(R)) return false;

    double visual_nis = std::numeric_limits<double>::quiet_NaN();
    int visual_gate_passed = -1;
    bool update_applied = visual_quality_gate_passed;
    cv::Vec3d row_state_enu = predicted_enu;
    cv::Matx33d row_cov_enu = predicted_cov;

    if (!visual_quality_gate_passed) {
        if (!continuous_filter_.initialized) {
            return false;
        }
        continuous_filter_.visual_reject_count += 1;
        continuous_filter_.state_enu = predicted_enu;
        continuous_filter_.state_vel_enu = predicted_vel_enu;
        continuous_filter_.cov_enu = predicted_cov;
        continuous_filter_.state_t = target_t;
        row_state_enu = predicted_enu;
        row_cov_enu = predicted_cov;
    } else if (!continuous_filter_.initialized || force_visual_reset_after_gap) {
        continuous_filter_.initialized = true;
        continuous_filter_.state_t = target_t;
        continuous_filter_.state_enu = z_enu;
        continuous_filter_.state_vel_enu = predicted_vel_enu;
        continuous_filter_.cov_enu = R;
        continuous_filter_.last_visual_update_t = target_t;
        continuous_filter_.visual_update_count += 1;
        row_state_enu = continuous_filter_.state_enu;
        row_cov_enu = continuous_filter_.cov_enu;
    } else {
        const cv::Matx33d S = predicted_cov + R;
        const cv::Vec3d innovation = z_enu - predicted_enu;
        visual_nis = computeNis3d(innovation, S);
        continuous_filter_.last_visual_nis = visual_nis;

        const double gate = std::max(0.0, opt_.kf_visual_nis_gate);
        visual_gate_passed =
            (!std::isfinite(visual_nis) || gate <= 0.0 || visual_nis <= gate) ? 1 : 0;
        adaptVisualCovarianceScaleLocked(visual_nis);

        if (visual_gate_passed == 0 && opt_.kf_visual_reject_on_gate) {
            update_applied = false;
            continuous_filter_.visual_reject_count += 1;
            continuous_filter_.state_enu = predicted_enu;
            continuous_filter_.state_vel_enu = predicted_vel_enu;
            continuous_filter_.cov_enu = predicted_cov;
            continuous_filter_.state_t = target_t;
            row_state_enu = predicted_enu;
            row_cov_enu = predicted_cov;
        } else {
            const cv::Matx33d S_inv = S.inv(cv::DECOMP_SVD);
            const cv::Matx33d K = predicted_cov * S_inv;
            continuous_filter_.state_enu = predicted_enu + K * innovation;
            continuous_filter_.state_vel_enu = predicted_vel_enu;
            const cv::Matx33d I = cv::Matx33d::eye();
            continuous_filter_.cov_enu =
                (I - K) * predicted_cov * (I - K).t() + K * R * K.t();
            continuous_filter_.state_t = target_t;
            continuous_filter_.last_visual_update_t = target_t;
            continuous_filter_.visual_update_count += 1;
            row_state_enu = continuous_filter_.state_enu;
            row_cov_enu = continuous_filter_.cov_enu;
        }
    }

    if (update_applied) {
        double anchor_heading_compass_rad = rec_heading_compass_rad;
        double anchor_imu_heading_rel_rad = rec_imu_heading_rel_rad;
        bool anchor_imu_valid = rec_imu_heading_valid;

        continuous_filter_.anchor_heading_compass_rad = anchor_heading_compass_rad;
        continuous_filter_.anchor_imu_heading_rel_rad = anchor_imu_heading_rel_rad;
        continuous_filter_.anchor_imu_valid = anchor_imu_valid;
    }
    if (std::isfinite(pred_prop.heading_used_rad)) {
        continuous_filter_.last_heading_used_rad = pred_prop.heading_used_rad;
    }
    continuous_filter_.last_speed_mps = pred_prop.speed_mps;
    appendContinuousFilterSnapshotLocked();

    double row_heading_used_rad = pred_prop.heading_used_rad;
    double row_speed_mps = pred_prop.speed_mps;
    double row_propagate_dt_sec = visual_propagate_dt_sec;
    if (replayed_from_history && replay_target_t > continuous_filter_.state_t + 1e-9) {
        if (!predictContinuousFusionLocked(replay_target_t, "replay_after_visual_update", nullptr)) {
            return false;
        }
        target_t = replay_target_t;
        row_state_enu = continuous_filter_.state_enu;
        row_cov_enu = continuous_filter_.cov_enu;
        row_heading_used_rad = continuous_filter_.last_heading_used_rad;
        row_speed_mps = continuous_filter_.last_speed_mps;
        row_propagate_dt_sec = std::max(0.0, replay_target_t - rec.image_t);
    }

    return buildContinuousFusionRowLocked(
        update_applied
            ? ((!was_initialized_before || continuous_filter_.sample_index == 0)
                   ? "visual_init"
                   : (force_visual_reset_after_gap
                          ? (replayed_from_history ? "visual_reinit_after_gap_replayed"
                                                   : "visual_reinit_after_gap")
                          : (replayed_from_history ? "visual_update_replayed"
                                                   : "visual_update")))
            : (!visual_quality_gate_passed
                   ? (replayed_from_history
                          ? (visual_quality_reject_reason + "_replayed")
                          : visual_quality_reject_reason)
                   : (replayed_from_history ? "visual_reject_nis_replayed"
                                            : "visual_reject_nis")),
        target_t,
        row_heading_used_rad,
        row_speed_mps,
        row_propagate_dt_sec,
        update_applied ? 1 : 0,
        visual_nis,
        visual_gate_passed,
        row_state_enu,
        row_cov_enu,
        out_row);
}

// 功能：线程安全地追加一条 continuous fusion CSV 记录。
void OnlineYprCropNode::appendContinuousFusionCsvRowThreadSafe(const ContinuousFusionRow& row) {
    constexpr double kContinuousCsvIntervalSec = 0.05;
    ContinuousFusionRow row_with_timing = row;
    const double output_wall_sec = currentSystemEpochSec();
    row_with_timing.output_wall_epoch_sec = output_wall_sec;
    if (std::isfinite(row_with_timing.stamp_sec)) {
        row_with_timing.stamp_to_output_latency_ms =
            (output_wall_sec - row_with_timing.stamp_sec) * 1000.0;
    }

    std::lock_guard<std::mutex> lock(continuous_fusion_csv_mutex_);
    if (std::isfinite(row_with_timing.stamp_sec)) {
        if (std::isfinite(continuous_last_csv_write_stamp_sec_)) {
            const double dt_since_last_write =
                row_with_timing.stamp_sec - continuous_last_csv_write_stamp_sec_;
            if (dt_since_last_write + 1e-9 < kContinuousCsvIntervalSec) {
                return;
            }
        }
        if (!continuous_latency_reference_ready_) {
            continuous_latency_reference_ready_ = true;
            continuous_latency_reference_stamp_sec_ = row_with_timing.stamp_sec;
            continuous_latency_reference_wall_sec_ = output_wall_sec;
        }
        row_with_timing.replay_relative_latency_ms =
            ((output_wall_sec - continuous_latency_reference_wall_sec_) -
             (row_with_timing.stamp_sec - continuous_latency_reference_stamp_sec_)) * 1000.0;
        continuous_last_csv_write_stamp_sec_ = row_with_timing.stamp_sec;
    }
    appendContinuousFusionCsvRow(opt_.high_rate_fusion_csv, row_with_timing);
}

void OnlineYprCropNode::writeMaybeCsvValue(std::ostream& os, double value) {
    if (std::isfinite(value)) os << value;
}

// 功能：向 JSON 文本写一个数值或 `null`。
void OnlineYprCropNode::writeJsonDoubleOrNull(std::ostream& os, double value) {
    if (std::isfinite(value)) os << value;
    else os << "null";
}

// 功能：把 fix 融合结果行序列化成 JSON 字符串。
std::string OnlineYprCropNode::fixFusionRowToJson(const OnlineYprCropNode::FixFusionRow& row) const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(9);
    os << '{';
    os << "\"fix_index\":" << row.fix_index;
    os << ",\"fix_stamp_sec\":"; writeJsonDoubleOrNull(os, row.fix_stamp_sec);
    os << ",\"gt_lon\":"; writeJsonDoubleOrNull(os, row.gt_lon);
    os << ",\"gt_lat\":"; writeJsonDoubleOrNull(os, row.gt_lat);
    os << ",\"gt_alt\":"; writeJsonDoubleOrNull(os, row.gt_alt);
    os << ",\"pred_available\":" << (row.pred_available ? "true" : "false");
    os << ",\"pred_lon\":"; writeJsonDoubleOrNull(os, row.pred_lon);
    os << ",\"pred_lat\":"; writeJsonDoubleOrNull(os, row.pred_lat);
    os << ",\"pred_alt\":"; writeJsonDoubleOrNull(os, row.pred_alt);
    os << ",\"visual_anchor_image_index\":" << row.visual_anchor_image_index;
    os << ",\"visual_anchor_stamp_sec\":"; writeJsonDoubleOrNull(os, row.visual_anchor_stamp_sec);
    os << ",\"visual_anchor_lon\":"; writeJsonDoubleOrNull(os, row.visual_anchor_lon);
    os << ",\"visual_anchor_lat\":"; writeJsonDoubleOrNull(os, row.visual_anchor_lat);
    os << ",\"visual_anchor_alt\":"; writeJsonDoubleOrNull(os, row.visual_anchor_alt);
    os << ",\"visual_anchor_heading_deg\":"; writeJsonDoubleOrNull(os, row.visual_anchor_heading_deg);
    os << ",\"imu_heading_delta_deg\":"; writeJsonDoubleOrNull(os, row.imu_heading_delta_deg);
    os << ",\"heading_used_deg\":"; writeJsonDoubleOrNull(os, row.heading_used_deg);
    os << ",\"speed_mps\":"; writeJsonDoubleOrNull(os, row.speed_mps);
    os << ",\"propagate_dt_sec\":"; writeJsonDoubleOrNull(os, row.propagate_dt_sec);
    os << ",\"visual_anchor_age_sec\":"; writeJsonDoubleOrNull(os, row.visual_anchor_age_sec);
    os << ",\"visual_anchor_sync_wait_ms\":"; writeJsonDoubleOrNull(os, row.visual_anchor_sync_wait_ms);
    os << ",\"visual_anchor_processing_ms\":"; writeJsonDoubleOrNull(os, row.visual_anchor_processing_ms);
    os << ",\"visual_anchor_topic_to_result_ms\":"; writeJsonDoubleOrNull(os, row.visual_anchor_topic_to_result_ms);
    os << ",\"estimated_staleness_ms\":"; writeJsonDoubleOrNull(os, row.estimated_staleness_ms);
    os << ",\"fix_compute_latency_ms\":"; writeJsonDoubleOrNull(os, row.fix_compute_latency_ms);
    os << ",\"fix_topic_to_fusion_result_ms\":"; writeJsonDoubleOrNull(os, row.fix_topic_to_fusion_result_ms);
    os << ",\"anchor_reject_count\":" << row.anchor_reject_count;
    os << ",\"anchor_reject_reason\":" << jsonEscape(row.anchor_reject_reason);
    os << ",\"heading_source\":" << jsonEscape(row.heading_source);
    os << ",\"failure_reason\":" << jsonEscape(row.failure_reason);
    os << '}';
    return os.str();
}

// 功能：把一条 fix 融合结果写成 CSV 行。
void OnlineYprCropNode::writeFixFusionCsvRow(std::ostream& os, const OnlineYprCropNode::FixFusionRow& row) const {
    os << row.fix_index << ',';
    writeMaybeCsvValue(os, row.fix_stamp_sec);
    os << ',';
    writeMaybeCsvValue(os, row.gt_lon);
    os << ',';
    writeMaybeCsvValue(os, row.gt_lat);
    os << ',';
    writeMaybeCsvValue(os, row.gt_alt);
    os << ',' << row.pred_available << ',';
    writeMaybeCsvValue(os, row.pred_lon);
    os << ',';
    writeMaybeCsvValue(os, row.pred_lat);
    os << ',';
    writeMaybeCsvValue(os, row.pred_alt);
    os << ',' << row.visual_anchor_image_index << ',';
    writeMaybeCsvValue(os, row.visual_anchor_stamp_sec);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_lon);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_lat);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_alt);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_heading_deg);
    os << ',';
    writeMaybeCsvValue(os, row.imu_heading_delta_deg);
    os << ',';
    writeMaybeCsvValue(os, row.heading_used_deg);
    os << ',';
    writeMaybeCsvValue(os, row.speed_mps);
    os << ',';
    writeMaybeCsvValue(os, row.propagate_dt_sec);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_age_sec);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_sync_wait_ms);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_processing_ms);
    os << ',';
    writeMaybeCsvValue(os, row.visual_anchor_topic_to_result_ms);
    os << ',';
    writeMaybeCsvValue(os, row.estimated_staleness_ms);
    os << ',';
    writeMaybeCsvValue(os, row.fix_compute_latency_ms);
    os << ',';
    writeMaybeCsvValue(os, row.fix_topic_to_fusion_result_ms);
      os << ',' << row.anchor_reject_count << ',' << csvEscape(row.anchor_reject_reason)
          << ',' << csvEscape(row.heading_source) << ',' << csvEscape(row.failure_reason) << '\n';
}

// 功能：把一条 GNSS 参考结果写成 CSV 行。
void OnlineYprCropNode::writeGnssPositionCsvRow(std::ostream& os, const OnlineYprCropNode::FixFusionRow& row) const {
    os << row.fix_index << ',';
    writeMaybeCsvValue(os, row.fix_stamp_sec);
    os << ',';
    writeMaybeCsvValue(os, row.gt_lon);
    os << ',';
    writeMaybeCsvValue(os, row.gt_lat);
    os << ',';
    writeMaybeCsvValue(os, row.gt_alt);
    os << '\n';
}

void OnlineYprCropNode::writeImuPositionCsvRow(std::ostream& os, const OnlineYprCropNode::FixFusionRow& row) const {
    os << row.fix_index << ',';
    writeMaybeCsvValue(os, row.fix_stamp_sec);
    os << ',' << row.imu_available << ',';
    writeMaybeCsvValue(os, row.imu_lon);
    os << ',';
    writeMaybeCsvValue(os, row.imu_lat);
    os << ',';
    writeMaybeCsvValue(os, row.imu_alt);
    os << ',' << row.visual_anchor_image_index << ',';
    writeMaybeCsvValue(os, row.visual_anchor_stamp_sec);
    os << '\n';
}

// 功能：追加一条 fix 融合结果到输出文件。
void OnlineYprCropNode::appendFixFusionCsvRow(const OnlineYprCropNode::FixFusionRow& row) {
    std::lock_guard<std::mutex> lock(fix_fusion_csv_mutex_);
    ensureCsvHeaderPresent(opt_.fix_fusion_csv, fixFusionCsvHeaderLine());
    ensureCsvHeaderPresent(opt_.gnss_position_csv, gnssPositionCsvHeaderLine());
    std::ofstream ofs(opt_.fix_fusion_csv, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to append fix fusion csv: " + opt_.fix_fusion_csv);
    }
    std::ofstream imu_ofs(opt_.imu_position_csv, std::ios::app);
    if (!imu_ofs.is_open()) {
        throw std::runtime_error("Failed to append imu position csv: " + opt_.imu_position_csv);
    }
    std::ofstream gnss_ofs(opt_.gnss_position_csv, std::ios::app);
    if (!gnss_ofs.is_open()) {
        throw std::runtime_error("Failed to append gnss position csv: " + opt_.gnss_position_csv);
    }
    ofs << std::fixed << std::setprecision(9);
    imu_ofs << std::fixed << std::setprecision(9);
    gnss_ofs << std::fixed << std::setprecision(9);
    writeFixFusionCsvRow(ofs, row);
    writeImuPositionCsvRow(imu_ofs, row);
    writeGnssPositionCsvRow(gnss_ofs, row);
}

void OnlineYprCropNode::publishFixFusionRow(const OnlineYprCropNode::FixFusionRow& row) {
    if (!fix_fusion_pub_) return;
    std_msgs::String msg;
    msg.data = fixFusionRowToJson(row);
    fix_fusion_pub_.publish(msg);
}

bool OnlineYprCropNode::makeVisualAnchorFromRecord(const FrameRecord& rec, OnlineYprCropNode::VisualAnchor& anchor) {
    if (!rec.localization_success) return false;
    if (!std::isfinite(rec.image_t)) {
        return false;
    }

    const bool has_fused_pose =
        std::isfinite(rec.fused_lon) &&
        std::isfinite(rec.fused_lat) &&
        std::isfinite(rec.fused_alt);
    const bool has_visual_pose =
        std::isfinite(rec.localization_lon) &&
        std::isfinite(rec.localization_lat) &&
        std::isfinite(rec.localization_alt);
    if (!has_fused_pose && !has_visual_pose) return false;

    anchor.image_index = rec.image_index;
    anchor.t = rec.image_t;
    // 锚点位置优先使用当前帧原始视觉定位，避免把上一轮融合传播误差回灌到新锚点里。
    if (has_visual_pose) {
        anchor.pos.lon = rec.localization_lon;
        anchor.pos.lat = rec.localization_lat;
        anchor.pos.alt = rec.localization_alt;
        if (has_fused_pose) {
            anchor.vel_enu = cv::Vec3d(rec.fused_vel_east_mps,
                                       rec.fused_vel_north_mps,
                                       rec.fused_vel_up_mps);
            if (!isFiniteVec(anchor.vel_enu)) {
                anchor.vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
            }
        } else {
            anchor.vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        }
    } else {
        anchor.pos.lon = rec.fused_lon;
        anchor.pos.lat = rec.fused_lat;
        anchor.pos.alt = rec.fused_alt;
        anchor.vel_enu = cv::Vec3d(rec.fused_vel_east_mps,
                                   rec.fused_vel_north_mps,
                                   rec.fused_vel_up_mps);
        if (!isFiniteVec(anchor.vel_enu)) {
            anchor.vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        }
    }
    anchor.used_points = rec.localization_used_points;
    anchor.inlier_points = rec.localization_inlier_points;
    if (rec.localization_used_points > 0) {
        anchor.inlier_ratio = static_cast<double>(rec.localization_inlier_points) /
                              static_cast<double>(rec.localization_used_points);
    }
    anchor.reproj_error = rec.localization_reproj_error;
    anchor.sync_wait_ms = rec.sync_wait_ms;
    anchor.processing_ms = rec.processing_ms;
    anchor.topic_to_result_ms = rec.topic_to_result_ms;
    if (std::isfinite(rec.yaw_compass_deg)) {
        anchor.heading_compass_rad = wrapToTwoPi(rec.yaw_compass_deg * global_heading::DEG2RAD);
    } else {
        global_heading::HeadingSample heading;
        if (heading_est_.query(rec.image_t, heading)) {
            anchor.heading_compass_rad = wrapToTwoPi(heading.psi_compass);
        }
    }

    double imu_heading_rel = 0.0;
    if (imu_tracker_.queryHeadingCompassRelativeRadCausal(rec.image_t, imu_heading_rel)) {
        anchor.imu_heading_rel_rad = imu_heading_rel;
        anchor.imu_heading_valid = true;
    }
    return true;
}

// 功能：检查视觉记录是否满足成为锚点的质量门限。
bool OnlineYprCropNode::passesVisualAnchorQualityGate(const OnlineYprCropNode::VisualAnchor& anchor,
                                   std::string& reject_reason) const {
    if (anchor.used_points < opt_.fusion_anchor_min_used_points) {
        reject_reason = "anchor_rejected_low_used_points";
        return false;
    }
    if (!std::isfinite(anchor.inlier_ratio) ||
        anchor.inlier_ratio < opt_.fusion_anchor_min_inlier_ratio) {
        reject_reason = "anchor_rejected_low_inlier_ratio";
        return false;
    }
    if (std::isfinite(anchor.reproj_error) &&
        anchor.reproj_error > opt_.fusion_anchor_max_reproj_error) {
        reject_reason = "anchor_rejected_high_reproj_error";
        return false;
    }
    return true;
}

// 功能：计算锚点创新门限对应的距离阈值。
double OnlineYprCropNode::computeAnchorInnovationGateMeters(double speed_mps,
                                         double anchor_dt_sec) const {
    const double clamped_speed = std::max(0.0, speed_mps);
    const double clamped_dt = std::max(0.0, anchor_dt_sec);
    return opt_.fusion_anchor_innovation_base_m +
           opt_.fusion_anchor_innovation_speed_gain * clamped_speed * clamped_dt;
}

// 功能：计算新旧锚点软更新时的融合系数。
double OnlineYprCropNode::computeAnchorBlendAlpha(const OnlineYprCropNode::VisualAnchor& anchor) const {
    const double min_alpha = std::clamp(opt_.fusion_anchor_soft_update_alpha_min, 0.0, 1.0);
    const double max_alpha = std::clamp(opt_.fusion_anchor_soft_update_alpha_max, min_alpha, 1.0);

    double used_score = 0.0;
    if (opt_.fusion_anchor_min_used_points > 0) {
        const double excess = static_cast<double>(anchor.used_points - opt_.fusion_anchor_min_used_points);
        const double denom = static_cast<double>(opt_.fusion_anchor_min_used_points);
        used_score = std::clamp(excess / denom, 0.0, 1.0);
    }

    double ratio_score = 0.0;
    if (std::isfinite(anchor.inlier_ratio)) {
        const double denom = std::max(1e-6, 1.0 - opt_.fusion_anchor_min_inlier_ratio);
        ratio_score = std::clamp(
            (anchor.inlier_ratio - opt_.fusion_anchor_min_inlier_ratio) / denom,
            0.0,
            1.0);
    }

    double reproj_score = 0.5;
    if (std::isfinite(anchor.reproj_error) && opt_.fusion_anchor_max_reproj_error > 1e-6) {
        reproj_score = std::clamp(
            (opt_.fusion_anchor_max_reproj_error - anchor.reproj_error) /
                opt_.fusion_anchor_max_reproj_error,
            0.0,
            1.0);
    }

    const double quality = (used_score + ratio_score + reproj_score) / 3.0;
    return min_alpha + (max_alpha - min_alpha) * quality;
}

// 功能：计算视觉锚点相对当前时刻的年龄。
double OnlineYprCropNode::computeAnchorAgeSec(const OnlineYprCropNode::VisualAnchor& anchor, double ref_t) const {
    if (!std::isfinite(anchor.t) || !std::isfinite(ref_t)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::max(0.0, ref_t - anchor.t);
}

// 功能：判断某个视觉锚点在指定时刻是否已经过期。
bool OnlineYprCropNode::isAnchorExpiredAtTime(const OnlineYprCropNode::VisualAnchor& anchor, double ref_t) const {
    if (!(opt_.fusion_anchor_max_age_sec > 0.0)) return false;
    const double age_sec = computeAnchorAgeSec(anchor, ref_t);
    return std::isfinite(age_sec) && age_sec > opt_.fusion_anchor_max_age_sec;
}

// 功能：重置当前活动视觉锚点及其相关状态。
void OnlineYprCropNode::resetFusionAnchorState(bool& has_active_anchor,
                            OnlineYprCropNode::VisualAnchor& active_anchor,
                            GeoPoint& fused_pos,
                            cv::Vec3d& fused_vel_enu,
                            double& last_prop_t,
                            double& last_heading_used_rad) const {
    has_active_anchor = false;
    active_anchor = OnlineYprCropNode::VisualAnchor{};
    fused_pos = GeoPoint{};
    fused_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
    last_prop_t = std::numeric_limits<double>::quiet_NaN();
    last_heading_used_rad = std::numeric_limits<double>::quiet_NaN();
}

// 功能：在锚点过老时主动使其失效。
bool OnlineYprCropNode::expireFusionAnchorIfNeeded(bool& has_active_anchor,
                                OnlineYprCropNode::VisualAnchor& active_anchor,
                                GeoPoint& fused_pos,
                                cv::Vec3d& fused_vel_enu,
                                double& last_prop_t,
                                double& last_heading_used_rad,
                                double ref_t,
                                std::string& reason_out) const {
    if (!has_active_anchor || !isAnchorExpiredAtTime(active_anchor, ref_t)) {
        return false;
    }
    resetFusionAnchorState(has_active_anchor,
                           active_anchor,
                           fused_pos,
                           fused_vel_enu,
                           last_prop_t,
                           last_heading_used_rad);
    reason_out = "visual_anchor_expired";
    return true;
}

// 功能：尝试在 fix 时刻接纳一个新的视觉锚点。
bool OnlineYprCropNode::tryAdoptAnchorAtFixTimeLocked(const OnlineYprCropNode::VisualAnchor& candidate,
                                   double fix_t,
                                   double speed_hint_mps,
                                   std::string& reject_reason) {
    if (isAnchorExpiredAtTime(candidate, fix_t)) {
        reject_reason = "anchor_rejected_stale";
        return false;
    }
    if (!passesVisualAnchorQualityGate(candidate, reject_reason)) {
        return false;
    }

    if (fusion_has_active_anchor_) {
        const double innovation_m = geoHorizontalDistanceMeters(fusion_active_anchor_.pos, candidate.pos);
        const double anchor_dt_sec = std::max(0.0, candidate.t - fusion_active_anchor_.t);
        const double innovation_gate_m =
            computeAnchorInnovationGateMeters(speed_hint_mps, anchor_dt_sec);
        if (std::isfinite(innovation_m) && innovation_m > innovation_gate_m) {
            reject_reason = "anchor_rejected_large_innovation";
            return false;
        }
    }

    fusion_fused_pos_ = candidate.pos;
    fusion_fused_vel_enu_ = candidate.vel_enu;
    fusion_active_anchor_ = candidate;
    fusion_has_active_anchor_ = true;
    fusion_last_prop_t_ = candidate.t;
    fusion_last_heading_used_rad_ = candidate.heading_compass_rad;
    (void)fix_t;
    return true;
}

// 功能：从视觉锚点出发，通过 IMU 传播预测目标时刻地理位置。
bool OnlineYprCropNode::predictGeoFromAnchorLocked(const OnlineYprCropNode::VisualAnchor& anchor,
                                double start_t,
                                const GeoPoint& start_geo,
                                const cv::Vec3d& start_vel_enu,
                                double target_t,
                                double fallback_heading_rad,
                                GeoPoint& pred_geo,
                                cv::Vec3d& pred_vel_enu,
                                OnlineYprCropNode::ContinuousPropagation& prop,
                                double& imu_heading_delta_deg) const {
    imu_heading_delta_deg = std::numeric_limits<double>::quiet_NaN();
    if (!isFiniteGeoPoint(anchor.pos) ||
        !isFiniteGeoPoint(start_geo) ||
        !std::isfinite(anchor.t) ||
        !std::isfinite(start_t) ||
        target_t + 1e-9 < start_t) {
        return false;
    }

    prop = buildContinuousPropagationLocked(
        start_t,
        target_t,
        anchor.heading_compass_rad,
        anchor.imu_heading_rel_rad,
        anchor.imu_heading_valid,
        start_vel_enu,
        fallback_heading_rad,
        cv::norm(start_vel_enu));
    if (!prop.valid) return false;

    if (anchor.imu_heading_valid) {
        double imu_heading_now = 0.0;
        if (imu_tracker_.queryHeadingCompassRelativeRadCausal(target_t, imu_heading_now)) {
            const double delta = global_heading::wrapToPi(imu_heading_now - anchor.imu_heading_rel_rad);
            imu_heading_delta_deg = delta * global_heading::RAD2DEG;
        }
    }

    pred_geo = shiftGeoPointByEnu(start_geo,
                                  prop.delta_enu[0],
                                  prop.delta_enu[1],
                                  prop.delta_enu[2]);
    pred_vel_enu = start_vel_enu + prop.delta_vel_enu;
    return isFiniteGeoPoint(pred_geo);
}

// 功能：同步在线视觉锚点队列直到给定 fix 时刻。
void OnlineYprCropNode::syncOnlineAnchorsUntilLocked(double fix_t) {
    fusion_reject_count_this_fix_ = 0;
    fusion_last_anchor_reject_reason_.clear();

    std::vector<FrameRecord> newly_finished_records;
    {
        std::lock_guard<std::mutex> records_lock(records_mutex_);
        if (fusion_record_scan_index_ < records_.size()) {
            newly_finished_records.assign(records_.begin() + static_cast<std::ptrdiff_t>(fusion_record_scan_index_),
                                          records_.end());
            fusion_record_scan_index_ = records_.size();
        }
    }

    expireFusionAnchorIfNeeded(fusion_has_active_anchor_,
                               fusion_active_anchor_,
                               fusion_fused_pos_,
                               fusion_fused_vel_enu_,
                               fusion_last_prop_t_,
                               fusion_last_heading_used_rad_,
                               fix_t,
                               fusion_last_anchor_reject_reason_);

    for (const FrameRecord& rec : newly_finished_records) {
        OnlineYprCropNode::VisualAnchor anchor;
        if (makeVisualAnchorFromRecord(rec, anchor)) {
            fusion_pending_anchors_.push_back(anchor);
        }
    }

    double speed_hint_mps = 0.0;

    while (!fusion_pending_anchors_.empty() && fusion_pending_anchors_.front().t <= fix_t) {
        const OnlineYprCropNode::VisualAnchor candidate_anchor = fusion_pending_anchors_.front();
        fusion_pending_anchors_.pop_front();
        std::string reject_reason;
        if (!tryAdoptAnchorAtFixTimeLocked(candidate_anchor, fix_t, speed_hint_mps, reject_reason)) {
            fusion_last_anchor_reject_reason_ = reject_reason;
            ++fusion_reject_count_this_fix_;
            continue;
        }
    }

    expireFusionAnchorIfNeeded(fusion_has_active_anchor_,
                               fusion_active_anchor_,
                               fusion_fused_pos_,
                               fusion_fused_vel_enu_,
                               fusion_last_prop_t_,
                               fusion_last_heading_used_rad_,
                               fix_t,
                               fusion_last_anchor_reject_reason_);
}

// 功能：构建在线模式下一条 fix 融合结果。
void OnlineYprCropNode::buildOnlineFixFusionRowLocked(const FixSample& fix, size_t fix_index, OnlineYprCropNode::FixFusionRow& row) {
    row.fix_index = fix_index;
    row.fix_stamp_sec = fix.t;
    row.gt_lon = fix.lon;
    row.gt_lat = fix.lat;
    row.gt_alt = fix.alt;
    row.heading_source = "none";
    row.anchor_reject_count = 0;

    if (!isFusionOpenLocked()) {
        row.failure_reason = std::isfinite(fusion_initial_altitude_)
            ? "fusion_not_open_height_not_reached"
            : "fusion_not_open_initializing_altitude";
        return;
    }

    syncOnlineAnchorsUntilLocked(fix.t);
    row.anchor_reject_count = fusion_reject_count_this_fix_;
    row.anchor_reject_reason = fusion_last_anchor_reject_reason_;
    if (!fusion_has_active_anchor_) {
        ContinuousFusionRow continuous_row;
        if (!continuous_filter_.initialized ||
            !predictContinuousFusionLocked(fix.t, "fix_imu_only", &continuous_row)) {
            row.failure_reason =
                fusion_last_anchor_reject_reason_.empty()
                    ? "no_visual_anchor"
                    : fusion_last_anchor_reject_reason_;
            return;
        }

        row.pred_available = (std::isfinite(continuous_row.lon) &&
                              std::isfinite(continuous_row.lat) &&
                              std::isfinite(continuous_row.alt))
            ? 1
            : 0;
        row.imu_available = row.pred_available;
        if (row.pred_available == 1) {
            row.imu_lon = continuous_row.lon;
            row.imu_lat = continuous_row.lat;
            row.imu_alt = continuous_row.alt;
            row.pred_lon = continuous_row.lon;
            row.pred_lat = continuous_row.lat;
            row.pred_alt = continuous_row.alt;
            row.speed_mps = continuous_row.speed_mps;
            row.heading_used_deg = continuous_row.heading_used_deg;
            row.propagate_dt_sec = std::isfinite(continuous_row.propagate_dt_sec)
                ? continuous_row.propagate_dt_sec
                : 0.0;
            row.heading_source = "continuous_filter_imu_only";
            row.failure_reason.clear();
        } else {
            row.failure_reason = "continuous_filter_position_invalid";
        }
        return;
    }

    const OnlineYprCropNode::VisualAnchor& active_anchor = fusion_active_anchor_;
    row.visual_anchor_image_index = static_cast<int>(active_anchor.image_index);
    row.visual_anchor_stamp_sec = active_anchor.t;
    row.visual_anchor_lon = active_anchor.pos.lon;
    row.visual_anchor_lat = active_anchor.pos.lat;
    row.visual_anchor_alt = active_anchor.pos.alt;
    if (std::isfinite(active_anchor.heading_compass_rad)) {
        row.visual_anchor_heading_deg = active_anchor.heading_compass_rad * global_heading::RAD2DEG;
    }
    row.visual_anchor_sync_wait_ms = active_anchor.sync_wait_ms;
    row.visual_anchor_processing_ms = active_anchor.processing_ms;
    row.visual_anchor_topic_to_result_ms = active_anchor.topic_to_result_ms;
    if (std::isfinite(fix.t) && std::isfinite(active_anchor.t)) {
        row.visual_anchor_age_sec = std::max(0.0, fix.t - active_anchor.t);
    }
    if (std::isfinite(row.visual_anchor_age_sec) && std::isfinite(active_anchor.topic_to_result_ms)) {
        const double anchor_output_t = active_anchor.t + active_anchor.topic_to_result_ms / 1000.0;
        row.estimated_staleness_ms = std::max(0.0, fix.t - anchor_output_t) * 1000.0;
    }

    row.propagate_dt_sec =
        std::isfinite(fusion_last_prop_t_) ? std::max(0.0, fix.t - fusion_last_prop_t_) : 0.0;

    GeoPoint pred_geo;
    cv::Vec3d pred_vel_enu(0.0, 0.0, 0.0);
    OnlineYprCropNode::ContinuousPropagation prop;
    if (!predictGeoFromAnchorLocked(active_anchor,
                                    fusion_last_prop_t_,
                                    fusion_fused_pos_,
                                    fusion_fused_vel_enu_,
                                    fix.t,
                                    fusion_last_heading_used_rad_,
                                    pred_geo,
                                    pred_vel_enu,
                                    prop,
                                    row.imu_heading_delta_deg)) {
        row.failure_reason = "imu_preintegration_failed";
        resetFusionAnchorState(fusion_has_active_anchor_,
                               fusion_active_anchor_,
                               fusion_fused_pos_,
                               fusion_fused_vel_enu_,
                               fusion_last_prop_t_,
                               fusion_last_heading_used_rad_);
        fusion_last_anchor_reject_reason_ = row.failure_reason;
        return;
    }

    row.speed_mps = prop.speed_mps;
    row.heading_source = prop.heading_source;
    if (std::isfinite(prop.heading_used_rad)) {
        fusion_last_heading_used_rad_ = prop.heading_used_rad;
        row.heading_used_deg = prop.heading_used_rad * global_heading::RAD2DEG;
    }

    fusion_fused_pos_ = pred_geo;
    fusion_fused_vel_enu_ = pred_vel_enu;
    fusion_last_prop_t_ = fix.t;
    row.pred_available = (std::isfinite(pred_geo.lon) &&
                          std::isfinite(pred_geo.lat) &&
                          std::isfinite(pred_geo.alt))
        ? 1
        : 0;
    if (row.pred_available == 1) {
        row.imu_available = 1;
        row.imu_lon = pred_geo.lon;
        row.imu_lat = pred_geo.lat;
        row.imu_alt = pred_geo.alt;
        row.pred_lon = pred_geo.lon;
        row.pred_lat = pred_geo.lat;
        row.pred_alt = pred_geo.alt;
    } else {
        row.failure_reason = "fused_position_invalid";
    }
}

// 功能：在离线模式下统一写出全部 fix 融合结果。
void OnlineYprCropNode::writeFixFusionCsvOffline() {
    writeFixFusionCsvHeader(opt_.fix_fusion_csv);
    writeImuPositionCsvHeader(opt_.imu_position_csv);
    writeGnssPositionCsvHeader(opt_.gnss_position_csv);

    std::ofstream ofs(opt_.fix_fusion_csv, std::ios::app);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to append fix fusion csv: " + opt_.fix_fusion_csv);
    }
    std::ofstream imu_ofs(opt_.imu_position_csv, std::ios::app);
    if (!imu_ofs.is_open()) {
        throw std::runtime_error("Failed to append imu position csv: " + opt_.imu_position_csv);
    }
    std::ofstream gnss_ofs(opt_.gnss_position_csv, std::ios::app);
    if (!gnss_ofs.is_open()) {
        throw std::runtime_error("Failed to append gnss position csv: " + opt_.gnss_position_csv);
    }
    ofs << std::fixed << std::setprecision(9);
    imu_ofs << std::fixed << std::setprecision(9);
    gnss_ofs << std::fixed << std::setprecision(9);

    std::vector<FrameRecord> records_snapshot;
    std::vector<FixSample> fixes_snapshot;
    {
        std::lock_guard<std::mutex> records_lock(records_mutex_);
        records_snapshot = records_;
    }
    {
        std::lock_guard<std::mutex> state_lock(state_mutex_);
        fixes_snapshot = all_fixes_;
    }

    std::vector<OnlineYprCropNode::VisualAnchor> anchors;
    anchors.reserve(records_snapshot.size());
    for (const FrameRecord& rec : records_snapshot) {
        OnlineYprCropNode::VisualAnchor anchor;
        if (makeVisualAnchorFromRecord(rec, anchor)) {
            anchors.push_back(anchor);
        }
    }

    size_t next_anchor_idx = 0;
    bool has_active_anchor = false;
    OnlineYprCropNode::VisualAnchor active_anchor;
    GeoPoint fused_pos;
    cv::Vec3d fused_vel_enu(0.0, 0.0, 0.0);
    double last_prop_t = std::numeric_limits<double>::quiet_NaN();
    double last_heading_used_rad = std::numeric_limits<double>::quiet_NaN();
    std::string last_anchor_reject_reason;

    if (!fixes_snapshot.empty()) {
        const FixSample& first_fix = fixes_snapshot.front();
        active_anchor.image_index = 0;
        active_anchor.t = first_fix.t;
        active_anchor.pos = GeoPoint{first_fix.lon, first_fix.lat, first_fix.alt};
        global_heading::HeadingSample heading;
        if (heading_est_.query(first_fix.t, heading) && heading.valid) {
            active_anchor.heading_compass_rad = wrapToTwoPi(heading.psi_compass);
            last_heading_used_rad = active_anchor.heading_compass_rad;
        }
        active_anchor.imu_heading_valid =
            imu_tracker_.queryHeadingCompassRelativeRadCausal(first_fix.t, active_anchor.imu_heading_rel_rad);
        fused_pos = active_anchor.pos;
        fused_vel_enu = cv::Vec3d(0.0, 0.0, 0.0);
        last_prop_t = first_fix.t;
        has_active_anchor = true;
    }

    for (size_t i = 0; i < fixes_snapshot.size(); ++i) {
        const FixSample& fix = fixes_snapshot[i];
        int anchor_reject_count = 0;
        std::string anchor_reject_reason;
        double speed_hint_mps = 0.0;

        expireFusionAnchorIfNeeded(has_active_anchor,
                                   active_anchor,
                                   fused_pos,
                                   fused_vel_enu,
                                   last_prop_t,
                                   last_heading_used_rad,
                                   fix.t,
                                   last_anchor_reject_reason);
        if (last_anchor_reject_reason == "visual_anchor_expired") {
            anchor_reject_reason = last_anchor_reject_reason;
        }

        while (next_anchor_idx < anchors.size() && anchors[next_anchor_idx].t <= fix.t) {
            const OnlineYprCropNode::VisualAnchor candidate = anchors[next_anchor_idx];
            std::string reject_reason;
            if (isAnchorExpiredAtTime(candidate, fix.t)) {
                last_anchor_reject_reason = "anchor_rejected_stale";
                anchor_reject_reason = last_anchor_reject_reason;
                ++anchor_reject_count;
                ++next_anchor_idx;
                continue;
            }
            if (!passesVisualAnchorQualityGate(candidate, reject_reason)) {
                last_anchor_reject_reason = reject_reason;
                anchor_reject_reason = reject_reason;
                ++anchor_reject_count;
                ++next_anchor_idx;
                continue;
            }

            if (has_active_anchor) {
                const double innovation_m = geoHorizontalDistanceMeters(active_anchor.pos, candidate.pos);
                const double anchor_dt_sec = std::max(0.0, candidate.t - active_anchor.t);
                const double innovation_gate_m =
                    computeAnchorInnovationGateMeters(speed_hint_mps, anchor_dt_sec);
                if (std::isfinite(innovation_m) && innovation_m > innovation_gate_m) {
                    last_anchor_reject_reason = "anchor_rejected_large_innovation";
                    anchor_reject_reason = last_anchor_reject_reason;
                    ++anchor_reject_count;
                    ++next_anchor_idx;
                    continue;
                }
            }

            fused_pos = candidate.pos;
            fused_vel_enu = candidate.vel_enu;
            active_anchor = candidate;
            has_active_anchor = true;
            last_prop_t = candidate.t;
            last_heading_used_rad = active_anchor.heading_compass_rad;
            last_anchor_reject_reason.clear();
            ++next_anchor_idx;
        }

        expireFusionAnchorIfNeeded(has_active_anchor,
                                   active_anchor,
                                   fused_pos,
                                   fused_vel_enu,
                                   last_prop_t,
                                   last_heading_used_rad,
                                   fix.t,
                                   last_anchor_reject_reason);
        if (anchor_reject_reason.empty() && last_anchor_reject_reason == "visual_anchor_expired") {
            anchor_reject_reason = last_anchor_reject_reason;
        }

        int pred_available = 0;
        double pred_lon = std::numeric_limits<double>::quiet_NaN();
        double pred_lat = std::numeric_limits<double>::quiet_NaN();
        double pred_alt = std::numeric_limits<double>::quiet_NaN();
        int imu_available = 0;
        double imu_lon = std::numeric_limits<double>::quiet_NaN();
        double imu_lat = std::numeric_limits<double>::quiet_NaN();
        double imu_alt = std::numeric_limits<double>::quiet_NaN();
        int anchor_image_index = -1;
        double anchor_stamp_sec = std::numeric_limits<double>::quiet_NaN();
        double anchor_lon = std::numeric_limits<double>::quiet_NaN();
        double anchor_lat = std::numeric_limits<double>::quiet_NaN();
        double anchor_alt = std::numeric_limits<double>::quiet_NaN();
        double anchor_heading_deg = std::numeric_limits<double>::quiet_NaN();
        double imu_heading_delta_deg = std::numeric_limits<double>::quiet_NaN();
        double heading_used_deg = std::numeric_limits<double>::quiet_NaN();
        double speed_mps = 0.0;
        double propagate_dt_sec = 0.0;
        double visual_anchor_age_sec = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_sync_wait_ms = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_processing_ms = std::numeric_limits<double>::quiet_NaN();
        double visual_anchor_topic_to_result_ms = std::numeric_limits<double>::quiet_NaN();
        double estimated_staleness_ms = std::numeric_limits<double>::quiet_NaN();
        double fix_compute_latency_ms = std::numeric_limits<double>::quiet_NaN();
        double fix_topic_to_fusion_result_ms = std::numeric_limits<double>::quiet_NaN();
        std::string heading_source = "none";
        std::string failure_reason;

        if (!has_active_anchor) {
            failure_reason =
                last_anchor_reject_reason.empty() ? "no_visual_anchor" : last_anchor_reject_reason;
        } else {
            anchor_image_index = static_cast<int>(active_anchor.image_index);
            anchor_stamp_sec = active_anchor.t;
            anchor_lon = active_anchor.pos.lon;
            anchor_lat = active_anchor.pos.lat;
            anchor_alt = active_anchor.pos.alt;
            if (std::isfinite(active_anchor.heading_compass_rad)) {
                anchor_heading_deg = active_anchor.heading_compass_rad * global_heading::RAD2DEG;
            }
            visual_anchor_sync_wait_ms = active_anchor.sync_wait_ms;
            visual_anchor_processing_ms = active_anchor.processing_ms;
            visual_anchor_topic_to_result_ms = active_anchor.topic_to_result_ms;
            if (std::isfinite(fix.t) && std::isfinite(active_anchor.t)) {
                visual_anchor_age_sec = std::max(0.0, fix.t - active_anchor.t);
            }
            if (std::isfinite(visual_anchor_age_sec) && std::isfinite(active_anchor.topic_to_result_ms)) {
                const double anchor_output_t = active_anchor.t + active_anchor.topic_to_result_ms / 1000.0;
                estimated_staleness_ms = std::max(0.0, fix.t - anchor_output_t) * 1000.0;
            }

            propagate_dt_sec = std::isfinite(last_prop_t) ? std::max(0.0, fix.t - last_prop_t) : 0.0;

            GeoPoint pred_geo;
            cv::Vec3d pred_vel_enu(0.0, 0.0, 0.0);
            OnlineYprCropNode::ContinuousPropagation prop;
            if (predictGeoFromAnchorLocked(active_anchor,
                                           last_prop_t,
                                           fused_pos,
                                           fused_vel_enu,
                                           fix.t,
                                           last_heading_used_rad,
                                           pred_geo,
                                           pred_vel_enu,
                                           prop,
                                           imu_heading_delta_deg)) {
                speed_mps = prop.speed_mps;
                heading_source = prop.heading_source;
                if (std::isfinite(prop.heading_used_rad)) {
                    last_heading_used_rad = prop.heading_used_rad;
                    heading_used_deg = prop.heading_used_rad * global_heading::RAD2DEG;
                }
                fused_pos = pred_geo;
                fused_vel_enu = pred_vel_enu;
                last_prop_t = fix.t;
            } else {
                failure_reason = "imu_preintegration_failed";
                resetFusionAnchorState(has_active_anchor,
                                       active_anchor,
                                       fused_pos,
                                       fused_vel_enu,
                                       last_prop_t,
                                       last_heading_used_rad);
                last_anchor_reject_reason = failure_reason;
            }

            pred_available = failure_reason.empty() &&
                             (std::isfinite(fused_pos.lon) &&
                              std::isfinite(fused_pos.lat) &&
                              std::isfinite(fused_pos.alt))
                ? 1
                : 0;
            if (pred_available == 1) {
                imu_available = 1;
                imu_lon = fused_pos.lon;
                imu_lat = fused_pos.lat;
                imu_alt = fused_pos.alt;
                pred_lon = fused_pos.lon;
                pred_lat = fused_pos.lat;
                pred_alt = fused_pos.alt;
            } else {
                failure_reason = "fused_position_invalid";
            }
        }

        OnlineYprCropNode::FixFusionRow row;
        row.fix_index = i + 1;
        row.fix_stamp_sec = fix.t;
        row.gt_lon = fix.lon;
        row.gt_lat = fix.lat;
        row.gt_alt = fix.alt;
        row.pred_available = pred_available;
        row.pred_lon = pred_lon;
        row.pred_lat = pred_lat;
        row.pred_alt = pred_alt;
        row.imu_available = imu_available;
        row.imu_lon = imu_lon;
        row.imu_lat = imu_lat;
        row.imu_alt = imu_alt;
        row.visual_anchor_image_index = anchor_image_index;
        row.visual_anchor_stamp_sec = anchor_stamp_sec;
        row.visual_anchor_lon = anchor_lon;
        row.visual_anchor_lat = anchor_lat;
        row.visual_anchor_alt = anchor_alt;
        row.visual_anchor_heading_deg = anchor_heading_deg;
        row.imu_heading_delta_deg = imu_heading_delta_deg;
        row.heading_used_deg = heading_used_deg;
        row.speed_mps = speed_mps;
        row.propagate_dt_sec = propagate_dt_sec;
        row.visual_anchor_age_sec = visual_anchor_age_sec;
        row.visual_anchor_sync_wait_ms = visual_anchor_sync_wait_ms;
        row.visual_anchor_processing_ms = visual_anchor_processing_ms;
        row.visual_anchor_topic_to_result_ms = visual_anchor_topic_to_result_ms;
        row.estimated_staleness_ms = estimated_staleness_ms;
        row.fix_compute_latency_ms = fix_compute_latency_ms;
        row.fix_topic_to_fusion_result_ms = fix_topic_to_fusion_result_ms;
        row.anchor_reject_count = anchor_reject_count;
        row.anchor_reject_reason = anchor_reject_reason;
        row.heading_source = heading_source;
        row.failure_reason = failure_reason;
        writeFixFusionCsvRow(ofs, row);
        writeImuPositionCsvRow(imu_ofs, row);
        writeGnssPositionCsvRow(gnss_ofs, row);
    }


    std::cout << "[Fusion] wrote fix_fusion_csv rows=" << fixes_snapshot.size()
              << " anchors=" << anchors.size()
              << " path=" << opt_.fix_fusion_csv << "\n";
}
