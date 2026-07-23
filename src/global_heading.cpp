/**
 * @file global_heading.cpp
 * @brief 基于 GNSS 连续位置的航向角解算（实现）
 */

#include "global_heading.h"

#include <cmath>
#include <algorithm>
#include <vector>

namespace global_heading {

// =============================================================
// 工具函数
// =============================================================

// 功能：计算两个经纬度点之间的局部 ENU 增量。
void geodeticToEnuDelta(double lat1_rad, double lon1_rad,
                        double lat2_rad, double lon2_rad,
                        double& dE, double& dN)
{
    // 用 lat1 作为参考纬度，计算局部曲率半径（短距离近似）
    const double sin_lat = std::sin(lat1_rad);
    const double denom   = 1.0 - WGS84_E2 * sin_lat * sin_lat;
    const double R_N     = WGS84_A / std::sqrt(denom);                  // 卯酉圈
    const double R_M     = WGS84_A * (1.0 - WGS84_E2) / (denom * std::sqrt(denom));  // 子午圈

    dE = (lon2_rad - lon1_rad) * R_N * std::cos(lat1_rad);
    dN = (lat2_rad - lat1_rad) * R_M;
}

// 功能：把角度约束到 `(-pi, pi]` 区间。
double wrapToPi(double a)
{
    // 把 a 规范到 (-pi, pi]
    a = std::fmod(a + PI, TWO_PI);
    if (a <= 0.0) a += TWO_PI;
    return a - PI;
}

// 功能：把数学坐标系航向转换成罗盘坐标系航向。
double mathToCompass(double psi_math)
{
    double c = HALF_PI - psi_math;
    c = std::fmod(c, TWO_PI);
    if (c < 0.0) c += TWO_PI;
    return c;
}

// =============================================================
// 构造
// =============================================================

// 功能：`GlobalHeadingEstimator` 的构造函数，用于设置滑窗和历史缓存参数。
GlobalHeadingEstimator::GlobalHeadingEstimator(double window_sec,
                                           double gnss_rate_hz,
                                           double min_speed_mps,
                                           double max_residual_m,
                                           double history_sec)
    : min_speed_(min_speed_mps),
      max_residual_(max_residual_m),
      last_psi_math_(0.0),
      last_valid_(false)
{
    int wl = static_cast<int>(window_sec * gnss_rate_hz);
    if (wl < 3) wl = 3;
    window_len_ = static_cast<std::size_t>(wl);

    int hl = static_cast<int>(history_sec * gnss_rate_hz) + 10;
    if (hl < 10) hl = 10;
    history_max_ = static_cast<std::size_t>(hl);
}

// =============================================================
// 内部：在当前 win_ 上做最小二乘拟合速度向量
// =============================================================

// 功能：在当前 GNSS 滑窗上拟合水平速度向量。
bool GlobalHeadingEstimator::fitVelocity(double& vE, double& vN, double& rms) const
{
    const std::size_t N = win_.size();
    if (N < 3) return false;

    // 以窗口第一个点为参考点，把所有点转成 ENU 增量
    const double lat0 = win_.front().lat_rad;
    const double lon0 = win_.front().lon_rad;
    const double t0   = win_.front().t;

    // 预计算均值，避免数值灾难
    double t_sum = 0.0, E_sum = 0.0, N_sum = 0.0;

    // 临时存储
    // （为避免堆分配，可改成栈数组；这里保持清晰）
    std::vector<double> ts(N), Es(N), Ns(N);

    for (std::size_t i = 0; i < N; ++i) {
        const GlobalHeadingPoint& p = win_[i];
        double dE = 0.0, dN = 0.0;
        geodeticToEnuDelta(lat0, lon0, p.lat_rad, p.lon_rad, dE, dN);
        ts[i] = p.t - t0;
        Es[i] = dE;
        Ns[i] = dN;
        t_sum += ts[i];
        E_sum += dE;
        N_sum += dN;
    }
    const double t_mean = t_sum / N;
    const double E_mean = E_sum / N;
    const double N_mean = N_sum / N;

    // 最小二乘：斜率 = sum((t-tm)*(y-ym)) / sum((t-tm)^2)
    double Stt = 0.0, StE = 0.0, StN = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double dt = ts[i] - t_mean;
        Stt += dt * dt;
        StE += dt * (Es[i] - E_mean);
        StN += dt * (Ns[i] - N_mean);
    }
    if (Stt <= 0.0) return false;   // 所有 t 相同（异常）

    vE = StE / Stt;
    vN = StN / Stt;

    // 截距
    const double bE = E_mean - vE * t_mean;
    const double bN = N_mean - vN * t_mean;

    // 残差 RMS
    double sse = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double rE = Es[i] - (vE * ts[i] + bE);
        const double rN = Ns[i] - (vN * ts[i] + bN);
        sse += rE * rE + rN * rN;
    }
    rms = std::sqrt(sse / static_cast<double>(N));

    return true;
}

// =============================================================
// 主入口：global 更新
// =============================================================

// 功能：用一条新的 global 样本更新航向估计。
HeadingSample GlobalHeadingEstimator::update(double t_sec, double lat_deg, double lon_deg)
{
    // 1) 入滑窗
    GlobalHeadingPoint p;
    p.t       = t_sec;
    p.lat_rad = lat_deg * DEG2RAD;
    p.lon_rad = lon_deg * DEG2RAD;
    win_.push_back(p);
    while (win_.size() > window_len_) win_.pop_front();

    HeadingSample s;
    s.t = t_sec;

    // 2) 样本不足 -> 沿用上一次（标记无效）
    if (win_.size() < 3) {
        s.psi_math    = last_valid_ ? last_psi_math_ : 0.0;
        s.psi_compass = mathToCompass(s.psi_math);
        s.valid       = false;
        // 入历史缓冲
        history_.push_back(s);
        while (history_.size() > history_max_) history_.pop_front();
        return s;
    }

    // 3) 拟合
    double vE = 0.0, vN = 0.0, rms = 0.0;
    bool ok = fitVelocity(vE, vN, rms);
    s.residual_rms = rms;

    if (!ok) {
        s.psi_math    = last_valid_ ? last_psi_math_ : 0.0;
        s.psi_compass = mathToCompass(s.psi_math);
        s.valid       = false;
    } else {
        const double speed = std::hypot(vE, vN);
        s.speed_mps = speed;

        bool valid = true;
        if (speed < min_speed_)        valid = false;   // 低速
        if (rms   > max_residual_)     valid = false;   // 非直线

        if (!valid) {
            // 无效：保持上一次（如果有）
            s.psi_math    = last_valid_ ? last_psi_math_ : std::atan2(vN, vE);
            s.psi_compass = mathToCompass(s.psi_math);
            s.valid       = false;
        } else {
            // 有效：算 psi_math 并 unwrap
            double psi = std::atan2(vN, vE);   // (-pi, pi]
            if (last_valid_) {
                double d = psi - last_psi_math_;
                // 使 |d| <= pi，从而 psi 连续
                if      (d >  PI) psi -= TWO_PI;
                else if (d < -PI) psi += TWO_PI;
            }
            s.psi_math    = psi;
            s.psi_compass = mathToCompass(psi);
            s.valid       = true;
            last_psi_math_ = psi;
            last_valid_    = true;
        }
    }

    // 4) 入历史缓冲
    history_.push_back(s);
    while (history_.size() > history_max_) history_.pop_front();
    return s;
}

// =============================================================
// 按时间戳查询（角度线性内插）
// =============================================================

// 功能：按时间戳查询历史航向估计结果。
bool GlobalHeadingEstimator::query(double t_query, HeadingSample& out) const
{
    if (history_.empty()) return false;

    // 边界处理
    if (t_query <= history_.front().t) {
        out = history_.front();
        return true;
    }
    if (t_query >= history_.back().t) {
        out = history_.back();
        return true;
    }

    // 在 history_ 中找到 [t1, t2] 包住 t_query
    // history_ 按时间单调递增（GNSS 点单调到达）
    for (std::size_t i = 1; i < history_.size(); ++i) {
        const HeadingSample& s1 = history_[i - 1];
        const HeadingSample& s2 = history_[i];
        if (s2.t >= t_query && s1.t <= t_query) {
            const double dt = s2.t - s1.t;
            if (dt <= 0.0) { out = s2; return true; }
            const double r = (t_query - s1.t) / dt;

            // 角度差值 unwrap 后再线性内插
            double dpsi = s2.psi_math - s1.psi_math;
            if      (dpsi >  PI) dpsi -= TWO_PI;
            else if (dpsi < -PI) dpsi += TWO_PI;

            double psi = s1.psi_math + r * dpsi;
            psi = wrapToPi(psi);

            out.t            = t_query;
            out.psi_math     = psi;
            out.psi_compass  = mathToCompass(psi);
            out.speed_mps    = s1.speed_mps + r * (s2.speed_mps - s1.speed_mps);
            out.residual_rms = std::max(s1.residual_rms, s2.residual_rms);
            // 只有两端都 valid 才认为内插结果 valid
            out.valid        = s1.valid && s2.valid;
            return true;
        }
    }
    out = history_.back();
    return true;
}

} // namespace global_heading
