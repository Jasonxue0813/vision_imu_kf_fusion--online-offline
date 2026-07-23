/**
 * @file   global_heading.h
 * @brief  基于 GNSS 连续位置的航向角解算 —— ENU + RFU
 *
 * 实现要点：
 *   1. 经纬度 -> ENU 平面增量（WGS84 椭球，局部曲率半径）
 *   2. 滑动窗口最小二乘拟合速度向量 (vE, vN)
 *   3. atan2(vN, vE) 得数学惯例航向 psi_math，并 unwrap
 *   4. 历史缓冲 + 时间戳查询（角度线性内插），按视觉帧频率输出
 *
 * 用法（伪流程）：
 *     GlobalHeadingEstimator est(1.5, 10, 2.0, 3.0);
 *     // GNSS 回调 (10 Hz)
 *     est.update(t_gps, lat_deg, lon_deg);
 *     // 视觉帧回调 (2-5 Hz)
 *     HeadingSample s;
 *     if (est.query(t_vision, s)) {
 *         // s.psi_math, s.psi_compass, s.valid
 *     }
 */

#ifndef GLOBAL_HEADING_H
#define GLOBAL_HEADING_H

#include <cstddef>
#include <deque>

namespace global_heading {

// =============================================================
// WGS84 椭球常量
// =============================================================
constexpr double WGS84_A   = 6378137.0;                 // 长半轴 (m)
constexpr double WGS84_E2  = 6.69437999014e-3;          // 第一偏心率平方
constexpr double PI        = 3.14159265358979323846;
constexpr double TWO_PI    = 2.0 * PI;
constexpr double HALF_PI   = 0.5 * PI;
constexpr double DEG2RAD   = PI / 180.0;
constexpr double RAD2DEG   = 180.0 / PI;

// =============================================================
// 单条 GNSS 样本（输入缓冲）
// =============================================================
struct GlobalHeadingPoint {
    double t;        // 时间戳 (s)
    double lat_rad;  // 纬度 (rad)
    double lon_rad;  // 经度 (rad)
};

// =============================================================
// 航向角输出样本（历史缓冲 / query 结果）
// =============================================================
struct HeadingSample {
    double t            = 0.0;     // 该样本对应的 GNSS 时间戳 (s)
    double psi_math     = 0.0;     // 数学惯例：东=0, 逆时针正, (-pi, pi]
    double psi_compass  = 0.0;     // 罗盘惯例：北=0, 顺时针正, [0, 2pi)
    double speed_mps    = 0.0;     // 水平速度大小 (m/s)
    double residual_rms = 0.0;     // 拟合残差 RMS (m)
    bool   valid        = false;   // 该时刻航向是否可信
};

// =============================================================
// 工具函数声明
// =============================================================

/**
 * @brief 计算 ENU 局部平面位置增量（短距离近似）
// 功能：计算两个经纬度点之间的局部 ENU 增量。
 * @param lat1_rad  起点纬度 (rad)
 * @param lon1_rad  起点经度 (rad)
 * @param lat2_rad  终点纬度 (rad)
 * @param lon2_rad  终点经度 (rad)
 * @param dE [out]  东向增量 (m)
 * @param dN [out]  北向增量 (m)
 */
void geodeticToEnuDelta(double lat1_rad, double lon1_rad,
                        double lat2_rad, double lon2_rad,
                        double& dE, double& dN);

/**
 * @brief 角度 wrap 到 (-pi, pi]
 */
double wrapToPi(double a);

/**
 * @brief ψ_math (东=0, 逆时针) -> ψ_compass (北=0, 顺时针)
 */
double mathToCompass(double psi_math);

// =============================================================
// 主估计器
// =============================================================
class GlobalHeadingEstimator {
public:
    /**
     * @param window_sec     滑窗时长 (s)，推荐 1.5
     * @param gnss_rate_hz   GNSS 名义频率 (Hz)，用于估算窗口点数
     * @param min_speed_mps  低速门限，低于此值航向无效，推荐 2.0
     * @param max_residual_m 直线性残差阈值 (m)，推荐 ~3*σ_p
     * @param history_sec    历史缓冲时长 (s)，覆盖最坏视觉延迟，推荐 5.0
     */
    GlobalHeadingEstimator(double window_sec     = 1.5,
                         double gnss_rate_hz   = 10.0,
                         double min_speed_mps  = 2.0,
                         double max_residual_m = 3.0,
                         double history_sec    = 5.0);

    /**
     * @brief GNSS 回调，每来一个点调用一次
     * @return 当前估计的样本（包含 valid 标志）
     */
// 功能：用一条新的 GNSS 样本更新航向估计。
    HeadingSample update(double t_sec, double lat_deg, double lon_deg);

    /**
     * @brief 按时间戳查询航向（线性内插）
     * @param t_query   查询时刻 (s)
     * @param out [out] 输出样本
     * @return 是否查到（缓冲为空 / t 太早返回 false）
     */
    bool query(double t_query, HeadingSample& out) const;

    /** 当前缓冲样本数（调试用） */
    std::size_t windowSize()  const { return win_.size(); }
    std::size_t historySize() const { return history_.size(); }

private:
    // 参数
    std::size_t window_len_;
    double      min_speed_;
    double      max_residual_;
    std::size_t history_max_;

    // 输入滑窗（最近 window_len_ 个 GNSS 点）
    std::deque<GlobalHeadingPoint> win_;

    // 输出历史缓冲（供 query 使用）
    std::deque<HeadingSample> history_;

    // 上一次有效 psi_math（用于 unwrap）
    double last_psi_math_;
    bool   last_valid_;

    // 内部：把 win_ 转成 ENU 序列并拟合速度向量
// 功能：在当前 GNSS 滑窗上拟合水平速度向量。
    bool fitVelocity(double& vE, double& vN, double& rms) const;
};

} // namespace global_heading

#endif // GLOBAL_HEADING_H
