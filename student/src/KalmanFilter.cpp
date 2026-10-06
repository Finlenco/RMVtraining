#include "EkfTracker.hpp"
#include <KalmanFilter.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {

// [xc,vx,yc,vy,z_even,z_odd,r1,r2,yaw,vyaw,vz]
constexpr int kStateSize = 11;
constexpr int kCx = 0, kVx = 1, kCy = 2, kVy = 3;
constexpr int kZEven = 4, kZOdd = 5, kR1 = 6, kR2 = 7, kYaw = 8, kVyaw = 9, kVz = 10;
constexpr int kSlotCount = 4, kSlotMeasurementSize = 4;
constexpr double kPi = 3.1415926;
constexpr double kInitialRadius = 0.25;
constexpr double kMinRadius = 0.12, kMaxRadius = 0.40;
constexpr double kMatchDistanceM = 0.12, kMatchYawDifference = 0.6;
constexpr double kMismatchDistanceM = 0.07, kMismatchYawDifference = 0.35;
constexpr double kYawSigma = 0.12;
constexpr double kMaxAcceptedYawStep = 0.25;
constexpr double kMaxAngularVelocity = 8.0;
constexpr double kMaxAngularAcceleration = 20.0;
constexpr int kStableUpdates = 3, kLockFrames = 10;
constexpr int kUnlockLostFrames = 5, kMaxMissedFrames = 50;

struct FilterContext {
    double last_timestamp{0.0}, dt{0.02};
    int missed_frames{0}, stable_updates{0}, lock_frames{0};
    bool target_locked{false};
    std::uint8_t target_id{0};
    std::array<double, kSlotCount> measurement_sigma{0.02, 0.02, 0.02, 0.02};
    std::array<double, kSlotCount> yaw_sigma{kYawSigma, kYawSigma, kYawSigma, kYawSigma};
    std::array<bool, kSlotCount> slot_observed{};
    std::unique_ptr<ExtendedKalmanFilter> filter;
};

FilterContext& context() { static FilterContext tracker; return tracker; }
double wrapAngle(double a) { return std::remainder(a, 2.0 * kPi); }

Eigen::Vector3d position(const ArmorPose& p) {
    return {p.position_gimbal_m[0], p.position_gimbal_m[1], p.position_gimbal_m[2]};
}

bool valid(const ArmorPose& p) {
    const auto xyz = position(p);
    return xyz.allFinite() && std::isfinite(p.armor_yaw) && p.detection.target_id > 0 &&
           p.detection.target_id <= 255 && xyz.x() > 0.0 && xyz.norm() >= 0.05 &&
           xyz.norm() <= 30.0 && (p.reprojection_error <= 0.0 || p.reprojection_error < 20.0);
}

double radius(const Eigen::VectorXd& x, int slot) { return slot % 2 == 0 ? x[kR1] : x[kR2]; }
int heightIndex(int slot) { return slot % 2 == 0 ? kZEven : kZOdd; }

Eigen::Vector3d slotPosition(const Eigen::VectorXd& x, int slot) {
    const double angle = x[kYaw] + slot * kPi / 2.0, r = radius(x, slot);
    return {x[kCx] - r * std::cos(angle), x[kCy] - r * std::sin(angle), x[heightIndex(slot)]};
}

Eigen::VectorXd observationModel(const Eigen::VectorXd& x) {
    Eigen::VectorXd z(kSlotCount * kSlotMeasurementSize);
    for (int slot = 0; slot < kSlotCount; ++slot) {
        const int i = slot * kSlotMeasurementSize;
        z.segment<3>(i) = slotPosition(x, slot);
        z[i + 3] = wrapAngle(x[kYaw] + slot * kPi / 2.0);
    }
    return z;
}

Eigen::MatrixXd observationJacobian(const Eigen::VectorXd& x) {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(kSlotCount * kSlotMeasurementSize, kStateSize);
    for (int slot = 0; slot < kSlotCount; ++slot) {
        const double angle = x[kYaw] + slot * kPi / 2.0, r = radius(x, slot);
        const int i = slot * kSlotMeasurementSize, ri = slot % 2 == 0 ? kR1 : kR2;
        h(i, kCx) = 1.0; h(i + 1, kCy) = 1.0; h(i + 2, heightIndex(slot)) = 1.0;
        h(i, kYaw) = r * std::sin(angle); h(i + 1, kYaw) = -r * std::cos(angle);
        h(i, ri) = -std::cos(angle); h(i + 1, ri) = -std::sin(angle);
        h(i + 3, kYaw) = 1.0;
    }
    return h;
}

void setConstantVelocityNoise(Eigen::MatrixXd& q, int p, int v, double variance, double dt) {
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
    q(p, p) = variance * dt4 / 4.0; q(p, v) = q(v, p) = variance * dt3 / 2.0;
    q(v, v) = variance * dt2;
}

void limitState(Eigen::VectorXd& x, const FilterContext& t) {
    x[kVyaw] = std::clamp(x[kVyaw], -kMaxAngularVelocity, kMaxAngularVelocity);
    x[kVz] = std::clamp(x[kVz], -5.0, 5.0); x[kYaw] = wrapAngle(x[kYaw]);
    const double lo = t.target_locked ? 0.20 : kMinRadius;
    const double hi = t.target_locked ? 0.30 : kMaxRadius;
    x[kR1] = std::clamp(x[kR1], lo, hi); x[kR2] = std::clamp(x[kR2], lo, hi);
}

void updateMeasurementNoise(FilterContext& t, const ArmorPose& p, int slot) {
    if (p.reprojection_error > 0.0) {
        t.measurement_sigma[slot] = std::clamp(0.015 + p.reprojection_error * 0.002, 0.015, 0.10);
        t.yaw_sigma[slot] = std::clamp(0.10 + p.reprojection_error * 0.012, 0.10, 0.25);
    } else {
        t.measurement_sigma[slot] = 0.02;
        t.yaw_sigma[slot] = kYawSigma;
    }
}

void createFilter(FilterContext& t, const ArmorPose& p) {
    const auto xyz = position(p);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(kStateSize);
    x[kCx] = xyz.x() + kInitialRadius * std::cos(p.armor_yaw);
    x[kCy] = xyz.y() + kInitialRadius * std::sin(p.armor_yaw);
    x[kYaw] = p.armor_yaw; x[kR1] = x[kR2] = kInitialRadius;
    x[kZEven] = x[kZOdd] = xyz.z();

    Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
    p0(kCx, kCx) = p0(kCy, kCy) = 0.25 * 0.25;
    p0(kVx, kVx) = p0(kVy, kVy) = p0(kVyaw, kVyaw) = 4.0;
    p0(kYaw, kYaw) = 0.25;
    p0(kR1, kR1) = p0(kR2, kR2) = 0.08 * 0.08;
    p0(kZEven, kZEven) = p0(kZOdd, kZOdd) = 0.01;

    auto f = [&t](const Eigen::VectorXd& s) {
        Eigen::VectorXd n = s;
        n[kCx] += t.dt * s[kVx]; n[kCy] += t.dt * s[kVy];
        n[kYaw] = wrapAngle(s[kYaw] + t.dt * s[kVyaw]);
        n[kZEven] += t.dt * s[kVz]; n[kZOdd] += t.dt * s[kVz];
        return n;
    };
    auto jf = [&t](const Eigen::VectorXd&) {
        Eigen::MatrixXd j = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
        j(kCx, kVx) = j(kCy, kVy) = j(kYaw, kVyaw) = t.dt;
        j(kZEven, kVz) = j(kZOdd, kVz) = t.dt;
        return j;
    };
    auto q = [&t]() {
        Eigen::MatrixXd n = Eigen::MatrixXd::Zero(kStateSize, kStateSize);
        setConstantVelocityNoise(n, kCx, kVx, 2.0, t.dt);
        setConstantVelocityNoise(n, kCy, kVy, 2.0, t.dt);
        setConstantVelocityNoise(n, kYaw, kVyaw, 4.0, t.dt);
        n(kVz, kVz) = 0.05 * t.dt;
        n(kR1, kR1) = n(kR2, kR2) = 1.0e-4 * t.dt;
        n(kZEven, kZEven) = n(kZOdd, kZOdd) = 0.01 * t.dt;
        return n;
    };
    auto r = [&t](const Eigen::VectorXd&) {
    Eigen::MatrixXd n = Eigen::MatrixXd::Zero(kSlotCount * kSlotMeasurementSize,
                                               kSlotCount * kSlotMeasurementSize);
        for (int slot = 0; slot < kSlotCount; ++slot) {
            const int i = slot * kSlotMeasurementSize;
            const double scale = t.slot_observed[slot] ? 1.0 : 25.0;
            const double sigma = t.measurement_sigma[slot];
            const double pv = sigma * sigma * scale;
            n(i, i) = n(i + 1, i + 1) = n(i + 2, i + 2) = pv;
            n(i + 3, i + 3) = t.yaw_sigma[slot] * t.yaw_sigma[slot] * scale;
        }
        return n;
    };
    auto normalize = [](const Eigen::VectorXd& residual) {
        Eigen::VectorXd n = residual;
        for (int i = 0; i < n.size(); i += kSlotMeasurementSize) n[i + 3] = wrapAngle(n[i + 3]);
        return n;
    };
    t.filter = std::make_unique<ExtendedKalmanFilter>(f, observationModel, jf, observationJacobian,
        q, r, normalize, p0, x);
}

struct Match {
    const ArmorPose* pose{nullptr};
    double distance{INFINITY}, yaw{INFINITY}, cost{INFINITY};
    bool same_id{false};
};

std::array<Match, kSlotCount> matchSlots(const FilterContext& t,
                                         const std::vector<ArmorPose>& observations) {
    std::array<Match, kSlotCount> matches{};
    const auto& x = t.filter->get_nochange_X();
    for (const auto& p : observations) {
        if (!valid(p)) continue;
        const bool same_id = p.detection.target_id == t.target_id;
        const double max_distance = same_id ? kMatchDistanceM : kMismatchDistanceM;
        const double max_yaw = same_id ? kMatchYawDifference : kMismatchYawDifference;
        int best_slot = -1;
        Match candidate;
        for (int slot = 0; slot < kSlotCount; ++slot) {
            const double d = (position(p) - slotPosition(x, slot)).norm();
            const double dy = std::abs(wrapAngle(
                p.armor_yaw - (x[kYaw] + slot * kPi / 2.0)));
            if (d >= max_distance || dy >= max_yaw) continue;
            const double cost = d + 0.08 * dy;
            if (cost < candidate.cost) {
                candidate = {&p, d, dy, cost, same_id};
                best_slot = slot;
            }
        }
        if (best_slot < 0) continue;
        if (!matches[best_slot].pose || (same_id && !matches[best_slot].same_id) ||
            (same_id == matches[best_slot].same_id && candidate.cost < matches[best_slot].cost)) {
            matches[best_slot] = candidate;
        }
    }
    return matches;
}

Eigen::VectorXd makeMeasurement(const Eigen::VectorXd& predicted,
                                const std::array<Match, kSlotCount>& matches,
                                const FilterContext& t) {
    Eigen::VectorXd z = predicted;
    for (int slot = 0; slot < kSlotCount; ++slot) {
        if (!t.slot_observed[slot]) continue;
        const int i = slot * kSlotMeasurementSize;
        z.segment<3>(i) = position(*matches[slot].pose);
        z[i + 3] = matches[slot].pose->armor_yaw;
    }
    return z;
}

void rejectYawOutliers(Eigen::VectorXd& measurement, const Eigen::VectorXd& predicted,
                       const FilterContext& t) {
    for (int slot = 0; slot < kSlotCount; ++slot) {
        if (!t.slot_observed[slot]) continue;
        const int i = slot * kSlotMeasurementSize + 3;
        if (std::abs(wrapAngle(measurement[i] - predicted[i])) > kMaxAcceptedYawStep) {
            measurement[i] = predicted[i];
        }
    }
}

void limitAngularUpdate(Eigen::VectorXd& state, const Eigen::VectorXd& previous,
                        double dt, const FilterContext& t) {
    const double yaw_step = wrapAngle(state[kYaw] - previous[kYaw]);
    state[kYaw] = previous[kYaw] +
                  std::clamp(yaw_step, -kMaxAcceptedYawStep, kMaxAcceptedYawStep);
    const double max_velocity_step = kMaxAngularAcceleration * std::max(dt, 0.001);
    state[kVyaw] = previous[kVyaw] +
                   std::clamp(state[kVyaw] - previous[kVyaw],
                              -max_velocity_step, max_velocity_step);
    state[kVyaw] = std::clamp(state[kVyaw], -kMaxAngularVelocity, kMaxAngularVelocity);
    (void)t;
}

double processingDelay(double timestamp) {
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const double delay = now - timestamp;
    return std::isfinite(delay) && delay >= 0.0 && delay < 0.5 ? delay : 0.0;
}
}

namespace ekf_tracker {
void reset() { context() = FilterContext{}; }

PredictionResult update(const std::vector<ArmorPose>& observations, const GimbalState& gimbal,
                        double timestamp_seconds) {
    auto& t = context();
    if (!std::isfinite(timestamp_seconds)) return {};
    if (t.filter) {
        const double elapsed = timestamp_seconds - t.last_timestamp;
        if (elapsed <= 0.0 || elapsed > 0.5) reset();
        else { t.dt = std::clamp(elapsed, 0.001, 0.1); t.filter->predict(); }
    }
    t.last_timestamp = timestamp_seconds;

    if (!t.filter) {
        for (const auto& p : observations) if (valid(p)) {
            updateMeasurementNoise(t, p, 0); createFilter(t, p);
            t.target_id = static_cast<std::uint8_t>(p.detection.target_id);
            t.stable_updates = t.lock_frames = 1;
            t.slot_observed.fill(false); t.slot_observed[0] = true;
            break;
        }
    } else {
        const auto matches = matchSlots(t, observations);
        t.slot_observed.fill(false);
        int count = 0;
        for (int i = 0; i < kSlotCount; ++i) {
            t.slot_observed[i] = matches[i].pose != nullptr;
            if (t.slot_observed[i]) {
                updateMeasurementNoise(t, *matches[i].pose, i);
                ++count;
            }
        }
        if (count > 0) {
            const auto predicted = observationModel(t.filter->get_nochange_X());
            Eigen::VectorXd measurement = makeMeasurement(predicted, matches, t);
            rejectYawOutliers(measurement, predicted, t);
            const Eigen::VectorXd previous = t.filter->get_nochange_X();
            t.filter->update(measurement);
            limitAngularUpdate(t.filter->get_X(), previous, t.dt, t);
            limitState(t.filter->get_X(), t);
            t.missed_frames = 0; t.stable_updates = std::min(t.stable_updates + 1, kStableUpdates);
            if (++t.lock_frames >= kLockFrames) t.target_locked = true;
        } else {
            ++t.missed_frames;
            t.stable_updates = 0;
            if (t.missed_frames > kUnlockLostFrames) {
                t.target_locked = false;
                t.lock_frames = 0;
            }
        }
    }
    if (!t.filter || t.missed_frames > kMaxMissedFrames) { reset(); return {}; }

    const auto& state = t.filter->get_nochange_X();
    const double lead = std::clamp(static_cast<double>(gimbal.prediction_bias_s) +
                                   processingDelay(timestamp_seconds), 0.0, 0.2);
    Eigen::VectorXd future = state;
    future[kCx] += lead * state[kVx]; future[kCy] += lead * state[kVy];
    future[kYaw] = wrapAngle(state[kYaw] + lead * state[kVyaw]);
    future[kZEven] += lead * state[kVz]; future[kZOdd] += lead * state[kVz];
    const auto target = slotPosition(future, 0);
    const Eigen::Vector3d velocity{state[kVx] + state[kR1] * std::sin(future[kYaw]) * state[kVyaw],
                                   state[kVy] - state[kR1] * std::cos(future[kYaw]) * state[kVyaw],
                                   state[kVz]};
    const double horizontal = std::hypot(target.x(), target.y()), distance = target.norm();
    PredictionResult result;
    result.state = t.stable_updates >= kStableUpdates && t.missed_frames == 0 ? TrackingState::Stable : TrackingState::Unstable;
    result.target_id = t.target_id;
    result.center_x_m = future[kCx]; result.center_y_m = future[kCy];
    result.center_velocity_x_m_s = state[kVx]; result.center_velocity_y_m_s = state[kVy];
    result.body_yaw = future[kYaw]; result.body_yaw_velocity = state[kVyaw];
    result.radius_1_m = state[kR1]; result.radius_2_m = state[kR2]; result.distance_m = distance;
    result.target_yaw = std::atan2(target.y(), target.x());
    result.target_pitch = std::atan2(target.z(), horizontal);
    if (horizontal > 1.0e-6) result.target_yaw_velocity =
        (target.x() * velocity.y() - target.y() * velocity.x()) / (horizontal * horizontal);
    if (distance > 1.0e-6) {
        const double radial = horizontal > 1.0e-6
            ? (target.x() * velocity.x() + target.y() * velocity.y()) / horizontal : 0.0;
        result.target_pitch_velocity = (horizontal * velocity.z() - target.z() * radial) /
                                       (distance * distance);
    }
    result.fire_allowed = result.state == TrackingState::Stable && t.target_id != 0;
    return result;
}
}
