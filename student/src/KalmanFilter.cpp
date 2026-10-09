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

// 车体模型：两块对置装甲板共享高度和半径。
// [xc, vxc, yc, vyc, z_even, z_odd, r_even, r_odd, yaw, vyaw, vz]
constexpr int kStateSize = 11;
constexpr int kX = 0;
constexpr int kVx = 1;
constexpr int kY = 2;
constexpr int kVy = 3;
constexpr int kZEven = 4;
constexpr int kZOdd = 5;
constexpr int kREven = 6;
constexpr int kROdd = 7;
constexpr int kYaw = 8;
constexpr int kYawRate = 9;
constexpr int kVz = 10;

constexpr int kSlotCount = 4;
constexpr int kSlotMeasurementSize = 4;  // x, y, z, armor yaw
constexpr int kMeasurementSize = kSlotCount * kSlotMeasurementSize;
constexpr double kPi = 3.1415926;

// 阈值
constexpr double kInitialRadius = 0.25;
constexpr double kMinRadius = 0.12;
constexpr double kMaxRadius = 0.40;
constexpr double kMatchDistance = 0.20;
constexpr double kMatchYawDifference = 1.0;
constexpr double kIdMismatchDistance = 0.14;
constexpr double kIdMismatchYawDifference = 0.60;

//检测若干帧进入 Tracking，短时丢失进入 TempLost
constexpr int kTrackingFrames = 5;
constexpr int kTempLostFrames = 5;
constexpr int kStablePredictionFrames = 3;
constexpr int kResetMissedFrames = 50;
// 目标装甲板短时不可见时继续沿用原槽位，避免输出瞬间跳到另一块板。
constexpr int kTargetSlotSwitchFrames = kTempLostFrames;

// 高速旋转参数。Q 的量纲是米/弧度状态对应的加速度方差
constexpr double kBaseYawSigma = 0.12;
constexpr double kYawResidualGate = 0.75;
constexpr double kMaxYawRate = 15.0;
constexpr double kMaxYawAcceleration = 60.0;
constexpr double kMissingMeasurementNoiseScale = 100.0;
constexpr double kMaxYawCorrection = 0.60;
constexpr double kPositionProcessNoise = 200.0;
constexpr double kYawProcessNoise = 1000.0;

enum class TrackState { Lost, Detecting, Tracking, TempLost };

struct TrackerContext {
    double last_timestamp{0.0};
    double dt{0.02};
    int missed_frames{0};
    int detect_frames{0};
    int lost_frames{0};
    TrackState state{TrackState::Lost};
    std::uint8_t target_id{0};
    int target_slot{0};
    int target_slot_missed{0};
    double lead_time_s{0.0};
    bool lead_time_initialized{false};
    std::array<bool, kSlotCount> observed{};
    std::array<double, kSlotCount> position_sigma{0.02, 0.02, 0.02, 0.02};
    std::array<double, kSlotCount> yaw_sigma{
        kBaseYawSigma, kBaseYawSigma, kBaseYawSigma, kBaseYawSigma};
    std::array<bool, kSlotCount> yaw_rejected{};
    std::unique_ptr<ExtendedKalmanFilter> filter;
};

TrackerContext& tracker() {
    static TrackerContext context;
    return context;
}

double wrapAngle(double angle) {
    return std::remainder(angle, 2.0 * kPi);
}

Eigen::Vector3d positionOf(const ArmorPose& pose) {
    return {pose.position_gimbal_m[0], pose.position_gimbal_m[1], pose.position_gimbal_m[2]};
}

bool validObservation(const ArmorPose& pose) {
    const Eigen::Vector3d position = positionOf(pose);
    return position.allFinite() && std::isfinite(pose.armor_yaw) &&
           pose.detection.target_id > 0 && pose.detection.target_id <= 255 &&
           position.x() > 0.0 && position.norm() >= 0.05 && position.norm() <= 30.0 &&
           (pose.reprojection_error <= 0.0 || pose.reprojection_error < 20.0);
}

int heightIndex(int slot) {
    return slot % 2 == 0 ? kZEven : kZOdd;
}

int radiusIndex(int slot) {
    return slot % 2 == 0 ? kREven : kROdd;
}

double radiusOf(const Eigen::VectorXd& state, int slot) {
    return state[radiusIndex(slot)];
}

Eigen::Vector3d predictedArmorPosition(const Eigen::VectorXd& state, int slot) {
    const double armor_yaw = state[kYaw] + slot * kPi / 2.0;
    const double radius = radiusOf(state, slot);
    return {state[kX] - radius * std::cos(armor_yaw),
            state[kY] - radius * std::sin(armor_yaw),
            state[heightIndex(slot)]};
}

Eigen::VectorXd observationModel(const Eigen::VectorXd& state) {
    Eigen::VectorXd expected(kMeasurementSize);
    for (int slot = 0; slot < kSlotCount; ++slot) {
        const int offset = slot * kSlotMeasurementSize;
        expected.segment<3>(offset) = predictedArmorPosition(state, slot);
        expected[offset + 3] = wrapAngle(state[kYaw] + slot * kPi / 2.0);
    }
    return expected;
}

Eigen::MatrixXd observationJacobian(const Eigen::VectorXd& state) {
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(kMeasurementSize, kStateSize);
    for (int slot = 0; slot < kSlotCount; ++slot) {
        const int offset = slot * kSlotMeasurementSize;
        const int radius_index = radiusIndex(slot);
        const double angle = state[kYaw] + slot * kPi / 2.0;
        const double radius = state[radius_index];

        jacobian(offset, kX) = 1.0;
        jacobian(offset + 1, kY) = 1.0;
        jacobian(offset + 2, heightIndex(slot)) = 1.0;
        jacobian(offset, kYaw) = radius * std::sin(angle);
        jacobian(offset + 1, kYaw) = -radius * std::cos(angle);
        jacobian(offset, radius_index) = -std::cos(angle);
        jacobian(offset + 1, radius_index) = -std::sin(angle);
        jacobian(offset + 3, kYaw) = 1.0;
    }
    return jacobian;
}

void setConstantVelocityNoise(Eigen::MatrixXd& noise, int position_index,
                              int velocity_index, double variance, double dt) {
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    noise(position_index, position_index) = variance * dt4 / 4.0;
    noise(position_index, velocity_index) = variance * dt3 / 2.0;
    noise(velocity_index, position_index) = noise(position_index, velocity_index);
    noise(velocity_index, velocity_index) = variance * dt2;
}

void limitState(Eigen::VectorXd& state) {
    state[kYaw] = wrapAngle(state[kYaw]);
    state[kYawRate] = std::clamp(state[kYawRate], -kMaxYawRate, kMaxYawRate);
    state[kVz] = std::clamp(state[kVz], -5.0, 5.0);
    state[kREven] = std::clamp(state[kREven], kMinRadius, kMaxRadius);
    state[kROdd] = std::clamp(state[kROdd], kMinRadius, kMaxRadius);
}

void updateObservationNoise(TrackerContext& context, const ArmorPose& pose, int slot) {
    if (pose.reprojection_error > 0.0) {
        context.position_sigma[slot] =
            std::clamp(0.025 + pose.reprojection_error * 0.0025, 0.025, 0.15);
        context.yaw_sigma[slot] =
            std::clamp(0.15 + pose.reprojection_error * 0.015, 0.15, 0.40);
    } else {
        context.position_sigma[slot] = 0.03;
        context.yaw_sigma[slot] = 0.15;
    }
}

void createFilter(TrackerContext& context, const ArmorPose& pose) {
    const Eigen::Vector3d position = positionOf(pose);
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(kStateSize);
    initial_state[kX] = position.x() + kInitialRadius * std::cos(pose.armor_yaw);
    initial_state[kY] = position.y() + kInitialRadius * std::sin(pose.armor_yaw);
    initial_state[kZEven] = position.z();
    initial_state[kZOdd] = position.z();
    initial_state[kREven] = kInitialRadius;
    initial_state[kROdd] = kInitialRadius;
    initial_state[kYaw] = pose.armor_yaw;

    Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
    initial_covariance(kX, kX) = 0.25 * 0.25;
    initial_covariance(kY, kY) = 0.25 * 0.25;
    initial_covariance(kVx, kVx) = 1.0;
    initial_covariance(kVy, kVy) = 1.0;
    initial_covariance(kZEven, kZEven) = 0.10 * 0.10;
    initial_covariance(kZOdd, kZOdd) = 0.10 * 0.10;
    initial_covariance(kYaw, kYaw) = 0.5 * 0.5;
    initial_covariance(kYawRate, kYawRate) = 1.0;
    initial_covariance(kREven, kREven) = 0.08 * 0.08;
    initial_covariance(kROdd, kROdd) = 0.08 * 0.08;

    auto process = [&context](const Eigen::VectorXd& state) {
        Eigen::VectorXd next = state;
        next[kX] += context.dt * state[kVx];
        next[kY] += context.dt * state[kVy];
        next[kZEven] += context.dt * state[kVz];
        next[kZOdd] += context.dt * state[kVz];
        next[kYaw] = wrapAngle(state[kYaw] + context.dt * state[kYawRate]);
        return next;
    };

    auto processJacobian = [&context](const Eigen::VectorXd&) {
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
        jacobian(kX, kVx) = context.dt;
        jacobian(kY, kVy) = context.dt;
        jacobian(kZEven, kVz) = context.dt;
        jacobian(kZOdd, kVz) = context.dt;
        jacobian(kYaw, kYawRate) = context.dt;
        return jacobian;
    };

    auto processNoise = [&context]() {
        Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(kStateSize, kStateSize);
        setConstantVelocityNoise(noise, kX, kVx, kPositionProcessNoise, context.dt);
        setConstantVelocityNoise(noise, kY, kVy, kPositionProcessNoise, context.dt);
        setConstantVelocityNoise(noise, kYaw, kYawRate, kYawProcessNoise, context.dt);
        noise(kZEven, kZEven) = 0.01 * context.dt;
        noise(kZOdd, kZOdd) = 0.01 * context.dt;
        noise(kVz, kVz) = 0.05 * context.dt;
        noise(kREven, kREven) = 1.0e-4 * context.dt;
        noise(kROdd, kROdd) = 1.0e-4 * context.dt;
        return noise;
    };

    auto measurementNoise = [&context](const Eigen::VectorXd&) {
        Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(kMeasurementSize, kMeasurementSize);
        for (int slot = 0; slot < kSlotCount; ++slot) {
            const int offset = slot * kSlotMeasurementSize;
            const double scale = context.observed[slot] ? 1.0 : kMissingMeasurementNoiseScale;
            const double position_variance = context.position_sigma[slot] *
                                              context.position_sigma[slot] * scale;
            noise(offset, offset) = position_variance;
            noise(offset + 1, offset + 1) = position_variance;
            noise(offset + 2, offset + 2) = position_variance;
            const double yaw_sigma = context.yaw_rejected[slot]
                ? 1.0 : context.yaw_sigma[slot];
            noise(offset + 3, offset + 3) = yaw_sigma * yaw_sigma * scale;
        }
        return noise;
    };

    auto normalizeResidual = [](const Eigen::VectorXd& residual) {
        Eigen::VectorXd normalized = residual;
        for (int slot = 0; slot < kSlotCount; ++slot) {
            normalized[slot * kSlotMeasurementSize + 3] =
                wrapAngle(normalized[slot * kSlotMeasurementSize + 3]);
        }
        return normalized;
    };

    context.filter = std::make_unique<ExtendedKalmanFilter>(
        process, observationModel, processJacobian, observationJacobian,
        processNoise, measurementNoise, normalizeResidual,
        initial_covariance, initial_state);
}

struct Match {
    const ArmorPose* pose{nullptr};
    double position_error{std::numeric_limits<double>::infinity()};
    double yaw_error{std::numeric_limits<double>::infinity()};
    double cost{std::numeric_limits<double>::infinity()};
    bool same_id{false};
};

struct Assignment {
    std::array<Match, kSlotCount> matches{};
    int count{0};
    int same_id_count{0};
    double cost{std::numeric_limits<double>::infinity()};
};

bool betterAssignment(const Assignment& candidate, const Assignment& best) {
    if (candidate.count != best.count) return candidate.count > best.count;
    if (candidate.same_id_count != best.same_id_count) {
        return candidate.same_id_count > best.same_id_count;
    }
    return candidate.cost < best.cost;
}

void searchAssignments(const TrackerContext& context,
                       const std::vector<ArmorPose>& observations,
                       const Eigen::VectorXd& state, int slot, std::uint64_t used,
                       Assignment current, Assignment& best) {
    if (slot == kSlotCount) {
        if (betterAssignment(current, best)) best = current;
        return;
    }

    // 允许当前槽位缺失，缺失观测会在 EKF 更新前用预测值填充。
    searchAssignments(context, observations, state, slot + 1, used, current, best);

    for (std::size_t index = 0; index < observations.size(); ++index) {
        if (index >= 64 || (used & (std::uint64_t{1} << index)) != 0 ||
            !validObservation(observations[index])) {
            continue;
        }
        const ArmorPose& pose = observations[index];
        const bool same_id = pose.detection.target_id == context.target_id;
        const double rotational_speed =
            std::abs(state[kYawRate]) * radiusOf(state, slot);
        const double center_speed = std::hypot(state[kVx], state[kVy]);
        const double predicted_motion =
            (center_speed + rotational_speed) * std::max(context.dt, 0.001);
        const double max_distance = std::clamp(
            kMatchDistance + 1.5 * predicted_motion, 0.20, 0.45);
        const double max_yaw = std::clamp(
            kMatchYawDifference + 1.5 * std::abs(state[kYawRate]) *
                std::max(context.dt, 0.001),
            1.0, 1.35);
        const Eigen::Vector3d delta = positionOf(pose) - predictedArmorPosition(state, slot);
        const double position_error = delta.norm();
        const double yaw_error = std::abs(wrapAngle(
            pose.armor_yaw - (state[kYaw] + slot * kPi / 2.0)));
        // 分类器偶发改号时，只在几何上仍高度一致的情况下沿用当前轨迹。

        const double accepted_distance = same_id
            ? max_distance
            : std::min(max_distance, kIdMismatchDistance + 0.75 * predicted_motion);
        const double accepted_yaw = same_id
            ? max_yaw
            : std::min(max_yaw, kIdMismatchYawDifference +
                                  0.75 * std::abs(state[kYawRate]) *
                                      std::max(context.dt, 0.001));
        if (position_error >= accepted_distance || yaw_error >= accepted_yaw) continue;

        Match match;
        match.pose = &pose;
        match.position_error = position_error;
        match.yaw_error = yaw_error;
        match.same_id = same_id;
        match.cost = position_error / max_distance + yaw_error / max_yaw +
                     (same_id ? 0.0 : 0.5);

        Assignment next = current;
        next.matches[slot] = match;
        ++next.count;
        next.same_id_count += same_id ? 1 : 0;
        if (!std::isfinite(next.cost)) next.cost = 0.0;
        next.cost += match.cost;
        searchAssignments(context, observations, state, slot + 1,
                          used | (std::uint64_t{1} << index), next, best);
    }
}

void updateTargetSlot(TrackerContext& context, const Assignment& assignment) {
    if (context.target_slot < 0 || context.target_slot >= kSlotCount) {
        context.target_slot = 0;
        context.target_slot_missed = 0;
    }

    // 当前装甲板仍被观测到时保持槽位，避免四块板之间来回切换。
    if (assignment.matches[context.target_slot].pose != nullptr) {
        context.target_slot_missed = 0;
        return;
    }

    ++context.target_slot_missed;
    if (context.target_slot_missed < kTargetSlotSwitchFrames) return;

    int candidate_slot = -1;
    double candidate_cost = std::numeric_limits<double>::infinity();
    for (int slot = 0; slot < kSlotCount; ++slot) {
        const Match& match = assignment.matches[slot];
        if (match.pose != nullptr && match.cost < candidate_cost) {
            candidate_slot = slot;
            candidate_cost = match.cost;
        }
    }
    if (candidate_slot >= 0) {
        context.target_slot = candidate_slot;
        context.target_slot_missed = 0;
    }
}

Assignment associate(const TrackerContext& context,
                     const std::vector<ArmorPose>& observations) {
    Assignment best;
    best.count = -1;
    if (!context.filter) return best;
    searchAssignments(context, observations, context.filter->get_nochange_X(),
                      0, 0, Assignment{}, best);
    return best;
}

const ArmorPose* chooseInitialObservation(const std::vector<ArmorPose>& observations) {
    const ArmorPose* selected = nullptr;
    double best_error = std::numeric_limits<double>::infinity();
    for (const ArmorPose& pose : observations) {
        if (!validObservation(pose)) continue;
        const double error = pose.reprojection_error > 0.0
            ? pose.reprojection_error : 0.0;
        if (selected == nullptr || error < best_error) {
            selected = &pose;
            best_error = error;
        }
    }
    return selected;
}

Eigen::VectorXd makeMeasurement(const Eigen::VectorXd& predicted,
                                const Assignment& assignment,
                                TrackerContext& context) {
    Eigen::VectorXd measurement = predicted;
    context.observed.fill(false);
    context.yaw_rejected.fill(false);
    for (int slot = 0; slot < kSlotCount; ++slot) {
        const Match& match = assignment.matches[slot];
        if (match.pose == nullptr) continue;
        context.observed[slot] = true;
        updateObservationNoise(context, *match.pose, slot);
        const int offset = slot * kSlotMeasurementSize;
        const Eigen::Vector3d position = positionOf(*match.pose);
        measurement.segment<3>(offset) = position;
        measurement[offset + 3] = match.pose->armor_yaw;
        if (match.yaw_error > kYawResidualGate) {
            measurement[offset + 3] = predicted[offset + 3];
            context.yaw_rejected[slot] = true;
        }
    }
    return measurement;
}

void limitAngularUpdate(Eigen::VectorXd& state, const Eigen::VectorXd& previous,
                        double dt) {
    // 位置观测也会通过雅可比修正 yaw。限制单帧校正量，避免一帧错误
    // PnP 把输出从当前装甲板拉到另一块板；正常旋转由过程模型先行推进。
    const double max_yaw_correction = std::clamp(
        kMaxYawRate * std::max(dt, 0.001) + 0.08, 0.10, kMaxYawCorrection);
    const double yaw_correction = wrapAngle(state[kYaw] - previous[kYaw]);
    state[kYaw] = wrapAngle(previous[kYaw] +
                            std::clamp(yaw_correction, -max_yaw_correction,
                                       max_yaw_correction));

    const double max_yaw_rate_change = kMaxYawAcceleration * std::max(dt, 0.001);
    state[kYawRate] = previous[kYawRate] + std::clamp(
        state[kYawRate] - previous[kYawRate],
        -max_yaw_rate_change, max_yaw_rate_change);
    state[kYawRate] = std::clamp(state[kYawRate], -kMaxYawRate, kMaxYawRate);
    state[kYaw] = wrapAngle(state[kYaw]);
}

bool finiteState(const Eigen::VectorXd& state) {
    return state.size() == kStateSize && state.allFinite();
}

bool finiteFilter(const ExtendedKalmanFilter& filter, bool posterior) {
    const Eigen::MatrixXd& covariance = posterior
        ? filter.get_P_post() : filter.get_P_pri();
    return finiteState(filter.get_nochange_X()) && covariance.allFinite();
}

void updateTrackState(TrackerContext& context, bool matched) {
    if (matched) {
        context.missed_frames = 0;
        context.lost_frames = 0;
        if (context.state == TrackState::Lost) {
            context.state = TrackState::Detecting;
            context.detect_frames = 1;
        } else if (context.state == TrackState::Detecting) {
            ++context.detect_frames;
            if (context.detect_frames >= kTrackingFrames) {
                context.state = TrackState::Tracking;
            }
        } else {
            context.state = TrackState::Tracking;
        }
        return;
    }

    ++context.missed_frames;
    if (context.state == TrackState::Detecting) {
        context.state = TrackState::Lost;
        context.detect_frames = 0;
    } else if (context.state == TrackState::Tracking) {
        ++context.lost_frames;
        // 短时漏检仍保持 Stable 输出预测值；超过这个窗口才报告 Unstable。
        if (context.lost_frames > kStablePredictionFrames) {
            context.state = TrackState::TempLost;
        }
    } else if (context.state == TrackState::TempLost) {
        ++context.lost_frames;
        if (context.lost_frames > kTempLostFrames) {
            context.state = TrackState::Lost;
            context.detect_frames = 0;
        }
    }
}

double processingDelay(double timestamp) {
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const double delay = now - timestamp;
    return std::isfinite(delay) && delay >= 0.0 && delay < 0.5 ? delay : 0.0;
}

double updateLeadTime(TrackerContext& context, const GimbalState& gimbal,
                      double timestamp_seconds) {
    const double prediction_bias = std::isfinite(gimbal.prediction_bias_s)
        ? static_cast<double>(gimbal.prediction_bias_s) : 0.0;
    const double requested = std::clamp(
        prediction_bias + processingDelay(timestamp_seconds), 0.0, 0.2);
    if (!context.lead_time_initialized) {
        context.lead_time_s = requested;
        context.lead_time_initialized = true;
        return context.lead_time_s;
    }

    // 处理耗时和串口 bias 的单帧变化不应直接变成目标角度跳变。
    const double max_step = std::clamp(0.5 * std::max(context.dt, 0.001),
                                       0.002, 0.02);
    context.lead_time_s += std::clamp(requested - context.lead_time_s,
                                     -max_step, max_step);
    return context.lead_time_s;
}

}  // namespace

namespace ekf_tracker {

void reset() {
    tracker() = TrackerContext{};
}

PredictionResult update(const std::vector<ArmorPose>& observations,
                        const GimbalState& gimbal, double timestamp_seconds) {
    TrackerContext& context = tracker();
    if (!std::isfinite(timestamp_seconds)) return {};

    if (context.filter) {
        const double elapsed = timestamp_seconds - context.last_timestamp;
        if (elapsed <= 0.0 || elapsed > 0.5) {
            reset();
        } else {
            context.dt = std::clamp(elapsed, 0.001, 0.1);
            context.filter->predict();
            if (!finiteFilter(*context.filter, false)) reset();
        }
    }
    context.last_timestamp = timestamp_seconds;

    if (!context.filter) {
        const ArmorPose* initial = chooseInitialObservation(observations);
        if (initial == nullptr) return {};
        updateObservationNoise(context, *initial, 0);
        createFilter(context, *initial);
        context.target_id = static_cast<std::uint8_t>(initial->detection.target_id);
        context.state = TrackState::Detecting;
        context.detect_frames = 1;
    } else {
        const Assignment assignment = associate(context, observations);
        if (assignment.count > 0) {
            const Eigen::VectorXd predicted = observationModel(context.filter->get_nochange_X());
            Eigen::VectorXd measurement = makeMeasurement(predicted, assignment, context);
            const Eigen::VectorXd previous = context.filter->get_nochange_X();
            context.filter->update(measurement);
            limitAngularUpdate(context.filter->get_X(), previous, context.dt);
            limitState(context.filter->get_X());
            if (!finiteFilter(*context.filter, true)) {
                reset();
                return {};
            }
            updateTargetSlot(context, assignment);
            updateTrackState(context, true);
        } else {
            context.observed.fill(false);
            context.yaw_rejected.fill(false);
            updateTargetSlot(context, Assignment{});
            updateTrackState(context, false);
        }
    }

    if (!context.filter || context.missed_frames > kResetMissedFrames) {
        reset();
        return {};
    }

    const Eigen::VectorXd& state = context.filter->get_nochange_X();
    const double lead_time = updateLeadTime(context, gimbal, timestamp_seconds);
    Eigen::VectorXd future = state;
    future[kX] += lead_time * state[kVx];
    future[kY] += lead_time * state[kVy];
    future[kZEven] += lead_time * state[kVz];
    future[kZOdd] += lead_time * state[kVz];
    future[kYaw] = wrapAngle(state[kYaw] + lead_time * state[kYawRate]);

    const int target_slot = std::clamp(context.target_slot, 0, kSlotCount - 1);
    const Eigen::Vector3d target = predictedArmorPosition(future, target_slot);
    const double target_radius = radiusOf(state, target_slot);
    const double target_angle = future[kYaw] + target_slot * kPi / 2.0;
    const Eigen::Vector3d target_velocity{
        state[kVx] + target_radius * std::sin(target_angle) * state[kYawRate],
        state[kVy] - target_radius * std::cos(target_angle) * state[kYawRate],
        state[kVz]};
    const double horizontal_distance = std::hypot(target.x(), target.y());
    const double distance = target.norm();

    PredictionResult result;
    result.state = context.state == TrackState::Tracking
                       ? TrackingState::Stable : TrackingState::Unstable;
    result.target_id = context.target_id;
    result.center_x_m = static_cast<float>(future[kX]);
    result.center_y_m = static_cast<float>(future[kY]);
    result.center_velocity_x_m_s = static_cast<float>(state[kVx]);
    result.center_velocity_y_m_s = static_cast<float>(state[kVy]);
    result.body_yaw = static_cast<float>(future[kYaw]);
    result.body_yaw_velocity = static_cast<float>(state[kYawRate]);
    result.radius_1_m = static_cast<float>(state[kREven]);
    result.radius_2_m = static_cast<float>(state[kROdd]);
    result.distance_m = static_cast<float>(distance);
    result.target_yaw = static_cast<float>(std::atan2(target.y(), target.x()));
    result.target_pitch = static_cast<float>(
        std::atan2(target.z(), horizontal_distance));

    if (horizontal_distance > 1.0e-6) {
        result.target_yaw_velocity = static_cast<float>(
            (target.x() * target_velocity.y() - target.y() * target_velocity.x()) /
            (horizontal_distance * horizontal_distance));
    }
    if (distance > 1.0e-6) {
        const double radial_velocity = horizontal_distance > 1.0e-6
            ? (target.x() * target_velocity.x() + target.y() * target_velocity.y()) /
              horizontal_distance : 0.0;
        result.target_pitch_velocity = static_cast<float>(
            (horizontal_distance * target_velocity.z() - target.z() * radial_velocity) /
            (distance * distance));
    }
    // 纯预测期间可以继续提供平滑瞄准角，但没有新观测时禁止开火。
    result.fire_allowed = result.state == TrackingState::Stable &&
                          context.target_id != 0 && context.missed_frames == 0;
    return result;
}

}  
