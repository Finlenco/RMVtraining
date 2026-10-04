#include "EkfTracker.hpp"

#include <KalmanFilter.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {


// 中心x, vx, 中心y, vy, 装甲板z, vz, yaw, yaw速度, 半径
constexpr int kStateSize = 9;
constexpr int kCenterX = 0;
constexpr int kVelocityX = 1;
constexpr int kCenterY = 2;
constexpr int kVelocityY = 3;
constexpr int kArmorZ = 4;
constexpr int kVelocityZ = 5;
constexpr int kYaw = 6;
constexpr int kYawVelocity = 7;
constexpr int kRadius = 8;
constexpr int kMeasurementSize = 4;
constexpr int kMeasurementYaw = 3;

constexpr double kPi = 3.1415926;
constexpr double kInitialRadius = 0.25;
constexpr double kMinRadius = 0.12;
constexpr double kMaxRadius = 0.40;
constexpr double kMatchDistanceM = 0.15;
constexpr double kMatchYawDifference = 1.0;
constexpr double kYawMeasurementSigma = 0.08;
constexpr int kStableUpdates = 3;
constexpr int kMaxMissedFrames = 10;

struct FilterContext {
    double last_timestamp{0.0};
    double dt{0.02};
    int missed_frames{0};
    int stable_updates{0};
    std::uint8_t target_id{0};
    double measurement_sigma{0.03};
    std::unique_ptr<ExtendedKalmanFilter> filter;
};

FilterContext& context() {
    static FilterContext tracker;
    return tracker;
}

double wrapAngle(double angle) {
    return std::remainder(angle, 2.0 * kPi);
}

Eigen::Vector3d observationPosition(const ArmorPose& pose) {
    return {pose.position_gimbal_m[0], pose.position_gimbal_m[1], pose.position_gimbal_m[2]};
}

Eigen::VectorXd makeMeasurement(const ArmorPose& pose) {
    const Eigen::Vector3d position = observationPosition(pose);
    Eigen::VectorXd measurement(kMeasurementSize);
    measurement << position.x(), position.y(), position.z(), pose.armor_yaw;
    return measurement;
}

bool validObservation(const ArmorPose& pose) {
    const Eigen::Vector3d position = observationPosition(pose);
    if (!position.allFinite() || !std::isfinite(pose.armor_yaw) ||
        pose.detection.target_id <= 0 || pose.detection.target_id > 255 ||
        position.x() <= 0.0 || position.norm() < 0.05 || position.norm() > 30.0) {
        return false;
    }
    return pose.reprojection_error <= 0.0 || pose.reprojection_error < 20.0;
}

Eigen::Vector3d armorPositionFromState(const Eigen::VectorXd& state) {
    const double yaw = state[kYaw];
    const double radius = state[kRadius];
    return {state[kCenterX] - radius * std::cos(yaw),
            state[kCenterY] - radius * std::sin(yaw), state[kArmorZ]};
}

Eigen::Vector3d armorVelocityFromState(const Eigen::VectorXd& state) {
    const double yaw = state[kYaw];
    const double radius = state[kRadius];
    const double yaw_velocity = state[kYawVelocity];
    return {state[kVelocityX] + radius * std::sin(yaw) * yaw_velocity,
            state[kVelocityY] - radius * std::cos(yaw) * yaw_velocity,
            state[kVelocityZ]};
}


//过程噪声矩阵
void setConstantVelocityNoise(Eigen::MatrixXd& noise, int position_index,
                              int velocity_index, double variance, double dt) {
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    noise(position_index, position_index) = variance * dt4 * 0.25;
    noise(position_index, velocity_index) = variance * dt3 * 0.5;
    noise(velocity_index, position_index) = noise(position_index, velocity_index);
    noise(velocity_index, velocity_index) = variance * dt2;
}

void limitState(Eigen::VectorXd& state) {
    state[kYawVelocity] = std::clamp(state[kYawVelocity], -20.0, 20.0);
    state[kVelocityZ] = std::clamp(state[kVelocityZ], -5.0, 5.0);
    state[kArmorZ] = std::clamp(state[kArmorZ], -2.0, 2.0);
    state[kRadius] = std::clamp(state[kRadius], kMinRadius, kMaxRadius);
}


//主要
void createFilter(FilterContext& tracker, const ArmorPose& observation) {
    const Eigen::VectorXd measurement = makeMeasurement(observation);
    const double yaw = measurement[kMeasurementYaw];
//初始化状态变量
    Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(kStateSize);
    initial_state[kCenterX] = measurement[0] + kInitialRadius * std::cos(yaw);
    initial_state[kCenterY] = measurement[1] + kInitialRadius * std::sin(yaw);
    initial_state[kArmorZ] = measurement[2];
    initial_state[kYaw] = yaw;
    initial_state[kRadius] = kInitialRadius;

//初始协方差
    Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
    initial_covariance(kCenterX, kCenterX) = 0.25 * 0.25;
    initial_covariance(kVelocityX, kVelocityX) = 4.0;
    initial_covariance(kCenterY, kCenterY) = 0.25 * 0.25;
    initial_covariance(kVelocityY, kVelocityY) = 4.0;
    initial_covariance(kArmorZ, kArmorZ) = 0.10 * 0.10;
    initial_covariance(kVelocityZ, kVelocityZ) = 1.0;
    initial_covariance(kYaw, kYaw) = 0.5 * 0.5;
    initial_covariance(kYawVelocity, kYawVelocity) = 4.0;
    initial_covariance(kRadius, kRadius) = 0.08 * 0.08;

    auto process = [&tracker](const Eigen::VectorXd& state) {
        Eigen::VectorXd next = state;
        next[kCenterX] += tracker.dt * state[kVelocityX];
        next[kCenterY] += tracker.dt * state[kVelocityY];
        next[kArmorZ] += tracker.dt * state[kVelocityZ];
        next[kYaw] += tracker.dt * state[kYawVelocity];
        return next;
    };
    
//EKF状态转移
    auto processJacobian = [&tracker](const Eigen::VectorXd&) {
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(kStateSize, kStateSize);
        jacobian(kCenterX, kVelocityX) = tracker.dt;
        jacobian(kCenterY, kVelocityY) = tracker.dt;
        jacobian(kArmorZ, kVelocityZ) = tracker.dt;
        jacobian(kYaw, kYawVelocity) = tracker.dt;
        return jacobian;
    };

    auto processNoise = [&tracker]() {
        Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(kStateSize, kStateSize);
        //假设加速度都具有不确定性
        constexpr double kLinearAccelerationVariance = 2.0;
        constexpr double kAngularAccelerationVariance = 4.0;
        setConstantVelocityNoise(noise, kCenterX, kVelocityX, kLinearAccelerationVariance, tracker.dt);
        setConstantVelocityNoise(noise, kCenterY, kVelocityY, kLinearAccelerationVariance, tracker.dt);
        setConstantVelocityNoise(noise, kArmorZ, kVelocityZ, kLinearAccelerationVariance, tracker.dt);
        setConstantVelocityNoise(noise, kYaw, kYawVelocity, kAngularAccelerationVariance, tracker.dt);
        
        
        noise(kRadius, kRadius) = 1.0e-4 * tracker.dt;//缓慢变化
        return noise;
    };

    auto observationModel = [](const Eigen::VectorXd& state) {
        const Eigen::Vector3d position = armorPositionFromState(state);
        Eigen::VectorXd expected(kMeasurementSize);

        expected << position.x(), position.y(), position.z(), state[kYaw];
        
        return expected;
    };

    auto observationJacobian = [](const Eigen::VectorXd& state) {
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(kMeasurementSize, kStateSize);
        const double yaw = state[kYaw];
        const double radius = state[kRadius];

        jacobian(0, kCenterX) = 1.0;
        jacobian(0, kYaw) = radius * std::sin(yaw);
        jacobian(0, kRadius) = -std::cos(yaw);
        jacobian(1, kCenterY) = 1.0;
        jacobian(1, kYaw) = -radius * std::cos(yaw);
        jacobian(1, kRadius) = -std::sin(yaw);
        jacobian(2, kArmorZ) = 1.0;
        jacobian(3, kYaw) = 1.0;
        return jacobian;
    };

    auto measurementNoise = [&tracker](const Eigen::VectorXd&) {
        Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(kMeasurementSize, kMeasurementSize);
        const double position_variance = tracker.measurement_sigma * tracker.measurement_sigma;
        noise(0, 0) = position_variance;
        noise(1, 1) = position_variance;
        noise(2, 2) = position_variance;
        noise(3, 3) = kYawMeasurementSigma * kYawMeasurementSigma;
        return noise;
    };
//残差角度归一化
    auto normalizeResidual = [](const Eigen::VectorXd& residual) {
        Eigen::VectorXd normalized = residual;
        normalized[kMeasurementYaw] = wrapAngle(normalized[kMeasurementYaw]);
        return normalized;
    };

    tracker.filter = std::make_unique<ExtendedKalmanFilter>(
        process, observationModel, processJacobian, observationJacobian,
        processNoise, measurementNoise, normalizeResidual, initial_covariance,
        initial_state);//把刚才所有计算结果交给EKF
}

struct Association {
    const ArmorPose* observation{nullptr};
    double position_difference{0.0};
    double yaw_difference{0.0};
    int same_id_count{0};
};

Association findObservation(const FilterContext& tracker,
                            const std::vector<ArmorPose>& observations) {
    Association best;
    if (!tracker.filter) {
        for (const ArmorPose& observation : observations) {
            if (validObservation(observation)) {
                best.observation = &observation;
                return best;
            }
        }
        return best;
    }

    const Eigen::VectorXd& predicted_state = tracker.filter->get_nochange_X();
    const Eigen::Vector3d predicted_position = armorPositionFromState(predicted_state);
    
    best.position_difference = std::numeric_limits<double>::infinity();

    for (const ArmorPose& observation : observations) {
        if (!validObservation(observation) ||
            observation.detection.target_id != tracker.target_id) {
            continue;
        }
        ++best.same_id_count;
        const double position_difference =
            (observationPosition(observation) - predicted_position).norm();
        if (position_difference < best.position_difference) {
            best.observation = &observation;
            best.position_difference = position_difference;
            best.yaw_difference = std::abs(wrapAngle(observation.armor_yaw - predicted_state[kYaw]));
        }//优先距离最近的装甲板
    }
    return best;
}

void updateMeasurementNoise(FilterContext& tracker, const ArmorPose& observation) {
    tracker.measurement_sigma = 0.02;
    if (observation.reprojection_error > 0.0) {
        tracker.measurement_sigma = std::clamp(0.015 + observation.reprojection_error * 0.002, 0.015, 0.10);
    }
}//限制1.5-10


//跳变处理
void handleArmorJump(FilterContext& tracker, const ArmorPose& observation) {
    Eigen::VectorXd state = tracker.filter->get_X();
    const Eigen::VectorXd measurement = makeMeasurement(observation);
    const double yaw = state[kYaw] + wrapAngle(measurement[kMeasurementYaw] - state[kYaw]);
    const Eigen::Vector3d position = observationPosition(observation);

    state[kYaw] = yaw;
    state[kArmorZ] = position.z();
    const Eigen::Vector3d inferred_position = armorPositionFromState(state);
    if ((inferred_position - position).norm() > kMatchDistanceM) {
        state[kCenterX] = position.x() + state[kRadius] * std::cos(yaw);
        state[kVelocityX] = 0.0;
        state[kCenterY] = position.y() + state[kRadius] * std::sin(yaw);
        state[kVelocityY] = 0.0;
        state[kVelocityZ] = 0.0;
    }

    tracker.filter->setState(state);
    tracker.filter->usePostAsPri();
    tracker.stable_updates = 0;
    tracker.missed_frames = 0;
}

double processingDelay(double frame_timestamp) {
    const double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const double delay = now - frame_timestamp;
    return std::isfinite(delay) && delay >= 0.0 && delay < 0.5 ? delay : 0.0;
}

}  

namespace ekf_tracker {

void reset() {
    context() = FilterContext{};
}

PredictionResult update(const std::vector<ArmorPose>& observations,const GimbalState& gimbal,  double timestamp_seconds) {
    
    FilterContext& tracker = context();
    if (!std::isfinite(timestamp_seconds))     return {};
    

    if (tracker.filter) {
        const double elapsed = timestamp_seconds - tracker.last_timestamp;
        if (elapsed <= 0.0 || elapsed > 0.5) {
            reset();
        } else {
            tracker.dt = std::clamp(elapsed, 0.001, 0.1);
            tracker.filter->predict();
        }
    }
    tracker.last_timestamp = timestamp_seconds;

    const Association association = findObservation(tracker, observations);
    bool matched = false;

    if (association.observation != nullptr) {
        if (!tracker.filter) {
            updateMeasurementNoise(tracker, *association.observation);
            createFilter(tracker, *association.observation);
            tracker.target_id = static_cast<std::uint8_t>(
                association.observation->detection.target_id);
            tracker.stable_updates = 1;
            matched = true;
        } else if (association.position_difference < kMatchDistanceM &&
                   association.yaw_difference < kMatchYawDifference) {
            updateMeasurementNoise(tracker, *association.observation);
            tracker.filter->update(makeMeasurement(*association.observation));
            limitState(tracker.filter->get_X());
            tracker.stable_updates = std::min(tracker.stable_updates + 1,  kStableUpdates);
            matched = true;//稳定更新
        } else if (association.same_id_count == 1 &&   association.yaw_difference >= kMatchYawDifference) {
            handleArmorJump(tracker, *association.observation); //发生跳变
        }
    }

    //异常处理
    if (matched) {
        tracker.missed_frames = 0;
    } else if (tracker.filter &&
               !(association.observation != nullptr &&
                 association.same_id_count == 1 &&
                 association.yaw_difference >= kMatchYawDifference)) {
        ++tracker.missed_frames;
    }

    if (!tracker.filter) {
        return {};
    }
    if (tracker.missed_frames > kMaxMissedFrames) {
        reset();
        return {};
    }

    const Eigen::VectorXd& state = tracker.filter->get_nochange_X();
    const double lead_time = std::clamp(static_cast<double>(gimbal.prediction_bias_s) +  processingDelay(timestamp_seconds), 0.0, 0.2);
    
    Eigen::VectorXd predicted_state = state;

    predicted_state[kCenterX] += lead_time * state[kVelocityX];
    predicted_state[kCenterY] += lead_time * state[kVelocityY];
    predicted_state[kArmorZ] += lead_time * state[kVelocityZ];
    predicted_state[kYaw] += lead_time * state[kYawVelocity];

    const Eigen::Vector3d target = armorPositionFromState(predicted_state);
    const Eigen::Vector3d target_velocity = armorVelocityFromState(predicted_state);
    const double horizontal_distance = std::hypot(target.x(), target.y());
    const double distance = target.norm();

    PredictionResult result;
    result.state = tracker.stable_updates >= kStableUpdates &&
                           tracker.missed_frames == 0
                       ? TrackingState::Stable
                       : TrackingState::Unstable;
    
    result.target_id = tracker.target_id;
    result.center_x_m = static_cast<float>(predicted_state[kCenterX]);
    result.center_y_m = static_cast<float>(predicted_state[kCenterY]);
    result.center_velocity_x_m_s = static_cast<float>(state[kVelocityX]);
    result.center_velocity_y_m_s = static_cast<float>(state[kVelocityY]);
    result.body_yaw = static_cast<float>(wrapAngle(predicted_state[kYaw]));
    result.body_yaw_velocity = static_cast<float>(state[kYawVelocity]);
    result.radius_1_m = static_cast<float>(state[kRadius]);
    result.radius_2_m = static_cast<float>(state[kRadius]);
    result.distance_m = static_cast<float>(distance);
    result.target_yaw = static_cast<float>(gimbal.yaw + std::atan2(target.y(), target.x()));
    result.target_pitch = static_cast<float>(gimbal.pitch + std::atan2(target.z(), horizontal_distance));

    if (horizontal_distance > 1.0e-6) {
        result.target_yaw_velocity = static_cast<float>(
            (target.x() * target_velocity.y() - target.y() * target_velocity.x()) /
            (horizontal_distance * horizontal_distance));
    }

    if (distance > 1.0e-6) {

        double radial_velocity = 0.0 ;
        if(horizontal_distance > 1.0e-6){
            radial_velocity = (target.x() * target_velocity.x() + target.y() * target_velocity.y()) / horizontal_distance;
        }

        result.target_pitch_velocity = static_cast<float>(
            (horizontal_distance * target_velocity.z() - target.z() * radial_velocity) / (distance * distance));
    }

    result.fire_allowed = result.state == TrackingState::Stable &&  tracker.target_id != 0;
    return result;
}

}  