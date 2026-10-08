#include "EkfTracker.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadiusM = 0.25;
constexpr double kCenterX = 2.0;
constexpr double kCenterY = 0.1;
constexpr double kCenterZ = 0.35;

double wrapAngle(double angle) {
    return std::remainder(angle, 2.0 * kPi);
}

ArmorPose makeObservation(double body_yaw, int target_id = 7) {
    ArmorPose pose;
    pose.detection.target_id = target_id;
    pose.detection.confidence = 0.95F;
    pose.position_gimbal_m = cv::Vec3d(
        kCenterX - kRadiusM * std::cos(body_yaw),
        kCenterY - kRadiusM * std::sin(body_yaw), kCenterZ);
    pose.position_camera_m = pose.position_gimbal_m;
    pose.armor_yaw = wrapAngle(body_yaw);
    pose.reprojection_error = 0.4;
    return pose;
}

GimbalState neutralGimbal() {
    return {};
}

void warmUp(double dt, double angular_velocity, int target_id = 7) {
    ekf_tracker::reset();
    for (int frame = 0; frame < 12; ++frame) {
        const double timestamp = frame * dt;
        const PredictionResult result = ekf_tracker::update(
            {makeObservation(angular_velocity * timestamp, target_id)},
            neutralGimbal(), timestamp);
        if (frame >= 6) {
            assert(result.state == TrackingState::Stable);
        }
    }
}

void test_high_speed_rotation_is_not_rate_limited_to_old_cap() {
    constexpr double dt = 0.014;
    constexpr double angular_velocity = 12.0;
    ekf_tracker::reset();

    PredictionResult result;
    for (int frame = 0; frame < 120; ++frame) {
        const double timestamp = frame * dt;
        result = ekf_tracker::update(
            {makeObservation(angular_velocity * timestamp)},
            neutralGimbal(), timestamp);
    }

    // The previous implementation clamped yaw rate to +/-8 rad/s. A stable
    // high-speed track must be able to follow a 12 rad/s target instead.
    assert(result.state == TrackingState::Stable);
    assert(std::abs(result.body_yaw_velocity) > 9.0F);
}

void test_short_measurement_drop_does_not_flash_unstable() {
    constexpr double dt = 0.014;
    constexpr double angular_velocity = 4.0;
    warmUp(dt, angular_velocity);

    const double first_missing_timestamp = 12 * dt;
    const PredictionResult first_missing = ekf_tracker::update(
        {}, neutralGimbal(), first_missing_timestamp);
    const PredictionResult second_missing = ekf_tracker::update(
        {}, neutralGimbal(), first_missing_timestamp + dt);

    // A one- or two-frame detector gap should stay in the temporary-loss
    // grace period without revoking the stable output state.
    assert(first_missing.state == TrackingState::Stable);
    assert(!first_missing.fire_allowed);
    assert(second_missing.state == TrackingState::Stable);
    assert(!second_missing.fire_allowed);

    const PredictionResult recovered = ekf_tracker::update(
        {makeObservation(angular_velocity * (first_missing_timestamp + 2 * dt))},
        neutralGimbal(), first_missing_timestamp + 2 * dt);
    assert(recovered.state == TrackingState::Stable);
    assert(recovered.target_id == 7U);

    PredictionResult expired = recovered;
    for (int frame = 0; frame < 6; ++frame) {
        expired = ekf_tracker::update({}, neutralGimbal(),
                                      first_missing_timestamp + (3 + frame) * dt);
    }
    assert(expired.state == TrackingState::Unstable);
}

void test_single_frame_id_flip_keeps_active_track() {
    constexpr double dt = 0.014;
    constexpr double angular_velocity = 4.0;
    warmUp(dt, angular_velocity);

    const double flip_timestamp = 12 * dt;
    const PredictionResult flipped = ekf_tracker::update(
        {makeObservation(angular_velocity * flip_timestamp, 8)},
        neutralGimbal(), flip_timestamp);
    assert(flipped.state == TrackingState::Stable);
    assert(flipped.target_id == 7U);

    const PredictionResult flipped_again = ekf_tracker::update(
        {makeObservation(angular_velocity * (flip_timestamp + dt), 8)},
        neutralGimbal(), flip_timestamp + dt);
    assert(flipped_again.state == TrackingState::Stable);
    assert(flipped_again.target_id == 7U);

    const PredictionResult recovered = ekf_tracker::update(
        {makeObservation(angular_velocity * (flip_timestamp + 2 * dt), 7)},
        neutralGimbal(), flip_timestamp + 2 * dt);
    assert(recovered.state == TrackingState::Stable);
    assert(recovered.target_id == 7U);
}

}  // namespace

int main() {
    test_high_speed_rotation_is_not_rate_limited_to_old_cap();
    test_short_measurement_drop_does_not_flash_unstable();
    test_single_frame_id_flip_keeps_active_track();
    std::cout << "EKF stability regression tests passed\n";
    return 0;
}
