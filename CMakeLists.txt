cmake_minimum_required(VERSION 3.16)
project(RMVTraining LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_WEB_DEBUG "Build the scalar Web Debug dashboard" ON)
option(BUILD_TESTING "Build the starter-project tests" ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d dnn highgui)
find_package(Threads REQUIRED)

# HIKROBOT MVS SDK
set(MVS_ROOT "/opt/MVS" CACHE PATH "HIKROBOT MVS SDK root")
find_path(MVS_INCLUDE_DIR MvCameraControl.h
    PATHS "${MVS_ROOT}/include"
    NO_DEFAULT_PATH)
find_library(MVS_LIBRARY NAMES MvCameraControl
    PATHS "${MVS_ROOT}/lib/64"
    NO_DEFAULT_PATH)
if(NOT MVS_INCLUDE_DIR OR NOT MVS_LIBRARY)
    message(FATAL_ERROR "HIKROBOT MVS SDK not found. Set MVS_ROOT to its installation directory.")
endif()

add_subdirectory(serialPort)
if(ENABLE_WEB_DEBUG)
    add_subdirectory(web_debug)
endif()

add_library(algorithm_support STATIC
    armor/src/BulletModel.cpp
    armor/src/KalmanFilter.cpp
)
target_include_directories(algorithm_support PUBLIC armor/include)
target_link_libraries(algorithm_support PUBLIC Eigen3::Eigen)

add_library(training_core STATIC
    src/AppSupport.cpp
    src/AutoAimTypes.cpp
    student/src/StudentTasks.cpp
    student/src/MVS.cpp
    student/src/ArmorDetector.cpp
    student/src/NumberClassifier.cpp
    student/src/ArmorSolver.cpp
    student/src/KalmanFilter.cpp
)
target_include_directories(training_core
    PUBLIC
        include
        armor/include
        student/include
        serialPort/include
        web_debug/include
        web_debug/3rdparty
        ${MVS_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(training_core PUBLIC serialPort algorithm_support Eigen3::Eigen ${OpenCV_LIBS} ${MVS_LIBRARY})
target_compile_definitions(training_core PRIVATE
    RMVTRAINING_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/student/model")
if(ENABLE_WEB_DEBUG)
    target_compile_definitions(training_core PRIVATE WEB_DEBUG)
endif()

add_executable(rmv_training main.cpp)
target_include_directories(rmv_training PRIVATE web_debug/include)
target_link_libraries(rmv_training
    PRIVATE
        training_core
        algorithm_support
        Threads::Threads
)
if(ENABLE_WEB_DEBUG)
    target_link_libraries(rmv_training PRIVATE web_debug)
endif()

if(BUILD_TESTING)
    enable_testing()
    add_executable(test_scaffold tests/test_scaffold.cpp)
    target_link_libraries(test_scaffold PRIVATE training_core)
    add_test(NAME scaffold_safety COMMAND test_scaffold)

    add_executable(test_ekf_stability tests/test_ekf_stability.cpp)
    target_link_libraries(test_ekf_stability PRIVATE training_core)
    add_test(NAME ekf_stability COMMAND test_ekf_stability)
    add_test(NAME cli_help COMMAND rmv_training --help)
endif()
