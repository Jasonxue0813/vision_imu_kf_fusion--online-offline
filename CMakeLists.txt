cmake_minimum_required(VERSION 3.10)
project(splg_fusion)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(GDAL REQUIRED)
find_package(Threads REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp rosbag sensor_msgs std_msgs)

set(ORT_DIR "" CACHE PATH "Optional path to ONNX Runtime root (contains include/ and lib/)")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
        ${ORT_DIR}
        ${ORT_DIR}/include
        /usr
        /usr/local
        /opt/onnxruntime
    PATH_SUFFIXES include)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        ${ORT_DIR}
        ${ORT_DIR}/lib
        ${ORT_DIR}/lib64
        /usr/lib
        /usr/local/lib
        /opt/onnxruntime/lib
        /opt/onnxruntime/lib64)

set(SPLG_FUSION_HAS_ORT TRUE)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    set(SPLG_FUSION_HAS_ORT FALSE)
    message(WARNING
        "ONNX Runtime not found. `sp_lg` target will be skipped. "
        "Set ORT_DIR to the runtime root that contains include/ and lib/ if you want to build it."
    )
endif()

if(SPLG_FUSION_HAS_ORT)
    add_executable(sp_lg
        src/vision/sp_lg.cpp
    )
endif()

add_executable(visual_imu_fusion
    src/main.cpp
    src/app/online_ypr_crop_node_core.cpp
    src/app/online_ypr_crop_node_runtime.cpp
    src/app/online_ypr_crop_node_utils.cpp
    src/vision/online_ypr_crop_node_vision.cpp
    src/fusion/online_ypr_crop_node_fusion.cpp
    src/global_heading.cpp
)

if(SPLG_FUSION_HAS_ORT)
    target_include_directories(sp_lg PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${GDAL_INCLUDE_DIR}
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

target_include_directories(visual_imu_fusion PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${GDAL_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(SPLG_FUSION_HAS_ORT)
    target_link_libraries(sp_lg PRIVATE
        ${OpenCV_LIBS}
        ${GDAL_LIBRARY}
        ${ONNXRUNTIME_LIBRARY}
    )
endif()

target_link_libraries(visual_imu_fusion PRIVATE
    ${OpenCV_LIBS}
    ${GDAL_LIBRARY}
    ${catkin_LIBRARIES}
    Threads::Threads
)

if(SPLG_FUSION_HAS_ORT)
    get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
    set_target_properties(sp_lg PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
endif()
