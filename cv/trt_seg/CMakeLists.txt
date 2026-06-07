cmake_minimum_required(VERSION 3.8)
project(trt_seg)

# =========================================================
# compiler config
# =========================================================
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# dependencies
# =========================================================
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_package(Eigen3 REQUIRED)

# =========================================================
# CUDA
# =========================================================
enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU arch (modify if needed)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# =========================================================
# TensorRT (与之前相同)
# =========================================================
set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})

if(NOT TENSORRT_ROOT)
  if(EXISTS "$ENV{HOME}/TensorRT-10.10.0.31")
    set(TENSORRT_ROOT "$ENV{HOME}/TensorRT-10.10.0.31")
  elseif(EXISTS "/usr/local/TensorRT")
    set(TENSORRT_ROOT "/usr/local/TensorRT")
  endif()
endif()

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${TENSORRT_ROOT}/include
)

find_library(TENSORRT_NVINFER_LIB
  NAMES nvinfer
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
)

find_library(TENSORRT_PLUGIN_LIB
  NAMES nvinfer_plugin
  HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_PLUGIN_LIB)
  message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT or install TensorRT.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib: ${TENSORRT_NVINFER_LIB}")

# =========================================================
# 目标：共享库（关键改动）
# =========================================================
add_library(${PROJECT_NAME} SHARED
  src/trt_node.cpp
  src/cuda_kernels.cu
)

# =========================================================
# 编译特性
# =========================================================
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# =========================================================
# includes
# =========================================================
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)

# =========================================================
# 链接库
# =========================================================
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  CUDA::cudart
  ${TENSORRT_NVINFER_LIB}
  ${TENSORRT_PLUGIN_LIB}
)

# ROS2 依赖
ament_target_dependencies(${PROJECT_NAME} PUBLIC
  rclcpp
  sensor_msgs
  cv_bridge
)

# =========================================================
# CUDA 属性
# =========================================================
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# =========================================================
# 安装（与 trt_yolo 类似）
# =========================================================
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY include/
  DESTINATION include
)

# =========================================================
# 导出依赖
# =========================================================
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  CUDAToolkit
  Eigen3
)

ament_package()