cmake_minimum_required(VERSION 3.14)
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 89 120) # 设置CUDA架构
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)         # 设置CUDA编译器

project(publisher)

enable_language(CXX CUDA) # 启用C++和CUDA语言 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -g") # 设置C++编译选项 为C++17
set(CMAKE_CXX_STANDARD 17) # 设置C++标准 为C++17
set(CMAKE_BUILD_TYPE Release) # 设置构建类型 为Release
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF) # 设置CUDA使用动态运行时库
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") # 设置Release模式下的编译选项

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED) # ROS2中基于CMake的软件包的构建系统
find_package(rosidl_default_generators REQUIRED) # 定义默认ROS接口生成器的配置包
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(base_interfaces REQUIRED)  
find_package(trt_yolo REQUIRED)  
find_package(rclcpp REQUIRED) # rclcpp提供了与ROS2交互的标准 C++API     
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
# === TF 相关依赖（新增）===
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# 如需 Eigen 下的变换，可再启用：
find_package(tf2_eigen REQUIRED)

# Eigen3
find_package(Eigen3 REQUIRED)

# CUDA
set(CUDA_DIR /usr/local/cuda)
find_package(CUDA REQUIRED)

# OpenCV
set(OpenCV_DIR /usr/local/share/opencv4)
find_package(OpenCV REQUIRED)

# PCL
set(PCL_DIR /usr/local/share/pcl-1.15)
find_package(PCL REQUIRED)

# realsense
set(realsense2_DIR /usr/share/realsense2)
find_package(realsense2 REQUIRED)

# tensorrt
set(TensorRT_INCLUDE_DIRS $ENV{HOME}/TensorRT-10.10.0.31/include)
set(TensorRT_LIBRARIES nvinfer nvinfer_plugin cudart)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 关键修复1：使用target_include_directories而不是include_directories
file(GLOB SOURCE_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
if(NOT SOURCE_LIST)
  message(FATAL_ERROR "No source files found in ${PROJECT_SOURCE_DIR}/src (expected at least one .cpp with main())")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_LIST})

# 关键修复2：正确设置包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # 添加当前项目的include目录
    ${CUDA_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIR} 
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${pcl_conversions_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    /opt/ros/jazzy/include/
    /opt/ros/jazzy/include/message_filters
    /opt/ros/jazzy/include/pcl_msgs
)

# 关键修复3：正确链接库
target_link_libraries(${PROJECT_NAME} 
    nvinfer 
    nvinfer_plugin 
    cudart 
    trt_yolo::trt_yolo 
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${realsense2_LIBRARY}
    ${TensorRT_LIBRARIES}
    ${pcl_conversions_LIBRARIES}
)

# 关键修复4：使用ament_target_dependencies
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  base_interfaces
  sensor_msgs
  std_msgs
  cv_bridge
  pcl_conversions
  nav_msgs
  tf2               # 新增
  tf2_ros           # 新增：TransformBuffer/Listener/Broadcaster
  tf2_geometry_msgs # 新增：Pose/Vector/Stamped 等与 TF 的互转
  tf2_eigen       # 若你在代码里用了 Eigen 与 TF 的互转，再放开
)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

ament_package()