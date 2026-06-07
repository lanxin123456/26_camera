cmake_minimum_required(VERSION 3.8)
project(trt_yolo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT (prefer env TENSORRT_ROOT, fallback to $HOME/TensorRT-*)
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
if(NOT TENSORRT_ROOT)
  set(TENSORRT_ROOT "$ENV{HOME}/TensorRT-10.10.0.31")
endif()

find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS "${TENSORRT_ROOT}/include" "${TENSORRT_ROOT}"
)

find_library(TensorRT_NVINFER_LIB
  NAMES nvinfer
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
)

find_library(TensorRT_NVINFER_PLUGIN_LIB
  NAMES nvinfer_plugin
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64"
)

if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_NVINFER_LIB OR NOT TensorRT_NVINFER_PLUGIN_LIB)
  message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT env var (or install TensorRT system-wide).")
endif()

add_library(${PROJECT_NAME} SHARED
  src/yolo.cpp
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${OpenCV_LIBS}
  CUDA::cudart
  ${TensorRT_NVINFER_LIB}
  ${TensorRT_NVINFER_PLUGIN_LIB}
)

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

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV CUDAToolkit)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
