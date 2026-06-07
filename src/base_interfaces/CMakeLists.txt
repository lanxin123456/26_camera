cmake_minimum_required(VERSION 3.8)
project(base_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set (dependencies
  geometry_msgs
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/McuReceive.msg"
  "msg/McuTransmit.msg"
  "msg/PosData.msg"
  "msg/Motor.msg"
  "msg/Flag.msg"
  "msg/SteeringWheel.msg"
  "msg/HeartBag.msg"
  "msg/ActionPlanState.msg"
  "action/ActionPlan.action"
  "msg/Camera.msg"
  "msg/CameraKfs.msg"
  "msg/CameraTask.msg"
  "msg/ChooseWeapon.msg"
  "msg/LidarData.msg"
  "msg/PoseCorrection.msg"
  "msg/Esc.msg"
  "msg/Align.msg"
  "msg/Pick.msg"
  "msg/GridDistances.msg"
  "msg/GridStart.msg"
  "msg/GridState.msg"
  "msg/D455Task.msg"
  "msg/CameraChoose.msg"
  DEPENDENCIES nav_msgs builtin_interfaces geometry_msgs std_msgs sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
