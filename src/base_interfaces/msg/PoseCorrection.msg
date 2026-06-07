# PoseCorrection.msg
# 包含map->base_link的位姿，用于矫正里程计

# Header信息
std_msgs/Header header

# map->base_link的位姿
geometry_msgs/Pose pose

# 矫正标志：需要矫正时置 1
bool need_correction
