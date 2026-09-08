#include <algorithm>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <planning/geometry/point_cloud_bin.hpp>

namespace {

bool IsAbsolutePath(const std::string& path) {
  return !path.empty() && path[0] == '/';
}

std::string ResolvePackageRelativePath(const std::string& path) {
  if (IsAbsolutePath(path)) {
    return path;
  }

  const std::string pkg_path = ros::package::getPath("planning");
  if (pkg_path.empty()) {
    throw std::runtime_error("ros::package::getPath(\"planning\") failed");
  }

  if (path.empty()) {
    return pkg_path + "/data/map_data/forest_point_cloud.bin";
  }

  return pkg_path + "/" + path;
}

float Clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}


planning::PointXYZf ConvertPoint(const planning::PointXYZf& p, bool convert_unity_to_ros) {
  if (!convert_unity_to_ros) return p;
  // Unity world -> ROS map: [x_ros, y_ros, z_ros] = [z_unity, -x_unity, y_unity].
  return planning::PointXYZf{p.z, -p.x, p.y};
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "map_loader_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string map_path_param;
  std::string frame_id;
  std::string topic;
  pnh.param<std::string>("map_path", map_path_param, "data/map_data/forest_point_cloud.bin");
  pnh.param<std::string>("frame_id", frame_id, "map");
  pnh.param<std::string>("topic", topic, "/xm/map_points");

  bool convert_unity_to_ros = true;
  bool color_by_height = true;
  bool invert_height_color = false;
  double color_min_z_param = 0.0;
  double color_max_z_param = 0.0;
  pnh.param<bool>("convert_unity_to_ros", convert_unity_to_ros, true);
  pnh.param<bool>("color_by_height", color_by_height, true);
  pnh.param<bool>("invert_height_color", invert_height_color, false);
  pnh.param<double>("color_min_z", color_min_z_param, 0.0);
  pnh.param<double>("color_max_z", color_max_z_param, 0.0);

  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>(topic, 1, true);

  std::string map_path;
  std::vector<planning::PointXYZf> src_points;
  try {
    map_path = ResolvePackageRelativePath(map_path_param);
    src_points = planning::LoadForestPointCloudBin(map_path);
  } catch (const std::exception& e) {
    ROS_ERROR("Failed to load map: %s", e.what());
    return 1;
  }

  std::vector<planning::PointXYZf> points;
  points.reserve(src_points.size());
  float min_z = std::numeric_limits<float>::infinity();
  float max_z = -std::numeric_limits<float>::infinity();
  for (const auto& p : src_points) {
    planning::PointXYZf q = ConvertPoint(p, convert_unity_to_ros);
    min_z = std::min(min_z, q.z);
    max_z = std::max(max_z, q.z);
    points.push_back(q);
  }

  if (points.empty()) {
    min_z = 0.0f;
    max_z = 1.0f;
  }

  if (color_max_z_param > color_min_z_param) {
    min_z = static_cast<float>(color_min_z_param);
    max_z = static_cast<float>(color_max_z_param);
  }
  const float z_range = std::max(1e-6f, max_z - min_z);

  sensor_msgs::PointCloud2 cloud;
  cloud.header.frame_id = frame_id;
  cloud.header.stamp = ros::Time::now();
  cloud.height = 1;
  cloud.width = static_cast<uint32_t>(points.size());
  cloud.is_bigendian = false;
  cloud.is_dense = true;

  sensor_msgs::PointCloud2Modifier modifier(cloud);
  // Full-resolution map, no downsampling.
  // Use x/y/z + intensity only. RViz displays height color via Intensity/Rainbow.
  // This keeps all points and reduces payload from 20 B/point to 16 B/point.
  modifier.setPointCloud2Fields(4,
                                "x", 1, sensor_msgs::PointField::FLOAT32,
                                "y", 1, sensor_msgs::PointField::FLOAT32,
                                "z", 1, sensor_msgs::PointField::FLOAT32,
                                "intensity", 1, sensor_msgs::PointField::FLOAT32);
  modifier.resize(points.size());

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud, "intensity");

  for (const auto& p : points) {
    *iter_x = p.x;
    *iter_y = p.y;
    *iter_z = p.z;

    float t = (p.z - min_z) / z_range;
    if (invert_height_color) t = 1.0f - t;
    t = Clamp01(t);
    *iter_intensity = color_by_height ? t : p.z;

    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_intensity;
  }

  pub.publish(cloud);
  ROS_INFO("Published full-resolution map point cloud: points=%zu topic=%s frame=%s map_path_param=%s resolved_path=%s convert_unity_to_ros=%s color_by_height=%s point_step=%u z_min=%.3f z_max=%.3f",
           points.size(), topic.c_str(), frame_id.c_str(), map_path_param.c_str(), map_path.c_str(),
           convert_unity_to_ros ? "true" : "false", color_by_height ? "true" : "false", cloud.point_step, min_z, max_z);

  ros::spin();
  return 0;
}
