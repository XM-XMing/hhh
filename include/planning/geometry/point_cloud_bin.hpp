#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace planning {

struct PointXYZf {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

inline std::vector<PointXYZf> LoadForestPointCloudBin(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Cannot open point cloud file: " + path);
  }

  int32_t count = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!in || count < 0) {
    throw std::runtime_error("Invalid point cloud header: " + path);
  }

  std::vector<PointXYZf> points(static_cast<size_t>(count));
  if (count > 0) {
    in.read(reinterpret_cast<char*>(points.data()),
            static_cast<std::streamsize>(points.size() * sizeof(PointXYZf)));
  }

  if (!in) {
    throw std::runtime_error("Point cloud payload is shorter than header count: " + path);
  }

  return points;
}

}
