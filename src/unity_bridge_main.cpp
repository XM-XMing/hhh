#include <ros/ros.h>

#include <planning/bridge/unity_bridge_node.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "unity_bridge_node");
  UnityBridgeNode node;
  if (!node.Init()) return 1;
  node.Spin();
  return 0;
}
