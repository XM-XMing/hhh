#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>

#include <planning/XMState.h>

namespace {

constexpr double kPi = 3.14159265358979323846;

geometry_msgs::Point MakePoint(double x, double y, double z) {
  geometry_msgs::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

std_msgs::ColorRGBA MakeColor(float r, float g, float b, float a) {
  std_msgs::ColorRGBA c;
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
  return c;
}

std::array<double, 3> RotateVector(const geometry_msgs::Quaternion& q,
                                   const std::array<double, 3>& v) {
  const double x = q.x;
  const double y = q.y;
  const double z = q.z;
  const double w = q.w;

  const double tx = 2.0 * (y * v[2] - z * v[1]);
  const double ty = 2.0 * (z * v[0] - x * v[2]);
  const double tz = 2.0 * (x * v[1] - y * v[0]);

  return {{
      v[0] + w * tx + (y * tz - z * ty),
      v[1] + w * ty + (z * tx - x * tz),
      v[2] + w * tz + (x * ty - y * tx)}};
}

visualization_msgs::Marker BaseMarker(const std::string& frame_id,
                                       const ros::Time& stamp,
                                       int id,
                                       int type,
                                       const std::string& ns) {
  visualization_msgs::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = ns;
  m.id = id;
  m.type = type;
  m.action = visualization_msgs::Marker::ADD;
  m.lifetime = ros::Duration(0.2);
  return m;
}

}  // namespace

class VisualizationNode {
 public:
  VisualizationNode() : nh_(""), pnh_("~") {
    pnh_.param<std::string>("state_topic", state_topic_, "/xm/state");
    pnh_.param<std::string>("marker_topic", marker_topic_, "/xm/markers");
    pnh_.param<double>("drone_radius", drone_radius_, 0.2);
    pnh_.param<double>("front_ray_angle_deg", front_ray_angle_deg_, 25.0);
    pnh_.param<double>("marker_lift_z", marker_lift_z_, 0.03);

    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(marker_topic_, 2);
    state_sub_ = nh_.subscribe(state_topic_, 10, &VisualizationNode::StateCallback, this);
  }

 private:
  void StateCallback(const planning::XMStateConstPtr& s) {
    visualization_msgs::MarkerArray arr;
    const std::string& frame = s->header.frame_id;
    const ros::Time stamp = s->header.stamp;

    visualization_msgs::Marker body = BaseMarker(frame, stamp, 0, visualization_msgs::Marker::SPHERE, "drone_body");
    body.pose.position = s->position;
    body.pose.position.z += marker_lift_z_;
    body.pose.orientation.w = 1.0;
    body.scale.x = drone_radius_ * 2.0;
    body.scale.y = drone_radius_ * 2.0;
    body.scale.z = drone_radius_ * 2.0;
    body.color = s->collided ? MakeColor(1.0f, 0.05f, 0.05f, 0.85f) : MakeColor(0.1f, 0.45f, 1.0f, 0.85f);
    arr.markers.push_back(body);

    const auto forward = RotateVector(s->orientation, {{1.0, 0.0, 0.0}});
    visualization_msgs::Marker arrow = BaseMarker(frame, stamp, 1, visualization_msgs::Marker::ARROW, "drone_forward");
    arrow.points.push_back(s->position);
    arrow.points.push_back(MakePoint(s->position.x + forward[0] * 0.9,
                                     s->position.y + forward[1] * 0.9,
                                     s->position.z + forward[2] * 0.9));
    arrow.scale.x = 0.04;
    arrow.scale.y = 0.10;
    arrow.scale.z = 0.10;
    arrow.color = MakeColor(0.0f, 1.0f, 0.1f, 0.9f);
    arr.markers.push_back(arrow);

    const std::array<double, 3> angles_deg{{front_ray_angle_deg_, 0.0, -front_ray_angle_deg_}};
    for (size_t i = 0; i < 3; ++i) {
      const double a = angles_deg[i] * kPi / 180.0;
      const auto dir = RotateVector(s->orientation, {{std::cos(a), std::sin(a), 0.0}});
      const double len = std::max(0.0f, s->front_clearances[i]);

      visualization_msgs::Marker line = BaseMarker(frame, stamp, static_cast<int>(10 + i),
                                                   visualization_msgs::Marker::LINE_STRIP,
                                                   "front_clearance");
      line.points.push_back(s->position);
      line.points.push_back(MakePoint(s->position.x + dir[0] * len,
                                      s->position.y + dir[1] * len,
                                      s->position.z + dir[2] * len));
      line.scale.x = 0.035;
      line.color = len < 0.5 ? MakeColor(1.0f, 0.0f, 0.0f, 0.9f) : MakeColor(1.0f, 0.8f, 0.0f, 0.9f);
      arr.markers.push_back(line);
    }

    visualization_msgs::Marker text = BaseMarker(frame, stamp, 20, visualization_msgs::Marker::TEXT_VIEW_FACING, "state_text");
    text.pose.position = s->position;
    text.pose.position.z += 0.6;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.25;
    text.color = MakeColor(1.0f, 1.0f, 1.0f, 0.95f);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "z=%.2f  min=%.2f  v=%.2f", s->position.z, s->min_clearance,
                  std::sqrt(s->velocity.x * s->velocity.x + s->velocity.y * s->velocity.y + s->velocity.z * s->velocity.z));
    text.text = buf;
    arr.markers.push_back(text);

    marker_pub_.publish(arr);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Publisher marker_pub_;
  ros::Subscriber state_sub_;

  std::string state_topic_;
  std::string marker_topic_;
  double drone_radius_ = 0.2;
  double front_ray_angle_deg_ = 25.0;
  double marker_lift_z_ = 0.03;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "visualization_node");
  VisualizationNode node;
  ros::spin();
  return 0;
}
