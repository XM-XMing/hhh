"""Stable Python APIs for the XM 3-D flight planning project."""

from pkgutil import extend_path

# ROS message generation contributes ``planning.msg`` from catkin devel/install
# space. Keep the source package and generated-message package composable.
__path__ = extend_path(__path__, __name__)

from planning.version import SOFTWARE_VERSION

__version__ = SOFTWARE_VERSION
