cmake_minimum_required(VERSION 3.0.2)
project(planning)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O2 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  roslib
  tf2
  tf2_ros
  visualization_msgs
  message_generation
)

catkin_python_setup()

find_package(OpenMP REQUIRED)
find_package(OpenSSL REQUIRED)

find_library(ZMQ_LIB zmq)
if(NOT ZMQ_LIB)
  message(FATAL_ERROR "libzmq not found. Install: sudo apt install libzmq3-dev")
endif()

find_library(MSGPACK_LIB msgpackc)
find_path(MSGPACK_INCLUDE_DIR msgpack.hpp)
if(NOT MSGPACK_INCLUDE_DIR)
  message(FATAL_ERROR "msgpack headers not found. Install: sudo apt install libmsgpack-dev")
endif()

add_message_files(
  FILES
  XMState.msg
  PrimitiveCommand.msg
  PrimitiveExecution.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp rospy std_msgs sensor_msgs geometry_msgs nav_msgs roslib
    tf2 tf2_ros visualization_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${MSGPACK_INCLUDE_DIR}
)

set(PLANNING_BRIDGE_SOURCES
  src/unity_bridge_main.cpp
  src/bridge/bridge_node.cpp
  src/bridge/bridge_util.cpp
  src/bridge/command_gateway.cpp
  src/bridge/result_gateway.cpp
  src/bridge/snapshot_gateway.cpp
  src/bridge/reset_gateway.cpp
  src/bridge/telemetry_bridge.cpp
  src/protocol/endpoint_observation_snapshot_wire.cpp
  src/protocol/telemetry_wire.cpp
  src/transport/bridge_transport.cpp
  src/transport/zmq_option_contract.cpp
  src/transport/zmq_socket.cpp
  src/transport/zmq_endpoint.cpp
  src/transport/command_transport.cpp
  src/transport/telemetry_transport.cpp
  src/transport/result_transport.cpp
  src/transport/snapshot_transport.cpp
)
add_executable(unity_bridge_node ${PLANNING_BRIDGE_SOURCES})
add_dependencies(unity_bridge_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(unity_bridge_node ${catkin_LIBRARIES} ${ZMQ_LIB} OpenSSL::Crypto)
if(MSGPACK_LIB)
  target_link_libraries(unity_bridge_node ${MSGPACK_LIB})
endif()

add_executable(map_loader_node src/tools/map_loader_node.cpp)
add_dependencies(map_loader_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(map_loader_node ${catkin_LIBRARIES})

add_executable(visualization_node src/tools/visualization_node.cpp)
add_dependencies(visualization_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(visualization_node ${catkin_LIBRARIES})

add_library(planning_voxel_map SHARED src/geometry/voxel_map.cpp)
target_compile_options(planning_voxel_map PRIVATE -O3)

add_library(planning_collision_checker SHARED src/geometry/collision_checker.cpp)
target_compile_options(planning_collision_checker PRIVATE -O3)
target_link_libraries(planning_collision_checker
  planning_voxel_map
  OpenMP::OpenMP_CXX
)

add_library(planning_depth_safety SHARED src/geometry/depth_safety.cpp)
target_compile_options(planning_depth_safety PRIVATE -O3)

set(PLANNING_GLOBAL_ROUTE_SOURCES
  src/navigation/global_route_planner.cpp
)
add_library(planning_global_route SHARED ${PLANNING_GLOBAL_ROUTE_SOURCES})
target_compile_options(planning_global_route PRIVATE -O3)
target_link_libraries(planning_global_route planning_voxel_map)

# Executable compatibility entry points. Shared APIs are migrated into the
# installable Python package under python/planning without changing rosrun UX.
catkin_install_python(PROGRAMS
  scripts/build_voxel_cache.py
  scripts/generate_motion_primitives.py
  scripts/prepare_teacher_missions.py
  scripts/collect_rollouts_parallel.py
  scripts/merge_teacher_rollouts.py
  scripts/label_teacher_rollouts.py
  scripts/generate_depth_action_masks.py
  scripts/audit_teacher_dataset.py
  scripts/build_bc_mmap_dataset.py
  scripts/train_soft_bc.py
  scripts/build_awac_final_holdout.py
  scripts/build_awac_mission_split.py
  scripts/build_awac_dev_set.py
  scripts/build_awac_calibration_split.py
  scripts/train_awac.py
  scripts/select_awac_dev_checkpoint.py
  scripts/audit_awac_replay.py
  scripts/evaluate_policy_unity.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  scripts/evaluate_policy_unity_managed.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS
  unity_bridge_node
  map_loader_node
  visualization_node
  planning_voxel_map
  planning_collision_checker
  planning_depth_safety
  planning_global_route
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)


install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  add_executable(voxel_map_parity_fixture
    tests/cpp/voxel_map_parity_fixture.cpp
  )
  add_dependencies(voxel_map_parity_fixture
    ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(voxel_map_parity_fixture
    planning_collision_checker
    planning_global_route
    ${CMAKE_DL_LIBS}
  )

  set(PLANNING_PYTEST_RESULT
    "${CATKIN_TEST_RESULTS_DIR}/${PROJECT_NAME}/pytest-unit.xml")
  set(PLANNING_PYTEST_COMMAND
    "${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_pytest_unit.py ${PLANNING_PYTEST_RESULT}")
  catkin_run_tests_target(
    "pytest"
    "unit"
    "pytest-unit.xml"
    COMMAND "${PLANNING_PYTEST_COMMAND}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

# Generated primitives are part of the runtime data contract, but are not
# created by a C++ build.  Keep source-only P checkouts installable without
# fabricating the generated artifact; a data-producing step must provide this
# directory before it is packaged.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/motion_primitives")
  install(DIRECTORY data/motion_primitives
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
  )
else()
  message(STATUS
    "Skipping generated data/motion_primitives install: directory is absent"
  )
endif()
