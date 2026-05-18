# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def spawn_robot_opaque(context, *args, **kwargs):
    # Get the urdf file
    TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
    model_folder = 'turtlebot3_' + TURTLEBOT3_MODEL
    urdf_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        model_folder,
        'model.sdf'
    )

    lidar_rate = LaunchConfiguration('lidar_rate').perform(context)

    # Read original SDF
    with open(urdf_path, 'r') as f:
        xml_content = f.read()

    # Inject the desired lidar update rate
    # We replace the hardcoded 10 with the parameter value
    xml_content = xml_content.replace('<update_rate>10</update_rate>', f'<update_rate>{lidar_rate}</update_rate>')

    # Write to temporary file so spawn_entity can read it
    temp_sdf = tempfile.NamedTemporaryFile(delete=False, suffix='.sdf')
    temp_sdf.write(xml_content.encode('utf-8'))
    temp_sdf.close()

    return [
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', TURTLEBOT3_MODEL,
                '-file', temp_sdf.name,
                '-x', LaunchConfiguration('x_pose'),
                '-y', LaunchConfiguration('y_pose'),
                '-z', '0.01'
            ],
            output='screen',
        )
    ]

def generate_launch_description():
    # Declare the launch arguments
    declare_x_position_cmd = DeclareLaunchArgument(
        'x_pose', default_value='0.0',
        description='Specify x namespace of the robot')

    declare_y_position_cmd = DeclareLaunchArgument(
        'y_pose', default_value='0.0',
        description='Specify y namespace of the robot')

    declare_lidar_rate_cmd = DeclareLaunchArgument(
        'lidar_rate', default_value='10',
        description='Specify LiDAR update rate')

    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_x_position_cmd)
    ld.add_action(declare_y_position_cmd)
    ld.add_action(declare_lidar_rate_cmd)

    # Add any conditioned actions
    ld.add_action(OpaqueFunction(function=spawn_robot_opaque))

    return ld
