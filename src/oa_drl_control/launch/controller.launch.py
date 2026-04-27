from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import TimerAction

def generate_launch_description():

    gazebo_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('turtlebot3_gazebo'),
            'launch', 'lab02.launch.py'
        )))


    controller_node = TimerAction(
        period=8.0,
        actions=[
        
        Node(
        package='oa_drl_control',
        executable='filter_lidar',
        name='lidar_filter_node',
        
        parameters= [PathJoinSubstitution([
            FindPackageShare('lab03_pkg'), 'config', 'controller_params.yaml'
        ])]
    )]

    )

    
    return LaunchDescription([
        gazebo_node,
        controller_node
    ])
