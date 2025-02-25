from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Include the robot launch file
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('go2_robot_sdk'),
                'launch',
                'robot.launch.py'
            ])
        ])
    )
    
    # Create and return launch description
    ld = LaunchDescription()
    
    # Add the robot launch
    ld.add_action(robot_launch)
    
    # Add the exploration node
    exploration_node = Node(
        package='go2_robot_sdk',
        executable='exploration_node',
        name='exploration_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    ld.add_action(exploration_node)
    
    return ld 