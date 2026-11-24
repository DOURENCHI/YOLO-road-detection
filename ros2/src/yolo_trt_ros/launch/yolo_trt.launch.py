"""启动 yolo_trt_node。

只负责节点本身，不负责 Gazebo —— 仿真用官方 launch 单独起，
这样换场景（turtlebot3_world / 自建 world）时不用改这个文件。

    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
    ros2 launch yolo_trt_ros yolo_trt.launch.py detector:=dummy
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('yolo_trt_ros')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    detector = LaunchConfiguration('detector')
    ep = LaunchConfiguration('ep')
    image_topic = LaunchConfiguration('image_topic')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_worker = LaunchConfiguration('use_worker_thread')
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        DeclareLaunchArgument(
            'detector', default_value='dummy',
            description='dummy 验证通信层 / ort 真推理'),
        DeclareLaunchArgument(
            'ep', default_value='cpu',
            description='cpu / cuda / trt'),
        DeclareLaunchArgument(
            'image_topic', default_value='/camera/image_raw',
            description='输入图像话题'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='参数文件'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='有 /clock 发布者（Gazebo）时才设 true；'
                        '用 fake_camera 或真实相机时必须 false，否则 now() 恒为 0'),
        DeclareLaunchArgument(
            'use_worker_thread', default_value='false',
            description='v2 优化开关'),
        DeclareLaunchArgument(
            'log_level', default_value='info',
            description='debug / info / warn / error'),

        Node(
            package='yolo_trt_ros',
            executable='yolo_trt_node',
            name='yolo_trt_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                params_file,
                {
                    'detector': detector,
                    'ep': ep,
                    'image_topic': image_topic,
                    # 必须包一层 ParameterValue：LaunchConfiguration 求值后是字符串，
                    # 直接传给 bool 参数会报
                    #   "parameter 'use_sim_time' has invalid type: expected [bool] got [string]"
                    'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
                    'use_worker_thread': ParameterValue(use_worker, value_type=bool),
                },
            ],
            arguments=['--ros-args', '--log-level', log_level],
        ),
    ])
