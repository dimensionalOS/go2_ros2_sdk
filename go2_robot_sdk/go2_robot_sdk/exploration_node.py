#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import math
from collections import deque

class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')
        
        # Create action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Create subscriber for map updates
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            qos)
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Store the map data
        self.map_data = None
        self.resolution = None
        self.width = None
        self.height = None
        self.origin = None
        
        # Frontier tracking
        self.current_goal = None
        self.frontiers = []
        self.visited_frontiers = set()
        
        self.get_logger().info('Exploration node initialized')

    def map_callback(self, msg):
        """Process incoming map data and find frontiers."""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.resolution = msg.info.resolution
        self.width = msg.info.width
        self.height = msg.info.height
        self.origin = msg.info.origin
        
        if self.current_goal is None:
            self.find_frontiers()
            self.send_exploration_goal()

    def find_frontiers(self):
        """Find frontier cells in the map."""
        self.frontiers = []
        
        # Define frontier criteria
        unknown = -1
        free = 0
        occupied = 100
        
        # Find frontier cells (free cells next to unknown cells)
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.map_data[y, x] == free:
                    # Check if this free cell is next to an unknown cell
                    if any(self.map_data[y + dy, x + dx] == unknown
                           for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                        frontier_point = (x, y)
                        if self.is_valid_frontier(frontier_point):
                            self.frontiers.append(frontier_point)

    def is_valid_frontier(self, point):
        """Check if a frontier point is valid and not already visited."""
        # Convert point to world coordinates
        world_point = self.map_to_world(point)
        point_key = f"{world_point[0]:.1f},{world_point[1]:.1f}"
        
        if point_key in self.visited_frontiers:
            return False
            
        # Add minimum distance check from current position
        try:
            robot_pose = self.get_robot_pose()
            if robot_pose is not None:
                distance = math.sqrt(
                    (world_point[0] - robot_pose.x) ** 2 +
                    (world_point[1] - robot_pose.y) ** 2
                )
                if distance < 0.5:  # Minimum 0.5m distance
                    return False
        except Exception as e:
            self.get_logger().warning(f'Could not get robot pose: {e}')
            return False
            
        return True

    def map_to_world(self, point):
        """Convert map coordinates to world coordinates."""
        x = point[0] * self.resolution + self.origin.position.x
        y = point[1] * self.resolution + self.origin.position.y
        return (x, y)

    def get_robot_pose(self):
        """Get the current robot pose in the map frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
            return transform.transform.translation
        except Exception as e:
            self.get_logger().warning(f'Failed to get robot pose: {e}')
            return None

    def send_exploration_goal(self):
        """Send the next exploration goal."""
        if not self.frontiers:
            self.get_logger().info('No more frontiers to explore!')
            return

        # Sort frontiers by distance to current position
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        # Find closest frontier
        closest_frontier = min(
            self.frontiers,
            key=lambda f: (
                (self.map_to_world(f)[0] - robot_pose.x) ** 2 +
                (self.map_to_world(f)[1] - robot_pose.y) ** 2
            )
        )

        # Convert frontier point to PoseStamped
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        world_point = self.map_to_world(closest_frontier)
        goal_pose.pose.position.x = world_point[0]
        goal_pose.pose.position.y = world_point[1]
        goal_pose.pose.orientation.w = 1.0

        # Send goal
        self.current_goal = goal_pose
        self.send_goal(goal_pose)

    def send_goal(self, pose):
        """Send a goal to the navigation stack."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.nav_client.wait_for_server()
        
        # Store the frontier as visited
        point_key = f"{pose.pose.position.x:.1f},{pose.pose.position.y:.1f}"
        self.visited_frontiers.add(point_key)
        
        # Send the goal
        self.get_logger().info(f'Sending goal: {pose.pose.position.x}, {pose.pose.position.y}')
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle the goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.current_goal = None
            self.send_exploration_goal()
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Handle the goal result."""
        result = future.result().result
        self.get_logger().info('Goal finished')
        self.current_goal = None
        # Find new frontiers and continue exploration
        self.find_frontiers()
        self.send_exploration_goal()

def main():
    rclpy.init()
    node = ExplorationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 