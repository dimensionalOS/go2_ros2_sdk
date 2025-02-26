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
        clusters = {}  # Store frontier clusters
        cluster_id = 0
        min_cluster_size = 3  # Reduced minimum cluster size for more frontiers
        
        # Scan with larger steps for efficiency on big maps
        step_size = 2
        for y in range(1, self.height - 1, step_size):
            for x in range(1, self.width - 1, step_size):
                if self.map_data[y, x] == free:
                    # Check if this free cell is next to an unknown cell
                    neighbors = [(y+dy, x+dx) for dy, dx in [
                        (0, step_size), (step_size, 0), (0, -step_size), (-step_size, 0),
                        (step_size, step_size), (-step_size, -step_size),
                        (step_size, -step_size), (-step_size, step_size)
                    ]]
                    unknown_count = sum(1 for ny, nx in neighbors 
                                     if 0 <= ny < self.height and 0 <= nx < self.width 
                                     and self.map_data[ny, nx] == unknown)
                    
                    if unknown_count > 0:
                        frontier_point = (x, y)
                        if self.is_valid_frontier(frontier_point):
                            # Check nearby points to form clusters
                            assigned_to_cluster = False
                            for ny, nx in neighbors:
                                if (nx, ny) in clusters:
                                    clusters[clusters[(nx, ny)]].append(frontier_point)
                                    clusters[frontier_point] = clusters[(nx, ny)]
                                    assigned_to_cluster = True
                                    break
                            
                            if not assigned_to_cluster:
                                clusters[cluster_id] = [frontier_point]
                                clusters[frontier_point] = cluster_id
                                cluster_id += 1

        # Filter clusters and add their centroids as frontiers
        for cluster_points in set(clusters.values()):
            if len(cluster_points) >= min_cluster_size:
                centroid_x = sum(p[0] for p in cluster_points) / len(cluster_points)
                centroid_y = sum(p[1] for p in cluster_points) / len(cluster_points)
                self.frontiers.append((int(centroid_x), int(centroid_y)))
        
        if not self.frontiers:
            # If no frontiers found, try a more aggressive search
            for y in range(1, self.height - 1, step_size):
                for x in range(1, self.width - 1, step_size):
                    if self.map_data[y, x] == free:
                        # Look further for unknown cells
                        search_range = 5
                        for dy in range(-search_range, search_range + 1, step_size):
                            for dx in range(-search_range, search_range + 1, step_size):
                                ny, nx = y + dy, x + dx
                                if (0 <= ny < self.height and 0 <= nx < self.width and 
                                    self.map_data[ny, nx] == unknown):
                                    frontier_point = (x, y)
                                    if self.is_valid_frontier(frontier_point):
                                        self.frontiers.append(frontier_point)
                                        break
                            if self.frontiers:
                                break
                    if self.frontiers:
                        break

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
                if distance < 1.0:  # Increased minimum distance to 1.0m
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

        # Sort frontiers by a weighted combination of distance and information gain
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        # Find best frontier using weighted distance and unexplored area
        def frontier_score(frontier):
            world_point = self.map_to_world(frontier)
            distance = math.sqrt(
                (world_point[0] - robot_pose.x) ** 2 +
                (world_point[1] - robot_pose.y) ** 2
            )
            # Count unknown cells in frontier neighborhood
            x, y = frontier
            unknown_count = 0
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < self.height and 0 <= nx < self.width and 
                        self.map_data[ny, nx] == -1):
                        unknown_count += 1
            
            # Weight distance vs exploration potential
            distance_weight = 0.3  # Lower weight means distance is less important
            exploration_weight = 0.7  # Higher weight favors areas with more unknown cells
            
            return -(distance_weight * distance - exploration_weight * unknown_count)

        best_frontier = max(self.frontiers, key=frontier_score)

        # Convert frontier point to PoseStamped
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        world_point = self.map_to_world(best_frontier)
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