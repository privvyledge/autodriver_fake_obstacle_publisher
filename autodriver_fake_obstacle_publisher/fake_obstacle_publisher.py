"""
Todo:
    * replace uninitialized parameters with default values or Falsey values
    * switch to SolidPrimitive().CYLINDER for pedestrians and specify size as radius
"""
import random
import yaml
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from derived_object_msgs.msg import Object, ObjectArray
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
import struct
import sys


class FakeObstaclePublisher(Node):
    def __init__(self):
        super(FakeObstaclePublisher, self).__init__('fake_obstacle_publisher')

        # Load parameters
        self.declare_parameter('config_file', '')
        self.declare_parameter('publish_rate', None)  # Optional overwrite
        self.declare_parameter('frame_id', None)  # Optional overwrite
        self.declare_parameter('roi', None)  # Optional region of interest (x_min, x_max, y_min, y_max, z_min, z_max)
        self.declare_parameter('respawn', True)  # Whether to respawn obstacles removed by ROI filtering
        self.declare_parameter('random_seed', None)  # Optional random seed for reproducibility
        self.declare_parameter('dropout_percentage', 0.0)  # Percentage of obstacles to drop (simulate false negatives)

        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        random_seed = self.get_parameter('random_seed').get_parameter_value().integer_value
        self.dropout_percentage = self.get_parameter('dropout_percentage').get_parameter_value().double_value

        if random_seed is not None:
            random.seed(random_seed)
            self.get_logger().info(f"Random seed set to: {random_seed}")
        else:
            self.get_logger().info("Random seed not set, using true randomness.")

        if config_file:
            with open(config_file, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.get_logger().error("No config file provided. Exiting.")
            self.destroy_node()
            sys.exit(1)

        # Get optional parameters or use defaults from the config file
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value or self.config.get('publish_rate', 10.0)
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value or self.config.get('frame_id', 'map')
        self.roi = self.get_parameter('roi').get_parameter_value().double_array_value or self.config.get('roi', [-20.0, 20.0, -20.0, 20.0, 0.0, 5.0])
        self.respawn = self.get_parameter('respawn').get_parameter_value().bool_value

        # Validate dropout percentage
        if not (0.0 <= self.dropout_percentage <= 100.0):
            self.get_logger().error("Dropout percentage must be between 0 and 100. Exiting.")
            self.destroy_node()
            return

        # Publishers
        self.obstacles_pub = self.create_publisher(MarkerArray, 'fake_obstacles/marker_array', 10)
        self.object_array_pub = self.create_publisher(ObjectArray, 'fake_obstacles/object_array', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, 'fake_obstacles/pointcloud', 10)
        self.path_pubs = []  # One Path publisher per obstacle

        # Timer for publishing
        self.dt = 1 / self.publish_rate
        self.timer = self.create_timer(self.dt, self.publish_obstacles)

        # Initialize obstacles and their paths
        self.obstacles = self.create_obstacles()
        self.paths = [[] for _ in range(len(self.obstacles))]  # List of paths, one per obstacle
        self.get_logger().info(f"Initialized {len(self.obstacles)} obstacles.")

        # Create a Path publisher for each obstacle
        for i in range(len(self.obstacles)):
            self.path_pubs.append(self.create_publisher(Path, f'fake_obstacles/obstacle_{i}/path', 10))

        # Log details
        self.get_logger().info(f"Publish rate: {self.publish_rate} Hz, Frame ID: {self.frame_id}")
        self.get_logger().info(f"ROI: {self.roi}, Respawn: {self.respawn}")
        self.get_logger().info(f"Dropout Percentage: {self.dropout_percentage}%")

    def create_obstacles(self):
        obstacles = []
        actor_types = self.config.get('actors', {})
        for actor_type, params in actor_types.items():
            num_actors = params.get('count', 0)
            for _ in range(num_actors):
                obstacle = {
                    'type': actor_type,
                    'position': params.get('initial_position', [random.uniform(self.roi[0], self.roi[1]), random.uniform(self.roi[2], self.roi[3]), random.uniform(self.roi[4], self.roi[5])]),
                    'velocity': params.get('velocity', [0.0, 0.0, 0.0]),
                    'acceleration': params.get('acceleration', [0.0, 0.0, 0.0]),
                    'size': params.get('size', [1.0, 1.0, 1.0]),
                    'orientation': params.get('initial_orientation', 0.0),  # For car-like actors
                    'steering_angle': 0.0,  # For car-like actors
                    'angular_velocity': params.get('angular_velocity', [0.0, 0.0, 0.0]),  # For pedestrians
                }
                obstacles.append(obstacle)
        return obstacles

    def update_kinematic_obstacle(self, obstacle, dt):
        # Kinematic model for car-like actors
        position = obstacle['position']
        orientation = obstacle['orientation']
        velocity = obstacle['velocity']
        steering_angle = obstacle['steering_angle']

        # Update position and orientation
        v = velocity[0]  # Assume velocity is along the x-axis of the car
        L = 2.5  # Wheelbase (adjust as needed)
        delta = steering_angle

        # Kinematic bicycle model
        dx = v * math.cos(orientation) * dt
        dy = v * math.sin(orientation) * dt
        dtheta = (v / L) * math.tan(delta) * dt

        obstacle['position'][0] += dx
        obstacle['position'][1] += dy
        obstacle['orientation'] += dtheta
        return obstacle

    def update_pedestrian(self, obstacle, dt):
        # 6-DOF model for pedestrian
        position = obstacle['position']
        velocity = obstacle['velocity']
        acceleration = obstacle['acceleration']
        angular_velocity = obstacle['angular_velocity']

        # Linear motion
        for i in range(3):
            position[i] += velocity[i] * dt
            velocity[i] += acceleration[i] * dt

        # Angular motion
        for i in range(3):
            obstacle['orientation'] += angular_velocity[i] * dt

        return obstacle

    def update_generic(self, obstacle, dt):
        # Simple motion model: position += velocity * dt
        obstacle['position'] = [
            obstacle['position'][i] + obstacle['velocity'][i] * dt
            for i in range(3)
        ]
        return obstacle

    def is_within_roi(self, position):
        x, y, z = position
        return (self.roi[0] <= x <= self.roi[1] and
                self.roi[2] <= y <= self.roi[3] and
                self.roi[4] <= z <= self.roi[5])

    def respawn_obstacle(self, obstacle):
        obstacle['position'] = [
            random.uniform(self.roi[0], self.roi[1]),
            random.uniform(self.roi[2], self.roi[3]),
            random.uniform(self.roi[4], self.roi[5]),
        ]
        obstacle['velocity'] = [0.0, 0.0, 0.0]
        obstacle['acceleration'] = [0.0, 0.0, 0.0]
        return obstacle

    def update_obstacle(self, obstacle, dt):
        if obstacle['type'] in ['car', 'truck', 'bus', 'cyclist']:
            return self.update_kinematic_obstacle(obstacle, dt)
        elif obstacle['type'] == 'pedestrian':
            return self.update_pedestrian(obstacle, dt)
        elif obstacle['type'] == 'generic':
            return self.update_generic(obstacle, dt)
        return obstacle

    def publish_obstacles(self):
        marker_array = MarkerArray()
        object_array = ObjectArray()
        object_array.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)

        points = []

        new_obstacles = []
        for i, obstacle in enumerate(self.obstacles):
            # Check if the obstacle should be dropped based on dropout percentage
            if random.uniform(0, 100) < self.dropout_percentage:
                continue  # Skip publishing this obstacle

            # Update obstacle position and check ROI
            obstacle = self.update_obstacle(obstacle, self.dt)
            if not self.is_within_roi(obstacle['position']):
                if self.respawn:
                    obstacle = self.respawn_obstacle(obstacle)
                else:
                    continue

            new_obstacles.append(obstacle)

            # Add point for PointCloud
            points.append(obstacle['position'])

            # Create a Marker for visualization
            marker = Marker()
            marker.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
            marker.ns = 'fake_obstacles'
            marker.id = i
            marker.type = Marker.CUBE if obstacle['type'] in ['car', 'truck', 'bus', 'cyclist'] else Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = obstacle['position']
            marker.pose.orientation.w = math.cos(obstacle['orientation'] / 2.0)
            marker.pose.orientation.z = math.sin(obstacle['orientation'] / 2.0)
            marker.scale.x, marker.scale.y, marker.scale.z = obstacle['size']
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = {
                'car': (0.0, 0.0, 1.0, 1.0),
                'truck': (0.5, 0.5, 0.5, 1.0),
                'bus': (1.0, 1.0, 0.0, 1.0),
                'pedestrian': (1.0, 0.0, 0.0, 1.0),
                'cyclist': (0.0, 1.0, 0.0, 1.0)
            }.get(obstacle['type'], (1.0, 1.0, 1.0, 1.0))

            marker_array.markers.append(marker)

            # Create an Object for state publishing
            obj = Object()
            obj.id = i
            """
            * https://docs.ros.org/en/melodic/api/derived_object_msgs/html/msg/ObjectArray.html 
            * https://docs.ros.org/en/jazzy/p/derived_object_msgs/interfaces/msg/Object.html
            """
            obj.detection_level = Object.OBJECT_TRACKED  # OBJECT_DETECTED or Object.OBJECT_TRACKED
            obj.object_classified = True
            obj.classification = Object.CLASSIFICATION_PEDESTRIAN if obstacle['type'] == 'pedestrian' else Object.CLASSIFICATION_CAR
            obj.classification_certainty = int(255)

            obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = obstacle['position']
            obj.pose.orientation.w = math.cos(obstacle['orientation'] / 2.0)
            obj.pose.orientation.z = math.sin(obstacle['orientation'] / 2.0)

            obj.twist.linear.x, obj.twist.linear.y, obj.twist.linear.z = obstacle['velocity']
            obj.twist.angular.x, obj.twist.angular.y, obj.twist.angular.z = obstacle['angular_velocity']

            obj.accel.linear.x, obj.accel.linear.y, obj.accel.linear.z = 0.0, 0.0, 0.0
            obj.accel.angular.x, obj.accel.angular.y, obj.accel.angular.z = 0.0, 0.0, 0.0

            obj.shape.type = SolidPrimitive().BOX
            obj.shape.dimensions = obstacle['size']

            object_array.objects.append(obj)

            # Update path and publish it
            self.update_and_publish_path(i, obstacle)

        self.obstacles = new_obstacles

        # Publish marker array and object array
        self.obstacles_pub.publish(marker_array)
        self.object_array_pub.publish(object_array)

        # Publish point cloud
        self.publish_pointcloud(points)

    def publish_pointcloud(self, points):
        cloud = PointCloud2()
        cloud.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)

        cloud.height = 1  # Unordered point cloud
        cloud.width = len(points)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12  # 3 fields * 4 bytes/field
        cloud.row_step = cloud.point_step * len(points)
        cloud.is_dense = True  # Assume no invalid points

        # Pack points into the data field
        buffer = []
        for p in points:
            buffer.append(struct.pack('fff', *p))
        cloud.data = b''.join(buffer)

        self.pointcloud_pub.publish(cloud)

    def update_and_publish_path(self, obstacle_id, obstacle):
        # Create a new PoseStamped for the current position
        pose_stamped = PoseStamped()
        pose_stamped.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
        pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = obstacle['position']
        pose_stamped.pose.orientation.w = math.cos(obstacle['orientation'] / 2.0)
        pose_stamped.pose.orientation.z = math.sin(obstacle['orientation'] / 2.0)

        # Append to the path for this obstacle
        self.paths[obstacle_id].append(pose_stamped)

        # Create a Path message and publish
        path_msg = Path()
        path_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
        path_msg.poses = self.paths[obstacle_id]

        self.path_pubs[obstacle_id].publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
