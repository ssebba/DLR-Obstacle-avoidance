import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data

class lidar_filter(Node):
    
    def __init__(self):
        super().__init__("lidar_filter")
        self.publisher_scans = self.create_publisher(Float32MultiArray, '/lidar_data', 10)        

        self.declare_parameter('lidar_max_range', 3.5)
        self.declare_parameter('num_lidar_ranges', 20)

        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.num_lidar_ranges = self.get_parameter('num_lidar_ranges').value
        
        self.subscription = self.create_subscription(
            LaserScan, '/scan',self.scan_callback,qos_profile_sensor_data)

        self.get_logger().info('Filtro dati lidar inizializzato')
        self.step_count = 0
    

    def scan_callback(self, msg: LaserScan):
        """Callback for LiDAR readings"""
        lidar_scan = msg
        out_data = Float32MultiArray()
        ranges = np.array(lidar_scan.ranges)
        
        # 1. Remove NaN e Inf
        ranges[np.isnan(ranges)] = lidar_scan.range_min if lidar_scan.range_min > 0 else 0.01 # Assegna max per NaN  (prima era scan.range_max)
        ranges[np.isinf(ranges)] = lidar_scan.range_max  # Assegna max per Inf
        
        # 2. Limit to max_range
        ranges[ranges > self.lidar_max_range] = self.lidar_max_range
        # 3. Reduce ranges # from 270 to num_lidar_ranges
        # Pick the minimum for each angular sector
        total_ranges = len(ranges)
        #indices = np.linspace(0, total_ranges - 1, self.num_lidar_ranges, dtype=int)
        sector_size = total_ranges // self.num_lidar_ranges

        filtered_ranges = np.zeros(self.num_lidar_ranges)
        filtered_angles = np.zeros(self.num_lidar_ranges)
        
        
        # for i in range(self.num_lidar_ranges):
        #     start_idx = i * sector_size
        #     end_idx = start_idx + sector_size if i < self.num_lidar_ranges - 1 else total_ranges
        #     filtered_ranges[i] = np.min(ranges[start_idx:end_idx])

        for i in range(self.num_lidar_ranges):
            start_idx = i * sector_size
            # Gestione dell'ultimo settore per coprire eventuali resti
            end_idx = start_idx + sector_size if i < self.num_lidar_ranges - 1 else total_ranges
            
            # Extract the sector
            sector = ranges[start_idx:end_idx]
            
            # Find the index related to the min value of the sector
            min_relative_idx = np.argmin(sector)
            
            # Compute the absolute index in the original array
            min_absolute_idx = start_idx + min_relative_idx
            
            # Save the min value
            filtered_ranges[i] = sector[min_relative_idx]
            
            # Compute the exact angle using the absolute index
            # angle = angle_min + index * angle_increment
            filtered_angles[i] = lidar_scan.angle_min + min_absolute_idx * lidar_scan.angle_increment

        out_data.data = [float(x) for x in filtered_ranges]
        if self.step_count % 100 == 0:
            self.step_count = 0
            self.get_logger().info('I am publishing!')

        self.publisher_scans.publish(out_data)
        self.step_count += 1


def main(args=None):
    rclpy.init(args=args)

    contr = lidar_filter()

    rclpy.spin(contr)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    contr.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()