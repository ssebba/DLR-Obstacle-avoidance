import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import csv
import time
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger, Empty
from rclpy.qos import qos_profile_sensor_data

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
# Limit use of GPU to avoid crashes
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        
        # Node parameters
        self.declare_parameter('collision_tol', 0.2)  # 15-25 cm
        self.declare_parameter('linear_velocity',0.2) # define constant linear speed
        self.declare_parameter('lidar_max_range',5.0)
        self.declare_parameter('max_steps', 19800)  # 10 min * 60 s * 33 Hz

        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.collision_tol = self.get_parameter('collision_tol').value/self.lidar_max_range
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.max_steps = self.get_parameter('max_steps').value

        # Subscribers
        self.scan_subscription = self.create_subscription(
            Float32MultiArray,
            '/lidar_data',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        # self.odom_subscription = self.create_subscription(
        #     Odometry,
        #     '/odom',
        #     self.odom_callback,
        #     10
        # )
        
        # Publisher
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.min_dist_publisher = self.create_publisher(
            Float32,
            '/min_lidar_distance',
            10
        )


        # Service clients
        self.respawn_client = self.create_client(Trigger, '/randomize_robot_pose')
        self.pause_physics_client = self.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = self.create_client(Empty, '/unpause_physics')

        # Metrics and state
        self.step_count = 0
        self.collision_count = 0
        self.feedback_rate = 2000
        self.test_finished = False

        # CSV logging
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.csv_path = f'/home/seba/ros_ws/logs/test_{timestamp}.csv'
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['step', 'min_distance_m', 'action', 'omega'])
        self.get_logger().info(f'Logging CSV su: {self.csv_path}')

        # load trained model
        #self.model = tf.keras.models.load_model('/home/seba/ros_ws/models/best_model_31_05.keras', compile=False)
        self.model = tf.keras.models.load_model('/home/seba/ros_ws/models/best_model_04_06.keras', compile=False)

        self.navigation_active = True
        self.stop_flag = False
        self.is_resetting = False
        self.skip_lidar_scans = 0
        self.state = None
        
        self.get_logger().info(
            f'Controller inizializzato in modalità TEST: '
            f'max_steps={self.max_steps} (~{self.max_steps/33/60:.1f} min)'
        )
    
    def scan_callback(self, msg: Float32MultiArray):
        """Callback for LiDAR readings"""
        if self.skip_lidar_scans > 0:
            self.skip_lidar_scans -= 1
            return
        if self.is_resetting:
            return

        self.state = np.array(msg.data) / self.lidar_max_range
        self.state = self.state.reshape(1, len(self.state))
        self.control_loop_callback()
    
    # def odom_callback(self, msg: Odometry):
    #     """Callback for robot odometry"""
    #     # Extract position
    #     x = msg.pose.pose.position.x
    #     y = msg.pose.pose.position.y
        
    #     # Extract orientation (quaternione -> euler)
    #     quat = msg.pose.pose.orientation
    #     _, _, theta = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        
    #     self.robot_pose = np.array([x, y, theta])
        
    #     # Estrai velocità
    #     v = msg.twist.twist.linear.x
    #     w = msg.twist.twist.angular.z
    #     self.robot_vel = np.array([v, w])
    
    
    def check_collision(self, distances) -> bool:
        """
        Check if the robot is too close to an obstacle
        
        Input: LaserScan message
        Output: True if the collision is close
        """

        if distances.size == 0:
            return False
        
        min_range = np.min(distances)
        collision_threshold = self.collision_tol

        if min_range < collision_threshold:
            self.get_logger().warn(f'Collisione rilevata! Min range: {min_range*self.lidar_max_range:.3f}m')
            self.stop_flag = True
            return True
        
        return False
        

    def respawn_robot(self):
        """Pause physics, then call /randomize_robot_pose to respawn the robot"""
        self.is_resetting = True
        self.pause_physics_client.call_async(Empty.Request())

        if not self.respawn_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Servizio /randomize_robot_pose non disponibile!')
            self.is_resetting = False
            return
        
        request = Trigger.Request()
        future = self.respawn_client.call_async(request)
        future.add_done_callback(self._respawn_done_callback)

    def _respawn_done_callback(self, future):
        """Callback when respawn service call completes"""
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'Respawn riuscito: {result.message}')
            else:
                self.get_logger().warn(f'Respawn fallito: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Errore nella chiamata respawn: {e}')
        
        # Resume navigation after respawn
        self.stop_flag = False
        self.state = None
        self.skip_lidar_scans = 15  # discard stale lidar data after respawn
        self.is_resetting = False
        self.unpause_physics_client.call_async(Empty.Request())

    def finish_test(self):
        """Stop the test and log collision results"""
        self.stop_robot()
        self.test_finished = True
        self.navigation_active = False
        # Close CSV
        self.csv_file.close()
        self.get_logger().info('=' * 60)
        self.get_logger().info('TEST COMPLETATO')
        self.get_logger().info(f'Step totali: {self.step_count}')
        self.get_logger().info(f'Collisioni totali: {self.collision_count}')
        self.get_logger().info(f'CSV salvato in: {self.csv_path}')
        self.get_logger().info('=' * 60)

    def control_loop_callback(self):
        """
        Callback of the timer for the DWA control loop
        """
        
        if self.state is None:
            return

        if self.test_finished:
            return

        if not self.navigation_active:
            return
        
        # Check if test duration has been reached
        if self.step_count >= self.max_steps:
            self.finish_test()
            return

        # 1. Check for collision
        if self.check_collision(self.state) or self.stop_flag:
            self.stop_robot()
            self.collision_count += 1
            self.get_logger().warn(
                f'Collisione #{self.collision_count} allo step {self.step_count}. Respawn in corso...'
            )
            self.respawn_robot()
            return
        
        q_values = self.model(self.state, training=False).numpy()
        # 3. Policy Greedy: Seleziona l'azione con il valore Q massimo 
        # Restituisce l'indice 'm' compreso tra 0 e 10
        m = np.argmax(q_values[0])

        # 4. Calcola la velocità angolare in base all'equazione del paper 
        omega_m = -0.8 + 0.16 * m
        
        # 7. CMD VEL PUBBLICATION
        cmd_msg = Twist()
        cmd_msg.linear.x = self.linear_velocity
        cmd_msg.angular.z = float(omega_m)
        self.cmd_vel_publisher.publish(cmd_msg)

        min_distance_meters = float(np.min(self.state) * self.lidar_max_range)
        msg = Float32()
        msg.data = min_distance_meters
        self.min_dist_publisher.publish(msg)

        # CSV logging
        self.csv_writer.writerow([self.step_count, f'{min_distance_meters:.4f}', m, f'{omega_m:.2f}'])
        
        # 8. Periodic feedback
        if self.step_count % self.feedback_rate == 0:
            self.csv_file.flush()  # flush periodico per sicurezza
            self.get_logger().info(
                f'Step {self.step_count} | '
            )
            
        
        self.step_count += 1
    
    def stop_robot(self):
        """Stop the robot"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info('Robot fermato')





def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
