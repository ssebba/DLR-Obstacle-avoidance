import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
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
        self.declare_parameter('control_frequency', 5) 
        self.declare_parameter('collision_tol', 0.20)  # 15-25 cm
        self.declare_parameter('linear_velocity',0.2) # define constant linear speed

        self.control_freq = self.get_parameter('control_frequency').value
        self.collision_tol = self.get_parameter('collision_tol').value
        self.linear_velocity = self.get_parameter('linear_velocity').value

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

        # Timer for the control loop
        self.timer = self.create_timer(self.control_freq, self.control_loop_callback)
        
        # Metrics and state
        self.step_count = 0
        self.feedback_rate = 50

        # load trained model
        self.model = tf.keras.models.load_model('/home/seba/ros_ws/models/dummy_model.h5')

        self.navigation_active = True
        self.stop_flag = False
        self.timeout_flag = False
        self.state = None
        
        self.get_logger().info('Controller inizializzato')
    
    def scan_callback(self, msg: Float32MultiArray):
        """Callback for LiDAR readings"""
        self.state = np.array(msg.data)
        self.state = self.state.reshape(1, len(self.state))
    
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
            self.get_logger().warn(f'Collisione rilevata! Min range: {min_range:.3f}m')
            self.stop_flag = True
            return True
        
        return False
        




    def control_loop_callback(self):
        """
        Callback of the timer for the DWA control loop
        """
        
        if self.state is None:
            return

        if not self.navigation_active:
            return
        
        # 1. Check for collision
        if self.check_collision(self.state) or self.stop_flag:
            self.stop_robot()
            self.get_logger().error('Task fallito: COLLISIONE')
            return
        

        
        # 5. Verify timeout
        if self.step_count > 300 or self.timeout_flag:
            self.stop_robot()
            self.timeout_flag = True
            self.get_logger().warn('Task fallito: TIMEOUT')
            return
        
        q_values = self.model.predict(self.state, verbose=0)
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

        self.get_logger().info('1 Step done')
        
        # 8. Periodic feedback
        if self.step_count % self.feedback_rate == 0:
            
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
