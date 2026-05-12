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
# for the RML
import os
import json
from pathlib import Path
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
from std_srvs.srv import Trigger, Empty
import random
from collections import deque
import csv
import sys


class Trainer(Node):
    def __init__(self):
        super().__init__('trainer')
        
        # Node parameters
        self.declare_parameter('control_frequency', 10) 
        self.declare_parameter('collision_tol', 0.15)  # 15-25 cm
        self.declare_parameter('linear_velocity',0.2) # define constant linear speed
        self.declare_parameter('num_lidar_ranges',50)

        self.control_freq = self.get_parameter('control_frequency').value
        self.collision_tol = self.get_parameter('collision_tol').value
        self.linear_velocity = self.get_parameter('linear_velocity').value

        # Parameters for DRL
        self.declare_parameter('action_size', 11) #number of option the robot can select
        self.declare_parameter('gamma',0.99) # weight of future prizes
        self.declare_parameter('epsilon',1.0) # Initial epsilon
        self.declare_parameter('epsilon_min',0.05) # minimum epsilon
        self.declare_parameter('beta',0.999) # beta factor
        self.declare_parameter('batch_size',64) # batch dimension 
        self.declare_parameter('target_update_freq',500)

        self.action_size = self.get_parameter('action_size').value
        self.gamma = self.get_parameter('gamma').value
        self.epsilon = self.get_parameter('epsilon').value
        self.epsilon_min = self.get_parameter('epsilon_min').value
        self.beta = self.get_parameter('beta').value
        self.batch_size = self.get_parameter('batch_size').value
        self.target_update_freq = self.get_parameter('target_update_freq').value

        # Subscribers
        self.scan_subscription = self.create_subscription(
            Float32MultiArray,
            '/lidar_data',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        # Publisher
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Clients
        self.reset_client = self.create_client(Trigger, '/randomize_robot_pose')
        self.pause_physics_client = self.create_client(Empty, '/pause_physics')
        self.unpause_physics_client = self.create_client(Empty, '/unpause_physics')

        # Metrics and state
        self.step_count = 0
        self.total_step_count = 0
        self.epoch_count = 0
        self.episode_reward = 0.0
        self.feedback_rate = 50

        # initialize robot
        self.navigation_active = True
        self.stop_flag = False
        self.state = None
        self.previous_state = None
        self.previous_action = None
        self.is_resetting = False
        self.skip_lidar_scans = 0

        # initialize Neural network
        self.memory = deque(maxlen=100000)
        
        model_path = Path.home() / "ros_ws" / "models" / "trained_model.h5"
        metadata_path = Path.home() / "ros_ws" / "models" / "training_metadata.json"
        
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
            self.get_logger().info('Trovato un modello pre-addestrato! Caricamento in corso...')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.epoch_count = metadata.get('epoch_count', 0)
                    self.epsilon = metadata.get('epsilon', self.get_parameter('epsilon').value)
                self.get_logger().info(f'Ripresa dal checkpoint: Episodio {self.epoch_count}, Epsilon {self.epsilon:.3f}')
            
            mode = 'a'
        else:
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.update_target_model() #at first the two networks has to be the same
            mode = 'w'
        
        # CSV Logging
        csv_path = Path.home() / "ros_ws" / "models" / "training_log.csv"
        file_exists = os.path.isfile(csv_path)
        self.csv_file = open(csv_path, mode=mode, newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if mode == 'w' or not file_exists:
            self.csv_writer.writerow(['Episode', 'Total_Reward', 'Avg_Q_Value', 'Steps'])
        self.episode_q_values = []
        
        self.get_logger().info('Controller inizializzato')

    def build_model(self):
        
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.get_parameter('num_lidar_ranges').value,)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.vstack([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # 2. LOGICA DDQN: Chiediamo alla Main Network quale azione farebbe nel 'next_state' [cite: 256, 260]
        next_q_values_main = self.model.predict(next_states, verbose=0)
        best_next_actions = np.argmax(next_q_values_main, axis=1)
        
        # 3. Chiediamo alla Target Network di "valutare" quell'azione [cite: 256]
        next_q_values_target = self.target_model.predict(next_states, verbose=0)
        
        # 4. Calcoliamo i Q-value attuali per poterli correggere
        target_q_values = self.model.predict(states, verbose=0)
        
        # 5. Applichiamo la formula matematica del paper per ogni ricordo nel batch
        for i in range(self.batch_size):
            if dones[i]: 
                # Se c'è stata collisione (riga 10-11 del paper) 
                target_q_values[i][actions[i]] = rewards[i] 
            else:
                # Altrimenti aggiungiamo il premio futuro scontato (gamma) (riga 12-13 del paper) 
                # yi = r_i+1 + gamma * Q_target(s_i+1, argmax(Q_main)) [cite: 253, 260]
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values_target[i][best_next_actions[i]]
                
        # 6. Addestriamo la rete (Discesa del Gradiente) sui valori corretti 
        self.model.fit(states, target_q_values, batch_size=self.batch_size, epochs=1, verbose=0)
    
    def scan_callback(self, msg: Float32MultiArray):
        """Callback for LiDAR readings"""
        if self.skip_lidar_scans > 0:
            self.skip_lidar_scans -= 1
            return
            
        if self.is_resetting:
            return
            
        req = Empty.Request()
        self.pause_physics_client.call_async(req) # pause gazebo

        self.state = np.array(msg.data)
        self.state = self.state.reshape(1, len(self.state))
        
        self.control_loop_callback() # execute the control loop 
        
        self.unpause_physics_client.call_async(req) # unpause gazebo
    
    
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
    
    def reset_simulation(self):
        """
        Resets the robot to inizial state in Gazebo environment
        """
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("In attesa del servizio /randomize_robot_pose")
            self.is_resetting = False
            return

        request = Trigger.Request()
        future = self.reset_client.call_async(request)      # sends request to reset the robot
        future.add_done_callback(self.reset_done_callback)    # after that execute the reset_done_callback function

        # Log the finished episode stats before resetting
        avg_q = float(np.mean(self.episode_q_values)) if self.episode_q_values else 0.0
        self.csv_writer.writerow([self.epoch_count, self.episode_reward, avg_q, self.step_count])
        self.csv_file.flush()
        self.episode_q_values = []

        self.previous_state = None  # clear previous state
        self.previous_action = None # clear previous action
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.beta   # update epsiolon with beta factor
        self.epoch_count += 1
        self.episode_reward = 0.0   # reset reward for the new episode

        if self.epoch_count == 3000:
            save_model_path_final = Path.home() / "ros_ws" / "models" / "trained_model_FINAL.h5"
            save_model_path = Path.home() / "ros_ws" / "models" / "trained_model.h5"
            metadata_path = Path.home() / "ros_ws" / "models" / "training_metadata.json"
            self.model.save(save_model_path_final)
            self.get_logger().info(f'Raggiunti 3000 episodi. Salvataggio FINAL model e chiusura totale.')
            os.system('killall -9 gzserver gzclient > /dev/null 2>&1')
            os.system('killall -9 filter_lidar spawn_entity.py respawner > /dev/null 2>&1')
            sys.exit(0)

        if self.epoch_count % 50 == 0:  # save the model every 50 epoch
            save_model_path = Path.home() / "ros_ws" / "models" / "trained_model.h5"
            metadata_path = Path.home() / "ros_ws" / "models" / "training_metadata.json"
            self.model.save(save_model_path)
            with open(metadata_path, 'w') as f:
                json.dump({'epoch_count': self.epoch_count, 'epsilon': self.epsilon}, f)
            self.get_logger().info(f'Modello e metadati salvati all\'episodio {self.epoch_count}!')

    def reset_done_callback(self, future):
        # try:
        #     future.result()
        #     self.get_logger().info('Reset succeded!')
        #     self.stop_flag = False
        #     self.step_count = 0
        #     self.is_resetting = False

        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Reset succeded! {response.message}')
                self.stop_flag = False
                self.step_count = 0
                self.state = None  
                self.skip_lidar_scans = 15  # Ignore next 15 scans to let physics and buffers settle
                self.is_resetting = False
            else:
                self.get_logger().error(f'Reset failed: {response.message}')
                self.is_resetting = False
        
        except Exception as e:
            self.get_logger().error(f'Impossible to reset the robot: {e}')
            self.is_resetting = False

        

    def control_loop_callback(self):
        """
        Callback of the timer for the DWA control loop
        """

        if self.state is None or not self.navigation_active or self.is_resetting:
            return

        # 1. Check for collision and assign reward for this step
        if self.check_collision(self.state) or self.stop_flag:
            reward = -1000
            collision = True
        else:
            reward = 5 
            collision = False
        self.episode_reward += reward

        # 2. Add to the memory this iteration step
        if self.previous_state is not None and self.previous_action is not None:
            self.memory.append((self.previous_state, self.previous_action, reward, self.state, collision))

        # 3. Reset robot if collision or timeout achieved 
        if self.step_count > 3000 or collision:
            self.stop_robot()  # assign to the robot 0 linear and angular speed 
            self.is_resetting = True
            err = 'COLLISION' if collision else 'TIMEOUT'
            self.get_logger().error(f'Episode {self.epoch_count} finished: {err}. Total reward: {self.episode_reward} Resetting the robot...')
            
            self.reset_simulation() # resets the robot pose
            self.total_step_count += 1
            return

        # 4. Select action of the robot
        q_values = self.model.predict(self.state, verbose=0)
        self.episode_q_values.append(float(np.max(q_values[0])))

        if random.random() < self.epsilon:  
            m = random.randint(0, self.action_size -1) # choose a random action with probability epsilon
            self.previous_state = self.state.copy()
        else:
            m = int(np.argmax(q_values[0]))
            self.previous_state = self.state.copy()
        omega_m = -0.8 + 0.16 * m   # find angular velocity of the robot
        
        # 5. Publish action
        cmd_msg = Twist()
        cmd_msg.linear.x = self.linear_velocity
        cmd_msg.angular.z = float(omega_m)
        self.cmd_vel_publisher.publish(cmd_msg)

        # 6. Train the model
        self.previous_action = m
        self.train_model()

        # 7. Update the neural network
        if self.total_step_count % self.target_update_freq == 0 and self.total_step_count != 0:
            self.target_model.set_weights(self.model.get_weights())
            self.get_logger().info('Target Network Updated!')

        # 8. Keep track of number of steps
        self.step_count += 1
        self.total_step_count += 1
    
    def stop_robot(self):
        """Stop the robot"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info('Robot stopped')


def main(args=None):
    rclpy.init(args=args)
    node = Trainer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
