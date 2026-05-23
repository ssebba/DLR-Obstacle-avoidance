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
import threading
import time


class Trainer(Node):
    def __init__(self):
        super().__init__('trainer')
        
        # Node parameters
        self.declare_parameter('control_frequency', 10) 
        self.declare_parameter('collision_tol', 0.15)  # 15-25 cm
        self.declare_parameter('linear_velocity',0.2) # define constant linear speed
        self.declare_parameter('num_lidar_ranges',50) # how many values for lidar data
        self.declare_parameter('lidar_max_range',5.0) # maximum range of the lidar

        self.control_freq = self.get_parameter('control_frequency').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.collision_tol = (self.get_parameter('collision_tol').value)/self.lidar_max_range # normalize collision tolerance to lidar max range
        self.linear_velocity = self.get_parameter('linear_velocity').value

        # Parameters for DRL
        self.declare_parameter('action_size', 11) #number of options (actions) the robot can select
        self.declare_parameter('gamma',0.99) # weight of future prizes
        self.declare_parameter('epsilon',1.0) # Initial epsilon
        self.declare_parameter('epsilon_min',0.05) # minimum epsilon
        self.declare_parameter('beta',0.999) # beta factor
        self.declare_parameter('batch_size',128) # batch dimension 
        self.declare_parameter('target_update_freq',5000) # after how many steps we update the target network

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
            1
        )
        
        # Publisher
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            1
        )

        # Clients
        self.reset_client = self.create_client(Trigger, '/randomize_robot_pose') # to reset the robot
        self.pause_physics_client = self.create_client(Empty, '/pause_physics') # to stop simulation
        self.unpause_physics_client = self.create_client(Empty, '/unpause_physics') # to resume simulation  

        # Initialize metrics and state
        self.step_count = 0 #steps counter for each episode
        self.total_step_count = 0 #total steps counter
        self.epoch_count = 1 #number of episodes 
        self.episode_reward = 0.0 #total reward for the episode
        self.feedback_rate = 50 #print feedback every 50 steps

        # initialize robot
        self.navigation_active = True #to check if the robot is active 
        self.stop_flag = False #to stop the robot
        self.state = None #current state of the robot
        self.previous_state = None #previous state of the robot
        self.previous_action = None #previous action of the robot
        self.is_resetting = False #to check if the robot is resetting
        self.skip_lidar_scans = 0 #to skip lidar scans 

        # initialize Neural network
        self.memory = deque(maxlen=100000) #memory to store past experiences, minibatch will sample from here
        
        #model_path = Path.home() / "ros_ws" / "models" / "trained_model.h5"
        model_path = Path.home() / "ros_ws" / "models" / "trained_model.keras"
        metadata_path = Path.home() / "ros_ws" / "models" / "training_metadata.json"
        

        # Load or create a neural network 

        # if os.path.exists(model_path): # load the neural network 
        #     self.model = tf.keras.models.load_model(model_path)
        #     self.target_model = tf.keras.models.load_model(model_path)
        #     self.get_logger().info('Trovato un modello pre-addestrato! Caricamento in corso...')
        if os.path.exists(model_path): # load the neural network
            # self.model = tf.keras.models.load_model(model_path, compile=False)
            # self.target_model = tf.keras.models.load_model(model_path, compile=False)
            # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            # self.target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            # self.get_logger().info('Found previous model! Loading and compiling...')
            self.model = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
            self.get_logger().info('Found previous model! Loading and compiling...')
            if os.path.exists(metadata_path): #load epoch count and epsilon from previous stopped training
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.epoch_count = metadata.get('epoch_count', 0)
                    self.epsilon = metadata.get('epsilon', self.get_parameter('epsilon').value)
                self.get_logger().info(f'Training resumed: Episode {self.epoch_count}, Epsilon {self.epsilon:.3f}')
            mode = 'a'
        else: # create a new neural network
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.update_target_model() #at first the two networks has to be the same
            mode = 'w'
        
        # CSV Logging to save reward of episode and average value of Q
        csv_path = Path.home() / "ros_ws" / "models" / "training_log.csv"
        file_exists = os.path.isfile(csv_path)
        self.csv_file = open(csv_path, mode=mode, newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if mode == 'w' or not file_exists:
            self.csv_writer.writerow(['Episode', 'Total_Reward', 'Avg_Q_Value', 'Steps'])
        self.episode_q_values = []
        
        # Start background training thread
        self.train_step_counter = 0
        self.memory_lock = threading.Lock()
        self.train_thread = threading.Thread(target=self.training_loop)
        self.train_thread.daemon = True
        self.train_thread.start()

        # Log that the trainer was initialized
        self.get_logger().info('Trainer initialized')

    def build_model(self):  # function to create the a new neural network
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.get_parameter('num_lidar_ranges').value,)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def update_target_model(self): # function to update the target network with the weights of the main network
        self.target_model.set_weights(self.model.get_weights())

    def training_loop(self):
        """Continuously train the network in the background"""
        while rclpy.ok():
            if not self.is_resetting:
                with self.memory_lock:
                    mem_len = len(self.memory)
                if mem_len >= self.batch_size:
                    self.train_model()
                    self.train_step_counter += 1
                    
                    if self.train_step_counter % self.target_update_freq == 0:
                        self.target_model.set_weights(self.model.get_weights())
                        self.get_logger().info('Target Network Updated!')
            time.sleep(0.01)

    def train_model(self): # function to train the neural network 
        with self.memory_lock:
            if len(self.memory) < self.batch_size: #check if there are enough experiences in the memory to train the network
                return
            minibatch = random.sample(list(self.memory), self.batch_size) # random sample from the memory
        states = np.vstack([x[0] for x in minibatch]) # stack the states
        actions = np.array([x[1] for x in minibatch]) # stack the actions
        rewards = np.array([x[2] for x in minibatch]) # stack the rewards
        next_states = np.vstack([x[3] for x in minibatch]) # stack the next states
        dones = np.array([x[4] for x in minibatch]) # stack the dones

        # Predict from main network and find index of best action
        #next_q_values_main = self.model.predict(next_states, verbose=0)
        next_q_values_main = self.model(next_states, training=False).numpy()
        best_next_actions = np.argmax(next_q_values_main, axis=1)
        
        # Predict from target network 
        #next_q_values_target = self.target_model.predict(next_states, verbose=0)
        next_q_values_target = self.target_model(next_states, training=False).numpy()

        # apply the formula of the paper (vectorized with NumPy for speed)
        batch_indices = np.arange(self.batch_size)
        
        # Calculate the discounted future reward
        # y_i = r_i+1 + gamma * Q_target(s_i+1, argmax(Q_main))
        updates = np.where(
            dones,
            rewards, # If collision, target is just the reward
            rewards + self.gamma * next_q_values_target[batch_indices, best_next_actions] # Otherwise add discounted future reward
        )
        updates = tf.convert_to_tensor(updates, dtype=tf.float32)
        
        # Use GradientTape to compute gradients ONLY for the actions taken
        with tf.GradientTape() as tape:
            # Calcola i Q-value attuali per tutte le azioni
            q_values = self.model(states, training=True)
            
            # Estrai solo i Q-value delle azioni che il robot ha effettivamente preso
            action_indices = tf.stack([batch_indices, actions], axis=1)
            q_action = tf.gather_nd(q_values, action_indices)
            
            # Calcola la loss MSE solo ed esclusivamente tra l'azione presa e il target
            loss = tf.keras.losses.MSE(updates, q_action)
            
        # Applica i gradienti
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def scan_callback(self, msg: Float32MultiArray):
        """Callback for LiDAR readings"""
        if self.skip_lidar_scans > 0:
            self.skip_lidar_scans -= 1
            return
            
        if self.is_resetting: # if the robot is resetting avoid to take action
            return
            
        req = Empty.Request()
        #self.pause_physics_client.call_async(req) # pause gazebo, it's needed to avoid the robot to move while executing control actions

        self.state = np.array(msg.data)/self.lidar_max_range
        self.state = self.state.reshape(1, len(self.state)) # reshape the state to be a 2D array instead of a vector
        
        self.control_loop_callback() # execute the control loop 
        
        #self.unpause_physics_client.call_async(req) # unpause gazebo
    
    
    def check_collision(self, distances) -> bool:
        """
        Check if the robot is too close to an obstacle
        
        Input: LaserScan message
        Output: True if the collision is close
        """

        if distances.size == 0:
            return False
        
        min_range = np.min(distances) # take the minimum range to check for collision
        collision_threshold = self.collision_tol

        if min_range < collision_threshold: # if the minimum range is less than the threshold -> collision
            self.get_logger().warn(f'Collision detected! Min range: {min_range*self.lidar_max_range:.3f}m')
            self.stop_flag = True
            return True
        
        return False
    
    def reset_simulation(self):
        """
        Resets the robot to inizial state in Gazebo environment
        """
        if not self.reset_client.wait_for_service(timeout_sec=1.0): # wait for the reset service to be available
            self.get_logger().info("In attesa del servizio /randomize_robot_pose")
            return

        request = Trigger.Request()
        future = self.reset_client.call_async(request)      # sends request to reset the robot to the server (another node)
        future.add_done_callback(self.reset_done_callback)    # after that execute the reset_done_callback function

        # Log on the csv file the reward and mean q values of the episode before resetting
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

        if self.epoch_count == 3000: # save the final model and stop the training
            #save_model_path_final = Path.home() / "ros_ws" / "models" / "trained_model_FINAL.h5"
            #save_model_path = Path.home() / "ros_ws" / "models" / "trained_model.h5"
            save_model_path_final = Path.home() / "ros_ws" / "models" / "trained_model_FINAL.keras"
            save_model_path = Path.home() / "ros_ws" / "models" / "trained_model.keras"
            metadata_path = Path.home() / "ros_ws" / "models" / "training_metadata.json"
            self.model.save(save_model_path_final)
            self.get_logger().info(f'Raggiunti 3000 episodi. Salvataggio FINAL model e chiusura totale.')
            os.system('killall -9 gzserver gzclient > /dev/null 2>&1')
            os.system('killall -9 filter_lidar spawn_entity.py respawner > /dev/null 2>&1')
            sys.exit(0)

        if self.epoch_count % 50 == 0:  # save the model every 50 epoch
            #save_model_path = Path.home() / "ros_ws" / "models" / "trained_model.h5"
            save_model_path = Path.home() / "ros_ws" / "models" / "trained_model.keras"
            metadata_path = Path.home() / "ros_ws" / "models" / "training_metadata.json"
            self.model.save(save_model_path)
            with open(metadata_path, 'w') as f:
                json.dump({'epoch_count': self.epoch_count, 'epsilon': self.epsilon}, f)
            self.get_logger().info(f'Modello e metadati salvati all\'episodio {self.epoch_count}!')

    def reset_done_callback(self, future):
        '''
        This function is called after the reset service is called
        It sets the state to None, the skip_lidar_scans to 15 and the is_resetting to False
        '''
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
        
        except Exception as e:
            self.get_logger().error(f'Impossible to reset the robot: {e}')

        

    def control_loop_callback(self):
        """
        Callback of the timer for the DWA control loop
        """

        if self.state is None or not self.navigation_active:
            return

        if self.is_resetting:
            self.reset_simulation()
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
            with self.memory_lock:
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
        #q_values = self.model.predict(self.state, verbose=0)
        q_values = self.model(self.state, training=False).numpy()
        self.episode_q_values.append(float(np.max(q_values[0])))

        if random.random() < self.epsilon:  
            m = random.randint(0, self.action_size -1) # choose a random index foraction with probability epsilon
        else:
            m = int(np.argmax(q_values[0])) # could write q_values without index, but better specify it 
        omega_m = -0.8 + 0.16 * m   # find angular velocity of the robot
        
        # 5. Publish action
        cmd_msg = Twist()
        cmd_msg.linear.x = self.linear_velocity
        cmd_msg.angular.z = float(omega_m)
        self.cmd_vel_publisher.publish(cmd_msg)

        # 6. Train the model
        self.previous_state = self.state.copy()
        self.previous_action = m
        # self.train_model() # Moved to background thread

        # 7. Update the neural network
        # Moved to background thread

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
