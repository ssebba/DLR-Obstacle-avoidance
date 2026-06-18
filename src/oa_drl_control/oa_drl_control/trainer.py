#!/usr/bin/env python3
"""
Trainer node for Deep Reinforcement Learning-based collision avoidance.

Implements the DDQN algorithm from:
    Feng, S.; Sebastian, B.; Ben-Tzvi, P.
    "A Collision Avoidance Method Based on Deep Reinforcement Learning"
    Robotics 2021, 10, 73.

Enhancements over the baseline (see differences.md):
  - Synchronous training: one gradient step per env step (paper Algorithm 1)
  - Correct loss: L = (1/2n) * sum(y_i - Q_i)^2            (paper Eq. 5)
  - Target network updated every N *environment* steps      (paper Alg. 1 l.16)
  - Best-model checkpoint: saved when smoothed avg-Q improves, not every N epochs
  - Early stopping with patience parameter
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger, Empty
import numpy as np
import tensorflow as tf
import os
import json

import random
import csv
import queue
import sys
import threading
from collections import deque
from pathlib import Path

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# ── GPU: allow memory growth to prevent OOM crashes ───────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║           HYPERPARAMETERS — edit this section to configure training         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Fixed parameters (paper-specified) ─────────────
ACTION_SIZE            = 11      # number of discrete angular-velocity actions
NUM_LIDAR_RANGES       = 50      # pre-processed LIDAR readings  (input dim)
LIDAR_MAX_RANGE        = 5.0     # sensor max range, metres
EPSILON_INITIAL        = 1.0     # starting ε for ε-greedy exploration
EPSILON_MIN            = 0.05    # minimum ε                       
REWARD_SAFE            = 5       # reward base per step w/o collision
REWARD_COLLISION       = -1000   # penalty on collision
HIDDEN_UNITS           = 300     # neurons per hidden layer

# ── Fixed parameters (user defined) ─────────────
# ── Exponential proximity penalty (calibrated on training_env.obj geometry) ────
# Corridors as narrow as 0.5m, median clearance 0.69m, 71% of space < 1.0m
# Collision at 0.04 norm (0.2m real); center of narrowest corridor = 0.05 norm
# Reward must stay POSITIVE at corridor center to not discourage navigation
D_SAFE                 = 0.10    # normalised threshold (= 0.5m real); penalty grows below this
PROX_K                 = 20.0    # exponential penalty intensity
CLEAR_W                = 2.0     # linear clearance bonus weight
MAX_EPOCHS             = 8000    # total training episodes
MAX_STEPS_PER_EPISODE  = 1800    # env steps before episode timeout
COLLISION_TOL          = 0.2    # collision distance threshold, metres
LINEAR_VELOCITY        = 0.2     # constant forward speed, m/s
BETA                   = 0.9996   # ε decay rate per episode
GAMMA                  = 0.99
LR                     = 0.001
BATCH_SIZE             = 256
TARGET_UPDATE_FREQ     = 1000   # env steps between target-network updates

# ── Training-control parameters ────────────────────────────────────────────────
PATIENCE               = 3000     # episodes without reward improvement → early stop
BEST_MODEL_WINDOW      = 200       # moving-average window for smoothed reward metric
MEMORY_SIZE            = 100000  # experience replay buffer capacity


# ══════════════════════════════════════════════════════════════════════════════


class Trainer(Node):
    """DDQN trainer node for obstacle avoidance via deep reinforcement learning."""

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__('trainer')

        # ── ROS2 I/O ──────────────────────────────────────────────────────────
        # Defined publishers, subscribers and clients
        self.scan_subscription = self.create_subscription(
            Float32MultiArray, '/lidar_data', self.scan_callback, 1)

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1)

        self.reset_client           = self.create_client(Trigger, '/randomize_robot_pose')
        self.pause_physics_client   = self.create_client(Empty,   '/pause_physics')
        self.unpause_physics_client = self.create_client(Empty,   '/unpause_physics')

        # ── Shared per-episode state ───────────────────────────────────────────
        #initialize the flags and the state variables
        self.navigation_active = True
        self.stop_flag         = False
        self.state             = None
        self.previous_state    = None
        self.previous_action   = None
        self.is_resetting      = False
        self.skip_lidar_scans  = 0
        self.step_count        = 0
        self.episode_reward    = 0.0
        self.episode_q_values  = []
        self._train_queue = queue.Queue()
        self._train_thread = threading.Thread(
            target=self._training_worker, daemon=True)
        self._train_thread.start()

        # ── Launch training ─────────────────────────────────────────────────────
        params = {
            'gamma':              GAMMA,
            'lr':                 LR,
            'batch_size':         BATCH_SIZE,
            'target_update_freq': TARGET_UPDATE_FREQ,
        }
        self._start_training_phase(params, force_fresh=False)

        self.get_logger().info('Trainer node initialised.')

    # ──────────────────────────────────────────────────────────────────────────
    # Neural network helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_model(self, lr: float) -> tf.keras.Model:
        """Build the two-hidden-layer Q-network."""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(NUM_LIDAR_RANGES,)),
            tf.keras.layers.Dense(HIDDEN_UNITS, activation='relu'),
            tf.keras.layers.Dense(HIDDEN_UNITS, activation='relu'),
            tf.keras.layers.Dense(ACTION_SIZE,  activation='linear'),
        ])
        # Compiled only to attach the Adam optimizer (loss is computed manually)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def _update_target_model(self):
        """Hard copy θ⁻ ← θ"""
        self.target_model.set_weights(self.model.get_weights())

    # ──────────────────────────────────────────────────────────────────────────
    # Training phase
    # ──────────────────────────────────────────────────────────────────────────

    def _start_training_phase(self, params: dict, force_fresh: bool = False):
        """Initialise (or resume) the DDQN training phase."""
        self.gamma              = params['gamma']
        self.batch_size         = params['batch_size']
        self.target_update_freq = params['target_update_freq']

        model_path = Path.home() / 'ros_ws' / 'models' / 'trained_model.keras'
        meta_path  = Path.home() / 'ros_ws' / 'models' / 'training_metadata.json'

        if not force_fresh and os.path.exists(model_path):
            # ── Resume from checkpoint ─────────────────────────────────────────
            self.model        = tf.keras.models.load_model(model_path)
            self.target_model = tf.keras.models.load_model(model_path)
            self.get_logger().info('Found saved model — resuming training...')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.epoch_count = meta.get('epoch_count', 1)
                self.epsilon     = meta.get('epsilon', EPSILON_INITIAL)
                self.best_avg_reward = meta.get('best_avg_reward', -float('inf'))
                self.patience_counter = meta.get('patience_counter', 0)
                self.get_logger().info(
                    f'  Resumed: episode={self.epoch_count}, '
                    f'epsilon={self.epsilon:.3f}, best_reward={self.best_avg_reward:.2f}'
                )
            else:
                self.epoch_count = 1
                self.epsilon     = EPSILON_INITIAL
                self.best_avg_reward = -float('inf')
                self.patience_counter = 0
            mode = 'a'
        else:
            # ── Fresh start (after selection or first-ever run) ────────────────
            self.model        = self._build_model(lr=params['lr'])
            self.target_model = self._build_model(lr=params['lr'])
            self._update_target_model()
            self.epoch_count  = 1
            self.epsilon      = EPSILON_INITIAL
            self.best_avg_reward = -float('inf')
            self.patience_counter = 0
            mode = 'w'

        self.memory           = deque(maxlen=MEMORY_SIZE)
        self.env_step_count   = 0
        self._drain_train_queue()
        self.recent_avg_reward = deque(maxlen=BEST_MODEL_WINDOW)
        self.best_model_path  = Path.home() / 'ros_ws' / 'models' / 'best_model.keras'

        # Reset episode state (is_resetting managed by reset_done_callback)
        self.step_count       = 0
        self.episode_reward   = 0.0
        self.episode_q_values = []
        self.previous_state   = None
        self.previous_action  = None
        self.stop_flag        = False
        self.state            = None

        # Open training CSV
        csv_path    = Path.home() / 'ros_ws' / 'models' / 'training_log.csv'
        file_exists = os.path.isfile(csv_path)
        self.csv_file   = open(csv_path, mode=mode, newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if mode == 'w' or not file_exists:
            self.csv_writer.writerow([
                'Episode', 'Total_Reward', 'Avg_Q_Value',
                'Steps', 'Smoothed_Reward', 'Best_Reward', 'Patience_Counter'
            ])

        self.get_logger().info(
            f'\n{"=" * 60}\n'
            f'  TRAINING PHASE\n'
            f'  gamma              = {self.gamma}\n'
            f'  learning_rate      = {params["lr"]}\n'
            f'  batch_size         = {self.batch_size}\n'
            f'  target_update_freq = {self.target_update_freq} env steps\n'
            f'  max_epochs         = {MAX_EPOCHS}\n'
            f'  patience           = {PATIENCE} episodes\n'
            f'  best_model_window  = {BEST_MODEL_WINDOW} episodes\n'
            f'  beta               = {BETA}\n'
            f'{"=" * 60}'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core DDQN training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_model(self):
        """
        One synchronous mini-batch gradient step (paper Algorithm 1, steps 9–15).

        Loss  : L(θ) = (1/2n) * Σ (y_i − Q(s_i, a_i; θ))²   [paper Eq. 5]
        Target: y_i  = r_{i+1} + γ * Q'(s_{i+1}, argmax_a Q(s_{i+1}; θ); θ⁻)
                                                               [paper Eq. 6]
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch   = random.sample(list(self.memory), self.batch_size)
        states      = np.vstack([x[0] for x in minibatch])
        actions     = np.array([x[1] for x in minibatch])
        rewards     = np.array([x[2] for x in minibatch])
        next_states = np.vstack([x[3] for x in minibatch])
        dones       = np.array([x[4] for x in minibatch])

        # DDQN: main network selects action, target network evaluates it
        next_q_main   = self.model(next_states, training=False).numpy()
        best_actions  = np.argmax(next_q_main, axis=1)
        next_q_target = self.target_model(next_states, training=False).numpy()

        batch_idx = np.arange(self.batch_size)

        # Compute target values y_i  (paper Eq. 6 / Algorithm 1, lines 10-14)
        targets = np.where(
            dones,
            rewards,                                                       # terminal
            rewards + self.gamma * next_q_target[batch_idx, best_actions]  # non-terminal
        )
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        # Gradient descent on paper Eq. 5: L(θ) = (1/2n) Σ (y − Q)²
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            act_idx  = tf.stack([batch_idx, actions], axis=1)
            q_taken  = tf.gather_nd(q_values, act_idx)
            loss     = 0.5 * tf.reduce_mean(tf.square(targets - q_taken))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

    # ──────────────────────────────────────────────────────────────────────────
    # ROS2 callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def scan_callback(self, msg: Float32MultiArray):
        """Receive pre-processed LIDAR data and trigger the control loop."""
        if self.skip_lidar_scans > 0:
            self.skip_lidar_scans -= 1
            return
        if self.is_resetting:
            return

        self.state = (np.array(msg.data) / LIDAR_MAX_RANGE).reshape(1, -1)
        self.control_loop_callback()

    def check_collision(self, state: np.ndarray) -> bool:
        """Return True if the nearest normalised range is below threshold."""
        if state.size == 0:
            return False
        threshold = COLLISION_TOL / LIDAR_MAX_RANGE
        if np.min(state) < threshold:
            self.get_logger().warn(
                f'Collision detected! Min range: '
                f'{np.min(state) * LIDAR_MAX_RANGE:.3f} m'
            )
            self.stop_flag = True
            return True
        return False

    def control_loop_callback(self):
        """
        Main DRL step — invoked once per LiDAR scan.
        Follows paper Algorithm 1 (synchronous: one gradient step per env step).
        """
        if self.state is None or not self.navigation_active:
            return
        if self.is_resetting:
            self.stop_robot()
            return

        # ── 1. Collision check → reward ────────────────────────────────────────
        collision = self.check_collision(self.state) or self.stop_flag
        if collision:
            reward = REWARD_COLLISION
        else:
            # Exponential proximity penalty + linear clearance bonus
            # Calibrated on training_env.obj: corridors 0.5–1.5m, median 0.69m
            min_distance = float(np.min(self.state))  # normalised [0, 1]

            # Base: survival reward
            base_reward = REWARD_SAFE  # +5

            # Exponential penalty: grows when min_dist < D_SAFE (0.5m real)
            # Tuned so center of narrowest corridor (0.25m) still gets positive reward
            if min_distance < D_SAFE:
                proximity_penalty = -PROX_K * np.exp(-4.0 * min_distance / D_SAFE)
            else:
                proximity_penalty = 0.0

            # Linear bonus: incentivises staying far from walls
            clearance_bonus = CLEAR_W * min_distance

            reward = base_reward + proximity_penalty + clearance_bonus
        self.episode_reward += reward

        # ── 2. Store transition (s_t, a_t, r_{t+1}, s_{t+1}) in D ─────────────
        if self.previous_state is not None and self.previous_action is not None:
            self.memory.append((
                self.previous_state, self.previous_action,
                reward, self.state, collision
            ))
    
        # ── 3. Episode end: collision or timeout ───────────────────────────────
        if collision or self.step_count > MAX_STEPS_PER_EPISODE:
            self.stop_robot()
            self.pause_physics_client.call_async(Empty.Request())
            self.is_resetting = True
            reason = 'COLLISION' if collision else 'TIMEOUT'
            self.get_logger().error(
                f'Train ep {self.epoch_count}/{MAX_EPOCHS} — {reason}. '
                f'Reward: {self.episode_reward:.0f}. Resetting...'
            )
            self.reset_simulation()
            return

        # ── 4. Env-step count & async training ──────────────────────────────
        self.env_step_count += 1
        self._train_queue.put_nowait(self.env_step_count)

        # ── 6. ε-greedy action selection ───────────────────────────────────────
        q_values = self.model(self.state, training=False).numpy()
        self.episode_q_values.append(float(np.max(q_values[0])))

        m       = (random.randint(0, ACTION_SIZE - 1)
                   if random.random() < self.epsilon
                   else int(np.argmax(q_values[0])))

        omega_m = -0.8 + 0.16 * m   # angular velocity (paper §3.4)

        # ── 7. Publish velocity command ─────────────────────────────────────────
        cmd           = Twist()
        cmd.linear.x  = LINEAR_VELOCITY
        cmd.angular.z = float(omega_m)
        self.cmd_vel_publisher.publish(cmd)

        # ── 8. Bookkeeping ──────────────────────────────────────────────────────
        self.previous_state  = self.state.copy()
        self.previous_action = m
        self.step_count     += 1

    def _training_worker(self):
        """Persistent background thread: pulls env-step signals and trains."""
        while True:
            try:
                env_step = self._train_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.train_model()
                # ── 5. Update target network every N env steps ──
                if env_step % self.target_update_freq == 0:
                    self._update_target_model()
                    self.get_logger().info(
                        f'Target network updated (env step {env_step})')
            except Exception as e:
                self.get_logger().error(f'Error in training worker: {e}')

    def _drain_train_queue(self):
        """Discard pending training signals (used on phase/candidate reset)."""
        while not self._train_queue.empty():
            try:
                self._train_queue.get_nowait()
            except queue.Empty:
                break

    # ──────────────────────────────────────────────────────────────────────────
    # Episode-end logic
    # ──────────────────────────────────────────────────────────────────────────

    def reset_simulation(self):
        """Trigger async robot-pose randomisation and run episode bookkeeping."""
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /randomize_robot_pose service...')
            return

        # Phase-specific end-of-episode logic (logging, best-model, patience…)
        self._on_episode_end()

        # Reset shared episode state
        self.episode_reward  = 0.0
        self.previous_state  = None
        self.previous_action = None

        # Request async robot-position randomisation
        future = self.reset_client.call_async(Trigger.Request())
        future.add_done_callback(self.reset_done_callback)

    def _on_episode_end(self):
        """End-of-episode bookkeeping: logging, best-model, patience."""
        avg_q = (float(np.mean(self.episode_q_values))
                 if self.episode_q_values else 0.0)
        self.episode_q_values = []
        self._on_training_episode_end(avg_q)

    def _on_training_episode_end(self, avg_q: float):
        """Log episode, track best model, check patience and termination."""
        self.recent_avg_reward.append(self.episode_reward)
        smoothed_reward = float(np.mean(self.recent_avg_reward))

        # CSV log — Total_Reward and Avg_Q_Value are kept for analysis;
        # Smoothed_Reward / Best_Reward drive best-model selection.
        self.csv_writer.writerow([
            self.epoch_count, self.episode_reward, avg_q,
            self.step_count, smoothed_reward, self.best_avg_reward, self.patience_counter
        ])
        self.csv_file.flush()

        # ── Best-model checkpoint (only when smoothed reward improves) ──────────
        if len(self.recent_avg_reward) < self.recent_avg_reward.maxlen:
            # Warmup: not enough data for a reliable smoothed-reward estimate yet
            pass
        elif smoothed_reward > self.best_avg_reward:
            self.best_avg_reward  = smoothed_reward
            self.patience_counter = 0
            self.model.save(self.best_model_path)
            self.get_logger().info(
                f'★ Best model saved! Episode {self.epoch_count}, '
                f'smoothed reward = {smoothed_reward:.2f}'
            )
        else:
            self.patience_counter += 1

        # Decay ε (paper Eq. 3: ε_{k+1} = β * ε_k)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= BETA

        # ── Periodic checkpoint every 50 episodes (for crash recovery) ─────────
        if self.epoch_count % 50 == 0:
            ckpt_path = Path.home() / 'ros_ws' / 'models' / 'trained_model.keras'
            meta_path = Path.home() / 'ros_ws' / 'models' / 'training_metadata.json'
            self.model.save(ckpt_path)
            with open(meta_path, 'w') as f:
                json.dump({
                    'epoch_count':     self.epoch_count + 1,
                    'epsilon':         self.epsilon,
                    'best_avg_reward': self.best_avg_reward,
                    'patience_counter': self.patience_counter,
                }, f)
            self.get_logger().info(
                f'Checkpoint saved — episode {self.epoch_count}, '
                f'ε={self.epsilon:.3f}, best_reward={self.best_avg_reward:.2f}'
            )

        self.epoch_count += 1

        # ── Early stopping: patience ────────────────────────────────────────────
        if self.patience_counter >= PATIENCE:
            self.get_logger().warn(
                f'Early stopping at episode {self.epoch_count - 1}: '
                f'no reward improvement for {PATIENCE} episodes. '
                f'Best smoothed reward = {self.best_avg_reward:.2f}'
            )
            self._terminate_training()

        # ── Max-epoch stop ──────────────────────────────────────────────────────
        if self.epoch_count > MAX_EPOCHS:
            self.get_logger().info(
                f'Training complete after {MAX_EPOCHS} episodes. '
                f'Best smoothed reward = {self.best_avg_reward:.2f}'
            )
            self._terminate_training()

    def _terminate_training(self):
        """Save final artefacts and shut down the simulation."""
        final_path = Path.home() / 'ros_ws' / 'models' / 'trained_model_FINAL.keras'
        meta_path  = Path.home() / 'ros_ws' / 'models' / 'training_metadata.json'

        self.model.save(final_path)
        with open(meta_path, 'w') as f:
            json.dump({
                'epoch_count':     self.epoch_count,
                'epsilon':         self.epsilon,
                'best_avg_reward': self.best_avg_reward,
                'patience_counter': self.patience_counter,
            }, f)

        self.get_logger().info(
            f'\n{"=" * 60}\n'
            f'  TRAINING TERMINATED\n'
            f'  Last-epoch model → {final_path}\n'
            f'  Best model       → {self.best_model_path}\n'
            f'  Best smoothed reward = {self.best_avg_reward:.2f}\n'
            f'{"=" * 60}'
        )

        self.csv_file.close()
        os.system('killall -9 gzserver gzclient > /dev/null 2>&1')
        os.system('killall -9 filter_lidar respawner > /dev/null 2>&1')
        sys.exit(0)

    def reset_done_callback(self, future):
        """Callback fired when the robot-pose randomisation service responds."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Reset succeeded! {response.message}')
                self.stop_flag        = False
                self.step_count       = 0
                self.state            = None
                self.skip_lidar_scans = 15   # let physics and ROS buffers settle
                self.is_resetting     = False
                self.unpause_physics_client.call_async(Empty.Request())
            else:
                self.get_logger().error(f'Reset failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Cannot reset robot: {e}')

    def stop_robot(self):
        """Publish a zero-velocity command to halt the robot."""
        self.cmd_vel_publisher.publish(Twist())
        self.get_logger().info('Robot stopped')


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = Trainer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
