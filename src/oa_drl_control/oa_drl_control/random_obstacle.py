#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class CyclicObstacle(Node):
    def __init__(self):
        super().__init__('cyclic_obstacle_node')

        # --- CONFIGURAZIONE PERCORSO CICLICO ---
        # Definiamo 4 angoli basati sui limiti del tuo .obj (-6.5 a 8.5 e -7.25 a 8.75)
        # Usiamo margini interni per non strisciare sui muri
        # Coordinate calcolate visivamente dalla griglia per evitare i muri.
        # Traccia un percorso ad anello nelle zone più larghe della mappa.
        self.waypoints = [
            {'x': 4.826, 'y':  -2},  # Punto 1
            {'x': 4.826, 'y': 0.296},  # Punto 2
            {'x': 7.612, 'y': 0.296},  # Punto 3
            {'x': 7.612, 'y': 4.134},  # Punto 4
            {'x': 3.22, 'y': 4.117},  # Punto 5
            {'x': 0.771, 'y':  2.247},  # Punto 6
            {'x': -1.757, 'y':  3.516},   # Punto 7
            {'x': -5.422, 'y':  3.516},  # Punto 8
            {'x': -5.422, 'y':  0.710},   # Punto 9
            {'x': -2.307, 'y':  -0.355},   # Punto 10
            {'x': -2.307, 'y':  -4.590},   # Punto 11
            {'x': -0.571, 'y':  -5.834},   # Punto 12
            {'x': 1.354, 'y':  -5.847},   # Punto 13
            {'x': 1.546, 'y':  -1.612}   # Punto 14
            #{'x': 4.826, 'y':  -2}   # Punto 15
        ]
        self.current_wp_idx = 0
        self.dist_threshold = 0.3  # Tolleranza abbassata per fargli seguire i punti in modo più rigoroso

        # Stato attuale
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/ostacolo/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/ostacolo/odom', self.odom_callback, 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Nodo Ostacolo Ciclico Avviato.')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Estrazione dello Yaw (rotazione Z) dai quaternioni
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        target = self.waypoints[self.current_wp_idx]
        
        # Calcolo distanza dal target
        dist = math.sqrt((target['x'] - self.current_x)**2 + (target['y'] - self.current_y)**2)
        
        # Se siamo vicini al punto, passiamo al prossimo
        if dist < self.dist_threshold:
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            self.get_logger().info(f'Punto raggiunto! Prossimo target: {self.current_wp_idx}')
            return

        # Calcolo angolo verso il target
        angle_to_target = math.atan2(target['y'] - self.current_y, target['x'] - self.current_x)
        angle_diff = angle_to_target - self.current_yaw
        
        # Normalizzazione angolo
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi

        msg = Twist()
        
        # Logica di movimento semplice:
        # Se l'angolo è molto sbagliato, gira sul posto
        if abs(angle_diff) > 0.5:
            msg.linear.x = 0.0
            msg.angular.z = 0.6 if angle_diff > 0 else -0.6
        else:
            # Se siamo puntati bene, vai avanti e correggi leggermente
            msg.linear.x = 0.3 # velocità diminuita 0.4 TROPPO ALTA, 0.15 TROPPO BASSA
            msg.angular.z = 0.5 * angle_diff

        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CyclicObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()