import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class PatrolWalker(Node):
    def __init__(self):
        super().__init__('random_walker') 
        
        self.publisher_ = self.create_publisher(Twist, '/human/cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, '/human/odom', self.odom_callback, 10)
        
        # ==========================================================
        # ROTTA DI PATTUGLIAMENTO ASSOLUTA (Centro esatto dei muri)
        # ==========================================================
        self.waypoints = [
            # --- ANDATA (Esplorazione) ---
            (1.5, 0.0),    # 0: Braccio sinistro (Partenza)
            (3.7, 0.0),    # 1: Incrocio principale sinistro
            (3.7, 4.9),    # 2: Salita fino al corridoio superiore
            (9.7, 4.9),    # 3: Tutta a destra nel corridoio superiore
            (9.7, 2.0),    # 4: Giù nella fessura tra blocco blu e muro alto
            (11.9, 2.0),   # 5: Destra, uscendo dalla fessura
            (11.9, -2.9),  # 6: Giù fino in fondo al corridoio di destra
            (6.5, -2.9),   # 7: Sinistra, passando SOTTO il blocco blu
            (6.5, 1.5),    # 8: Su, dentro la stanza del Goal (Vicolo cieco)
            
            # --- RITORNO (Backtracking per non incastrarsi) ---
            (6.5, -2.9),   # 9: Esce dalla stanza del Goal tornando giù
            (11.9, -2.9),  # 10: Ritorna all'angolo in basso a destra
            (11.9, 2.0),   # 11: Risale il corridoio di destra
            (9.7, 2.0),    # 12: Sinistra per infilarsi nella fessura
            (9.7, 4.9),    # 13: Su per tornare nel corridoio superiore
            (3.7, 4.9),    # 14: Tutta a sinistra fino all'angolo
            (3.7, 0.0),    # 15: Giù fino all'incrocio principale
            # (Da qui ricomincia dal Waypoint 0 chiudendo il giro infinito)
        ]
        self.current_wp_index = 0
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        self.current_twist = Twist()
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info("Pedone Pattugliatore (Andata e Ritorno Mappato) avviato!")

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.navigate_to_waypoint()

    def navigate_to_waypoint(self):
        target_x, target_y = self.waypoints[self.current_wp_index]
        
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Tolleranza precisa: appena è a 40cm dal centro del punto, vira al prossimo.
        # Questo garantisce curve fluide senza "spigolare" troppo.
        if distance < 0.4:
            self.get_logger().info(f"Waypoint {self.current_wp_index} raggiunto! Punto al { (self.current_wp_index + 1) % len(self.waypoints) }")
            self.current_wp_index = (self.current_wp_index + 1) % len(self.waypoints)
            return

        target_yaw = math.atan2(dy, dx)
        error_yaw = target_yaw - self.current_yaw
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))
        
        # Sterzata differenziale: se l'errore è ampio frena e si gira, altrimenti accelera
        if abs(error_yaw) > 0.4:
            self.current_twist.linear.x = 0.2
            self.current_twist.angular.z = 1.5 * (error_yaw / abs(error_yaw)) 
        else:
            self.current_twist.linear.x = 0.6
            self.current_twist.angular.z = 2.0 * error_yaw 

    def timer_callback(self):
        self.publisher_.publish(self.current_twist)

def main(args=None):
    rclpy.init(args=args)
    node = PatrolWalker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()