import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CircleMover(Node):
    def __init__(self):
        super().__init__('circle_mover')
        # Ci colleghiamo al nuovo canale creato dal cilindro
        self.publisher_ = self.create_publisher(Twist, '/obstacle_circ/cmd_vel', 10)
        
        # Invia il comando 10 volte al secondo
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        
       
        # Velocità lineare 
        msg.linear.x = 0.5  
        
        # Velocità angolare 
        msg.angular.z = 0.25 
        
        # Inviando questi due valori in modo fisso e continuo, 
        # il modello farà un cerchio perfetto.
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CircleMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()