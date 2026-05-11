import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class ObstacleMover(Node):
    def __init__(self):
        super().__init__('obstacle_mover')
        # Pubblica la velocità sul topic specifico dell'ostacolo
        self.publisher_ = self.create_publisher(Twist, '/obstacle1/cmd_vel', 10)
        
        # Un timer che gira ogni 0.1 secondi
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.time_elapsed = 0.0
        self.direction = 1.0

    def timer_callback(self):
        msg = Twist()
        
        # Inverti la direzione ogni 4 secondi
        if self.time_elapsed >= 4.0:
            self.direction *= -1.0
            self.time_elapsed = 0.0

        # Si muove a 1 metro al secondo sull'asse Y
        msg.linear.y = 1.0 * self.direction
        
        self.publisher_.publish(msg)
        self.time_elapsed += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()