import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Trigger
from gazebo_msgs.srv import SetEntityState
import xml.etree.ElementTree as ET
import os
from ament_index_python.packages import get_package_share_directory
import random
import math

class Respawner(Node):
    
    def __init__(self):
        super().__init__("respawner")

        self.declare_parameter('package_name', 'oa_drl_control')
        self.declare_parameter('world_file', 'world1.world')
        self.declare_parameter('robot_name', 'burger')

        pkg_name = self.get_parameter('package_name').value
        world_file = self.get_parameter('world_file').value
        self.robot_name = self.get_parameter('robot_name').value

        self.map_obstacles = self.parse_world_file(pkg_name, world_file)
        if self.map_obstacles:
            self.map_min_x = min([wx - (sx / 2.0) for wx, wy, sx, sy in self.map_obstacles])
            self.map_max_x = max([wx + (sx / 2.0) for wx, wy, sx, sy in self.map_obstacles])
            self.map_min_y = min([wy - (sy / 2.0) for wx, wy, sx, sy in self.map_obstacles])
            self.map_max_y = max([wy + (sy / 2.0) for wx, wy, sx, sy in self.map_obstacles])
            
            self.get_logger().info(f'Computed map limits: X[{self.map_min_x:.2f}, {self.map_max_x:.2f}], Y[{self.map_min_y:.2f}, {self.map_max_y:.2f}]')
        else:
            self.map_min_x, self.map_max_x = -5.0, 5.0
            self.map_min_y, self.map_max_y = -5.0, 5.0
            self.get_logger().info(f'Failed to compute limits, used the defaults: X[{self.map_min_x:.2f}, {self.map_max_x:.2f}], Y[{self.map_min_y:.2f}, {self.map_max_y:.2f}]')

        self.cb_group = ReentrantCallbackGroup()
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state', callback_group=self.cb_group)
        self.srv = self.create_service(Trigger, '/randomize_robot_pose', self.handle_randomize_pose, callback_group=self.cb_group)

        self.get_logger().info('Respawner initialized.')
    

    def parse_world_file(self, package_name, world_file_name):
        walls = []
        try:
            pkg_share_dir = get_package_share_directory(package_name)
            world_path = os.path.join(pkg_share_dir, 'worlds', world_file_name) 

            with open(world_path, 'r') as f:
                world_content = f.read()

            clean_content = world_content.replace('ignition::', 'ignition_')
            root = ET.fromstring(clean_content)

            # First extract global offsets from the <state> section
            model_offsets = {}
            for state_model in root.findall('.//state/model'):
                name = state_model.get('name')
                pose_tag = state_model.find('pose')
                if pose_tag is not None:
                    mp_vals = [float(v) for v in pose_tag.text.split()]
                    model_offsets[name] = (mp_vals[0], mp_vals[1], mp_vals[5])

            for model in root.findall('.//model'):
                model_name = model.get('name')
                if model_name in ['ground_plane', 'turtlebot3_burger']:
                    continue

                # Read model global offset: prefer state offset, fallback to model pose
                mx, my, myaw = 0.0, 0.0, 0.0
                if model_name in model_offsets:
                    mx, my, myaw = model_offsets[model_name]
                else:
                    model_pose_tag = model.find('pose')
                    if model_pose_tag is not None:
                        mp_vals = [float(v) for v in model_pose_tag.text.split()]
                        mx, my, myaw = mp_vals[0], mp_vals[1], mp_vals[5]

                for link in model.findall('.//link'):
                    pose_tag = link.find('pose')
                    size_tag = link.find('.//collision/geometry/box/size')

                    if pose_tag is not None and size_tag is not None:
                        pose_vals = [float(v) for v in pose_tag.text.split()]
                        size_vals = [float(v) for v in size_tag.text.split()]
                        
                        lx, ly, lyaw = pose_vals[0], pose_vals[1], pose_vals[5]
                        
                        # Apply model offset and rotation
                        world_x = mx + lx * math.cos(myaw) - ly * math.sin(myaw)
                        world_y = my + lx * math.sin(myaw) + ly * math.cos(myaw)
                        global_yaw = myaw + lyaw
                        
                        sx, sy = size_vals[0], size_vals[1]
                        
                        # If the wall is rotated ~90 or ~270 degrees, swap sx and sy for the bounding box
                        if abs(math.cos(global_yaw)) < 0.5:
                            sx, sy = sy, sx
                            
                        walls.append((world_x, world_y, sx, sy))
                        
            self.get_logger().info(f'Caricati {len(walls)} ostacoli dal file .world')
        except Exception as e:
            self.get_logger().error(f'Errore nel parsing del file .world: {e}')
        return walls

    def yaw_to_quaternion(self, yaw):
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def get_random_safe_pose(self, margin=0.6, forward_clearance=0.55):
        min_x = self.map_min_x + margin
        max_x = self.map_max_x - margin
        min_y = self.map_min_y + margin
        max_y = self.map_max_y - margin

        # Creiamo degli step per controllare lo spazio davanti al robot.
        # Controllare solo la fine del segmento (0.55) potrebbe farci saltare
        # muri molto sottili che si trovano in mezzo.
        steps = 5
        check_distances = [forward_clearance * (i / steps) for i in range(steps + 1)]
        # Questo genererà distanze: [0.0, 0.11, 0.22, 0.33, 0.44, 0.55]

        while True:
            px = random.uniform(min_x, max_x)
            py = random.uniform(min_y, max_y)
            yaw = random.uniform(-math.pi, math.pi)

            is_safe = True
            
            # Controlliamo il centro del robot e i punti di fronte a lui
            for d in check_distances:
                # Calcoliamo il punto "d" metri davanti al robot
                check_x = px + d * math.cos(yaw)
                check_y = py + d * math.sin(yaw)

                # 1. Verifica che lo spazio davanti non esca dai limiti della mappa
                if not (self.map_min_x < check_x < self.map_max_x and 
                        self.map_min_y < check_y < self.map_max_y):
                    is_safe = False
                    break

                # 2. Verifica che il punto non sia dentro nessun ostacolo
                for (wx, wy, sx, sy) in self.map_obstacles:
                    # Manteniamo il margine per assicurarci che ci sia spazio non solo 
                    # per un singolo punto, ma per l'ingombro del robot
                    w_min_x = wx - (sx / 2.0) - margin
                    w_max_x = wx + (sx / 2.0) + margin
                    w_min_y = wy - (sy / 2.0) - margin
                    w_max_y = wy + (sy / 2.0) + margin

                    if (w_min_x < check_x < w_max_x) and (w_min_y < check_y < w_max_y):
                        is_safe = False
                        break # Esce dal ciclo degli ostacoli
                
                if not is_safe:
                    break # Esce dal ciclo delle distanze (riparte il While True)

            if is_safe:
                return px, py, yaw

    def handle_randomize_pose(self, request, response):
        px, py, yaw = self.get_random_safe_pose(margin=0.6)

        # Invia la richiesta a Gazebo tramite comando da terminale,
        # aggirando il bug del servizio ROS 2 /set_entity_state in Humble
        cmd = f"gz model -m {self.robot_name} -x {px:.3f} -y {py:.3f} -z 0.05 -Y {yaw:.3f}"
        os.system(cmd)

        response.success = True
        response.message = f"Riposizionato in x:{px:.2f}, y:{py:.2f}"
        return response

def main(args=None):
    rclpy.init(args=args)

    contr = Respawner()
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(contr, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        contr.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()