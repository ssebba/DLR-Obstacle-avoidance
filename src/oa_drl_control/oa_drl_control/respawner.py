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
from pathlib import Path
import csv

class Respawner(Node):
    
    def __init__(self):
        super().__init__("respawner")

        self.declare_parameter('package_name', 'oa_drl_control')
        self.declare_parameter('world_file', 'training_env.world')
        self.declare_parameter('robot_name', 'burger')
        self.declare_parameter('margin', 0.2)

        pkg_name = self.get_parameter('package_name').value
        world_file = self.get_parameter('world_file').value
        self.robot_name = self.get_parameter('robot_name').value
        self.margin = self.get_parameter('margin').value

        self.map_obstacles = self.parse_world_file(pkg_name, world_file)
        if self.map_obstacles:
            map_min_x, map_max_x = float('inf'), float('-inf')
            map_min_y, map_max_y = float('inf'), float('-inf')
            for obs in self.map_obstacles:
                if obs[0] == 'rect':
                    _, wx, wy, sx, sy = obs
                    map_min_x = min(map_min_x, wx - sx/2.0)
                    map_max_x = max(map_max_x, wx + sx/2.0)
                    map_min_y = min(map_min_y, wy - sy/2.0)
                    map_max_y = max(map_max_y, wy + sy/2.0)
                elif obs[0] == 'tri':
                    _, p1, p2, p3 = obs
                    map_min_x = min(map_min_x, p1[0], p2[0], p3[0])
                    map_max_x = max(map_max_x, p1[0], p2[0], p3[0])
                    map_min_y = min(map_min_y, p1[1], p2[1], p3[1])
                    map_max_y = max(map_max_y, p1[1], p2[1], p3[1])
            self.map_min_x, self.map_max_x = map_min_x, map_max_x
            self.map_min_y, self.map_max_y = map_min_y, map_max_y
            
            self.get_logger().info(f'Computed map limits: X[{self.map_min_x:.2f}, {self.map_max_x:.2f}], Y[{self.map_min_y:.2f}, {self.map_max_y:.2f}]')
        else:
            self.map_min_x, self.map_max_x = -5.0, 5.0
            self.map_min_y, self.map_max_y = -5.0, 5.0
            self.get_logger().info(f'Failed to compute limits, used the defaults: X[{self.map_min_x:.2f}, {self.map_max_x:.2f}], Y[{self.map_min_y:.2f}, {self.map_max_y:.2f}]')

        self.cb_group = ReentrantCallbackGroup()
        self.set_state_client = self.create_client(SetEntityState, '/set_entity_state', callback_group=self.cb_group)
        self.srv = self.create_service(Trigger, '/randomize_robot_pose', self.handle_randomize_pose, callback_group=self.cb_group)

        self.csv_filepath = Path.home() / "ros_ws" / "models" / "valid_poses.csv"
        
        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x', 'y', 'yaw'])
            self.get_logger().info(f'Created new CSV for the poses: {self.csv_filepath}')

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

            # Check if there is an OBJ file referenced in the world
            obj_mesh_path = None
            for uri in root.findall('.//mesh/uri'):
                if uri.text and uri.text.endswith('.obj'):
                    obj_mesh_path = uri.text
                    break

            if obj_mesh_path:
                if obj_mesh_path.startswith('model://'):
                    parts = obj_mesh_path[len('model://'):].split('/')
                    pkg = parts[0]
                    rel_path = '/'.join(parts[1:])
                    obj_path = os.path.join(get_package_share_directory(pkg), rel_path)
                elif obj_mesh_path.startswith('file://'):
                    obj_path = obj_mesh_path[len('file://'):]
                else:
                    obj_path = os.path.join(pkg_share_dir, 'worlds', os.path.basename(obj_mesh_path))
                
                self.get_logger().info(f'Parsing OBJ file for obstacles: {obj_path}')
                
                vertices = []
                triangles = []
                with open(obj_path, 'r') as f_obj:
                    for line in f_obj:
                        if line.startswith('v '):
                            parts = line.split()
                            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                        elif line.startswith('f '):
                            parts = line.split()
                            v1 = int(parts[1].split('/')[0]) - 1
                            v2 = int(parts[2].split('/')[0]) - 1
                            v3 = int(parts[3].split('/')[0]) - 1
                            triangles.append((vertices[v1], vertices[v2], vertices[v3]))
                
                # We assume Gazebo uses Z-up, while Fusion OBJ uses Y-up (hence the 90-degree world rotation).
                for t in triangles:
                    # Filter for top of the walls (vertices above 1.0 on Y axis)
                    if all(v[1] > 1.0 for v in t):
                        # Gazebo X = OBJ X, Gazebo Y = -OBJ Z
                        gx = [v[0] for v in t]
                        gy = [-v[2] for v in t]
                        
                        walls.append(('tri', (gx[0], gy[0]), (gx[1], gy[1]), (gx[2], gy[2])))
            else:
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
                                
                            walls.append(('rect', world_x, world_y, sx, sy))
                            
            self.get_logger().info(f'Caricati {len(walls)} ostacoli dal file .world')
        except Exception as e:
            self.get_logger().error(f'Errore nel parsing del file .world: {e}')
        return walls

    def yaw_to_quaternion(self, yaw):
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def pt_seg_dist(self, px, py, x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        c1 = wx * vx + wy * vy
        if c1 <= 0: return math.hypot(px - x1, py - y1)
        c2 = vx * vx + vy * vy
        if c2 == 0: return math.hypot(px - x1, py - y1)
        if c2 <= c1: return math.hypot(px - x2, py - y2)
        b = c1 / c2
        return math.hypot(px - (x1 + b * vx), py - (y1 + b * vy))

    def is_point_in_triangle(self, px, py, p1, p2, p3):
        def sign(x1, y1, x2, y2, x3, y3):
            return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
        d1 = sign(px, py, p1[0], p1[1], p2[0], p2[1])
        d2 = sign(px, py, p2[0], p2[1], p3[0], p3[1])
        d3 = sign(px, py, p3[0], p3[1], p1[0], p1[1])
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def get_random_safe_pose(self, margin=0.6, forward_clearance=0.6):
        min_x = self.map_min_x + margin
        max_x = self.map_max_x - margin
        min_y = self.map_min_y + margin
        max_y = self.map_max_y - margin

        steps = 5
        check_distances = [forward_clearance * (i / steps) for i in range(steps + 1)]

        # Consideriamo un margine ridotto per i punti "proiettati" in avanti.
        # Questo rappresenta circa l'ingombro del robot stesso (es. raggio di 15-20cm),
        # altrimenti sommeresti 60cm al punto che è già 55cm in avanti!
        point_margin = 0.20 

        while True:
            px = random.uniform(min_x, max_x)
            py = random.uniform(min_y, max_y)
            yaw = random.uniform(-math.pi, math.pi)

            is_safe = True
            
            for d in check_distances:
                check_x = px + d * math.cos(yaw)
                check_y = py + d * math.sin(yaw)

                # 1. Verifica limiti mappa
                if not (self.map_min_x < check_x < self.map_max_x and 
                        self.map_min_y < check_y < self.map_max_y):
                    is_safe = False
                    break

                # Scegliamo quale margine usare: 
                # Se d == 0 (centro del robot), usiamo il margin grande (0.6)
                # Se d > 0 (punto di proiezione frontale), usiamo il margin piccolo per l'ingombro
                current_margin = margin if d == 0.0 else point_margin

                # 2. Verifica collisione con ostacoli
                for obs in self.map_obstacles:
                    if obs[0] == 'rect':
                        _, wx, wy, sx, sy = obs
                        w_min_x = wx - (sx / 2.0) - current_margin
                        w_max_x = wx + (sx / 2.0) + current_margin
                        w_min_y = wy - (sy / 2.0) - current_margin
                        w_max_y = wy + (sy / 2.0) + current_margin

                        if (w_min_x < check_x < w_max_x) and (w_min_y < check_y < w_max_y):
                            is_safe = False
                            break
                    elif obs[0] == 'tri':
                        _, p1, p2, p3 = obs
                        if self.is_point_in_triangle(check_x, check_y, p1, p2, p3):
                            is_safe = False
                            break
                        if self.pt_seg_dist(check_x, check_y, p1[0], p1[1], p2[0], p2[1]) < current_margin or \
                           self.pt_seg_dist(check_x, check_y, p2[0], p2[1], p3[0], p3[1]) < current_margin or \
                           self.pt_seg_dist(check_x, check_y, p3[0], p3[1], p1[0], p1[1]) < current_margin:
                            is_safe = False
                            break
                
                if not is_safe:
                    break 

            if is_safe:
                return px, py, yaw

    def handle_randomize_pose(self, request, response):
        px, py, yaw = self.get_random_safe_pose(margin=self.margin)

        try:
            with open(self.csv_filepath, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Arrotondiamo a 3 cifre decimali per mantenere il CSV pulito
                writer.writerow([f"{px:.3f}", f"{py:.3f}", f"{yaw:.3f}"])
        except Exception as e:
            self.get_logger().error(f"Errore durante il salvataggio nel CSV: {e}")

        # Invia la richiesta a Gazebo tramite comando da terminale,
        # aggirando il bug del servizio ROS 2 /set_entity_state in Humble
        cmd = f"gz model -m {self.robot_name} -x {px:.3f} -y {py:.3f} -z 0.01 -R 0.0 -P 0.0 -Y {yaw:.3f}"
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