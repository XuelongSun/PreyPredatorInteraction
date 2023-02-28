from itertools import combinations
from collections import deque

# import os
# if 'WAYLAND_DISPLAY' in os.environ and 'PYOPENGL_PLATFORM' not in os.environ:
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# if 'WAYLAND_DISPLAY' in os.environ:
#     del os.environ['WAYLAND_DISPLAY']
    
import numpy as np
import trimesh as tr
import pyrender as pr
import cv2

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
# from simulator import Simulator

def world2image_coordinates_transfer(p, boundary):
    c = int(np.clip(p[0], boundary[0], boundary[1]-1) - boundary[0])
    r = int(boundary[3] - np.clip(p[1], boundary[2]+1, boundary[3]))
    return [r, c]

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def calculate_pose_matrix(rotation, translation):
    '''
    create camera pose matrix in homogeneous format.add()
    
    Parameters
    ---
    rotation - dict: {'x':r, 'y': phi, 'z': theta}
    translation - array: [x, y, z]
    '''
    M = np.identity(4)
    
    # rotation
    rotate_matrix = np.identity(3)
    if 'x' in rotation.keys():
        angle = rotation['x']
        if not angle == 0:
            Rx = np.zeros(shape=(3, 3))
            Rx[0, 0] = 1
            Rx[1, 1] = np.cos(angle)
            Rx[1, 2] = -np.sin(angle)
            Rx[2, 1] = np.sin(angle)
            Rx[2, 2] = np.cos(angle)
            rotate_matrix = np.matmul(rotate_matrix, Rx)
    if 'y' in rotation.keys():
        angle = rotation['y']
        if not angle == 0:
            Ry = np.zeros(shape=(3, 3))
            Ry[0, 0] = np.cos(angle)
            Ry[0, 2] = -np.sin(angle)
            Ry[2, 0] = np.sin(angle)
            Ry[2, 2] = np.cos(angle)
            Ry[1, 1] = 1
            rotate_matrix = np.matmul(rotate_matrix, Ry)
    if 'z' in rotation.keys():
        angle = rotation['z']
        if not angle == 0:
            Rz = np.zeros(shape=(3, 3))
            Rz[0, 0] = np.cos(angle)
            Rz[0, 1] = -np.sin(angle)
            Rz[1, 0] = np.sin(angle)
            Rz[1, 1] = np.cos(angle)
            Rz[2, 2] = 1
            rotate_matrix = np.matmul(rotate_matrix, Rz)
    M[:3, :3] = rotate_matrix
    # translation
    M[:3, 3] = translation
    
    return M

def create_cylinder_robot_node(radius, height, color=(135, 206, 235)):
    bar = tr.creation.box(extents=(0.5, radius*0.9, 0.1))
    bar.visual.face_colors = (229, 240, 16)
    pose = calculate_pose_matrix({},[0.0, radius/2, height/2])
    bar_node = pr.Node(mesh=pr.Mesh.from_trimesh(bar, smooth=False),
                       matrix=pose)
    camera = pr.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.2)
    pose = calculate_pose_matrix({'x':np.pi/2,'y':0},[0,radius,0])
    camera_node = pr.Node(camera=camera, matrix=pose)
    
    c = tr.creation.cylinder(radius=radius, height=height)
    c.visual.face_colors = color
    c_node = pr.Node(mesh=pr.Mesh.from_trimesh(c, smooth=False),
                     children=[bar_node, camera_node])
    
    return c_node

def color_rgb2hsv(color):
    '''
    convert RGB color to HSV
    
    HSV is in the range as open-cv: i.e.:
    H : 0 ~ 255
    S : 0 ~ 255
    V : 0 ~ 255
    color: rgb(0-255, 0-255, 0-255)
    '''
    c_max = max(color)
    c_min = min(color)
    
    if c_max == c_min:
        h = 0
    elif color.index(c_max) == 0 and color[1] >= color[2]:
        h = 60 * (color[1] - color[2])/(c_max - c_min)
    elif color.index(c_max) == 0 and color[1] < color[2]:
        h = 60 * (color[1] - color[2])/(c_max - c_min) + 360
    elif color.index(c_max) == 1:
        h = 60 * (color[2] - color[0])/(c_max - c_min) + 120
    elif color.index(c_max) == 2:
        h = 60 * (color[0] - color[1])/(c_max - c_min) + 240
    
    s = 0 if c_max == 0 else (c_max - c_min)/c_max
    
    v = c_max
    
    return (int(h/2), int(s*255), int(v))

# 3d view
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 150

class OdorSensor:
    def __init__(self) -> None:
        self.radius = 1
        self.value = 0
    
    def get_value(self, p, phi_field, boundary):
        p_ = world2image_coordinates_transfer(p, boundary)
        self.value = phi_field[p_[0], p_[1]]
        return self.value

class Agent:
    def __init__(self, scene3d, render3d, uuid, init_heading, init_position,
                 raduis=10, height=5, color=(135, 206, 235)):
        self.id = uuid
        self.cluster_id = None

        self.scene3d = scene3d
        self.render3d = render3d
        
        # motion
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.position = np.array(init_position, np.float32)
        self.heading = init_heading
        
        self.collision = False
        self.collision_theta = 0
        self.collision_counter = 0
        
        # scale
        self.size = raduis
        self.collision_distance = self.size
        # visual 3D object
        self.node3d = create_cylinder_robot_node(raduis, height, color)
        self.scene3d.add_node(self.node3d)
        pose = calculate_pose_matrix({'z':self.heading-np.pi/2},
                                     [self.position[0],
                                      self.position[1], 0])
        self.scene3d.set_pose(self.node3d, pose=pose)
        self.body_color_rgb = color
        self.body_color_hsv = color_rgb2hsv(color)
        
        self.time = 0
        
        self.last_cluster_prey_num = 0

    def update_motion(self, boundary, obstacles):
        # if still in collision
        if self.collision_counter > 0:
            self.linear_velocity = 2
            self.angular_velocity = 0
            self.collision_counter -= 1
        else:
            # if at  the arena boundary
            if (self.position[0] < boundary[0] + self.size) or \
                (self.position[0] > boundary[1] - self.size) or \
                    (self.position[1] < boundary[2] + self.size) or \
                        (self.position[1] > boundary[3] - self.size):
                            # sharp turn
                            self.collision_counter = 20
                            self.linear_velocity = 1.2
                            self.angular_velocity = np.random.uniform(np.pi/2, np.pi)
            elif self.collision_obstacles(obstacles):
                self.angular_velocity = np.random.vonmises(3*np.pi/4, 100.0, 1)[0]
            
            if self.collision:
                # completely inelastic collision
                self.linear_velocity *= np.sin(self.heading - self.collision_theta)
                self.angular_velocity -= self.collision_theta

        self.heading += self.angular_velocity
        # scale to -pi~pi
        self.heading = (self.heading + np.pi) % (np.pi*2) - np.pi
        self.position += self.linear_velocity * np.array([np.cos(self.heading),
                                                        np.sin(self.heading)])
        # update visual object in 3d scene
        pose = calculate_pose_matrix({'z':self.heading-np.pi/2},
                                    [self.position[0], self.position[1], 0])
        self.scene3d.set_pose(self.node3d, pose=pose)
        
    
    def get_camera_data(self):
        self.scene3d.main_camera_node = self.node3d.children[1]
        c, d = self.render3d.render(self.scene3d)
        return c

    def collision_obstacles(self, obstacles):
        if obstacles is not None:
            for obstacle in obstacles:
                if obstacle['shape'] == 'circle':
                    if distance(self.position, obstacle['position']) <= \
                        obstacle['radius'] + self.size + self.collision_distance:
                            return True
                elif obstacle['shape'] == 'rectangle':
                    margin_x = self.size + obstacle['width']/2 + self.collision_distance
                    margin_y = self.size + obstacle['height']/2 + self.collision_distance
                    if abs(self.position[0] - obstacle['center'][0]) <= margin_x and \
                        abs(self.position[1] - obstacle['center'][1]) <= margin_y:
                            return True
            return False
        else:
            return False

class Prey(Agent):
    def __init__(self, scene3d, render3d, uuid, init_heading, init_position):
        super().__init__(scene3d, render3d, uuid, init_heading, init_position)
        
        self.odor_sensors = [OdorSensor(), OdorSensor()]
        self.energy = 2
        self.state = "wandering"
        
        self.f_gather = np.random.uniform(0.02, 0.32)
        self.f_avoid = np.random.uniform(0.01,10.01)
        
        self.odor_sensors_positions = np.array([[0, self.size/2], [0, -self.size/2]])
        self.view = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], np.uint8)
        self.view_processed = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 3], np.uint8)
        
        # visual based control
        self.hsv_l = (self.body_color_hsv[0] - 5, 50, 50)
        self.hsv_h = (self.body_color_hsv[0] + 5, 255, 255)
        self.thr_start_pixel_num = 0.05
        self.thr_stop_pixel_num = 0.4
        self.max_pixel_num = 0.7
        self.f_size = 0
        
        # olfactory based control
        self.odor_sensor_l_values = deque([0,0,0,0],maxlen=4)
        self.odor_sensor_r_values = deque([0,0,0,0],maxlen=4)
        
    def get_odor_sensor_position(self):
        positions = []
        r_matrix = np.array([[np.cos(self.heading), np.sin(self.heading)],
                             [-np.sin(self.heading), np.cos(self.heading)]])
        for p in self.odor_sensors_positions:
            rotated = np.matmul(p, r_matrix)
            positions.append(self.position + rotated)
        return positions
    
    def update(self, dt, phero_f, boundary):
        if (self.energy <= 0) and (self.state != 'death'):
            self.state = "death"
            # remove from the 3d scene
            self.scene3d.remove_node(self.node3d)
            return
        elif(self.energy>=30):
            self.energy=30
        
        if self.state == 'death':
            return
        
        # get predator's alarm pheromone
        s_p = self.get_odor_sensor_position()
        o_value_l = self.odor_sensors[0].get_value(
            s_p[0], phero_f, boundary
        )[0]
        self.odor_sensor_l_values.append(o_value_l)
        o_value_r = self.odor_sensors[1].get_value(
            s_p[1], phero_f, boundary
        )[0]
        self.odor_sensor_r_values.append(o_value_r)
        # get camera data (image view)
        if self.time % 100 < dt:
            self.view = self.get_camera_data()
        
        # state update depending on sensory inputs
        if (o_value_l > self.f_avoid) or (o_value_r > self.f_avoid):
            self.state = 'avoiding'
        else:
            # find if there is interesting color in the view
            hsv = cv2.cvtColor(self.view, cv2.COLOR_RGB2HSV)
            img_f = cv2.inRange(hsv, self.hsv_l, self.hsv_h)
            size = img_f.sum()/7650000
            if size >= self.max_pixel_num:
                self.state = 'stop'
            elif size >= self.thr_stop_pixel_num:
                self.state = 'stop'
            elif size >= self.f_gather:
                self.state = 'gathering'
            else:
                self.state = 'wandering'

            self.view_processed = img_f.copy()
            self.f_size = size

        # motion control depend on state
        if self.state == 'avoiding':
            f = np.max([0.2, np.min([self.energy * 0.2, 4])])
            self.energy -= f * dt / 1000
            self.linear_velocity = f
            
            odor_old = self.odor_sensor_l_values[-2] + self.odor_sensor_r_values[-2]
            odor_now = self.odor_sensor_l_values[-1] + self.odor_sensor_r_values[-1]
            if odor_now > odor_old:
                self.angular_velocity = np.pi/4*3
            else:
                self.angular_velocity = 0
                
        elif self.state == 'gathering':
            self.energy -= 0.2 * dt / 1000
            self.linear_velocity = 1.2
            # depend on the visual input
            x = np.mean(np.where(img_f > 1)[1])
            self.angular_velocity = (IMAGE_WIDTH/2 - x) / IMAGE_WIDTH * (np.pi/4)
        elif self.state == 'stop':
            self.energy += 2 * dt / 1000
            self.linear_velocity = 0 
            self.angular_velocity = 0
        elif self.state == 'wandering':
            self.energy -= 0.2 * dt / 1000
            self.linear_velocity = 1
            self.angular_velocity = np.random.vonmises(0.0, 100.0, 1)[0]

        self.update_motion(boundary, None)
        
        self.time += dt

class Predator(Agent):
    def __init__(self, scene3d, render3d, uuid, init_heading, init_position, speed=4):
        super().__init__(scene3d, render3d, uuid, init_heading, init_position, color=(200, 0, 20))
        self.phero_radius = 10
        self.linear_velocity = speed
        self.goal_defined = False
        self.energy = 16
        self.state = 'hunting'
        
    def update(self, boundary, cluster, num_prey):
        self.energy -= self.energy*0.0001
        if 18 <= self.energy:
            self.state = 'stopping'
            return
        else:
            self.state = 'hunting'
            # found cluster with highest density
            sorted_c = sorted(cluster.items(), key=lambda v:len(v[1]))
            _, v1 = sorted_c[-1]
            v2 = [] if len(sorted_c) <=1 else sorted_c[-2][1]
            if len(v1) >= 5:
                if len(v1) >= num_prey - 2:
                    i = np.random.randint(0, len(v1))
                    px = v1[i].position[0]
                    py = v1[i].position[1]
                    goal_dir = np.arctan2(self.position[1]-py, self.position[0]-px)
                    self.angular_velocity = goal_dir - self.heading + np.pi
                else:
                    if len(v1) >= self.last_cluster_prey_num+2:
                        px = np.array([a.position[0] for a in v1]).mean()
                        py = np.array([a.position[1] for a in v1]).mean()
                        goal_dir = np.arctan2(self.position[1]-py, self.position[0]-px)
                        self.angular_velocity = goal_dir - self.heading + np.pi
                    else:
                        self.angular_velocity = 0
            self.last_cluster_prey_num = len(v1)
            self.update_motion(boundary, None)


class Pheromone:
    def __init__(self, width, height, dt):
        self.dt = dt
        self.width = width
        self.height = height
        self.field = np.zeros([height, width, 3])
        # pheromone parameters
        self.evaporation = 1e3
        self.diffusion = 0.99
        self.diffusion_kernel = np.array([[(1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8],
                                          [(1 - self.diffusion) / 8,
                                           self.diffusion - 1,
                                           (1 - self.diffusion) / 8],
                                          [(1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8,
                                           (1 - self.diffusion) / 8]])
        
    def update(self, inject_pos, inject_size, inject_k):
        for i in range(2):
            # injection
            for pos, k in zip(inject_pos, inject_k):
                
                start_x = pos[0]-inject_size
                start_y = pos[1]-inject_size
                end_x = pos[0]+inject_size
                end_y = pos[1]+inject_size
                
                start_x = max(min(self.width-1, start_x), 0)
                start_y = max(min(self.height-1, start_y), 0)
                end_x = max(min(self.width-1, end_x), 0)
                end_y = max(min(self.height-1, end_y), 0)
                
                self.field[start_x:end_x, start_y:end_y,i]+= k[i]
            
        # evaporation
        self.field += self.field * (-1/self.evaporation) * self.dt
        
        # diffusion
        r, g, b = cv2.split(self.field)
        r = cv2.filter2D(r, -1, self.diffusion_kernel)
        g = cv2.filter2D(g, -1, self.diffusion_kernel)
        b = cv2.filter2D(b, -1, self.diffusion_kernel)
        d = cv2.merge([r, g, b])
        self.field += d*self.dt
        
        self.field = np.clip(self.field, 0.01, 1e4)


class Environment:
    def __init__(self, width, height, boundary, dt, obstacles=None):
        self.pheromone = Pheromone(width, height, dt)
        self.pheromone_mask = np.ones(self.pheromone.field.shape)

        self.boundary = boundary
        self.food_catchment = 20
        self.food_position = [self.boundary[0]+self.food_catchment,
                              self.boundary[3]/2]
        
        self.nest_catchment = 20
        self.nest_position = [self.boundary[1]-self.nest_catchment,
                              self.boundary[3]/2]

        if obstacles is not None:
            self.obstacles = obstacles
            # pheromone mask
            for obs in self.obstacles:
                if obs['shape'] == 'rectangle':
                    p1_ = [obs['center'][0] - 0.5*obs['width'], obs['center'][1] + 0.5*obs['height']]
                    p2_ = [obs['center'][0] + 0.5*obs['width'], obs['center'][1] - 0.5*obs['height']]
                    p1 = world2image_coordinates_transfer(p1_,
                                                        self.boundary)
                    p2 = world2image_coordinates_transfer(p2_,
                                                        self.boundary)
                    self.pheromone_mask[p1[0]:p2[0], p1[1]:p2[1]] = 0
        else:
            self.obstacles = None 
            
class Simulator:
    def __init__(self, time_out, 
                 location, width, height, 
                 obstacles=None, prey_nums=30, 
                 phero_width=20, predator_speed=4,
                 ):

        self.time = 0
        self.dt = 20 # ms
        self.time_out = time_out
        boundary = [location[0], location[0] + width,
                    location[1], location[1] + height]
        self.environment = Environment(width, height,
                                       boundary, self.dt,
                                       obstacles)

        self.scene3d = pr.Scene(ambient_light=1.0, bg_color=(0,0,0))
        self.render3d = pr.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
        
        self.preys = []
        for i in range(prey_nums):
            h = np.random.vonmises(0, 100, 1)[0]
            f = False
            while not f:
                f = True
                pos = [np.random.uniform(30, width-30),
                   np.random.uniform(30, height-30)]
                for b in self.preys:
                    if distance(pos, b.position) <= b.size*2:
                        f = False
                        break
            self.preys.append(Prey(self.scene3d, self.render3d, i, h, pos))
        
        self.alive_preys = self.preys.copy()
        self.dead_preys = []
        
        self.predators = []
        # predator releas pheromones
        self.pheromone_inject_k = []
        for i in range(1):
            h = np.random.vonmises(0, 100, 1)[0]
            pos = [width/2, height/2] 
            self.pheromone_inject_k.append([100, 0].copy())
            self.predators.append(Predator(self.scene3d, self.render3d, i, h, pos, predator_speed))
        self.pheromone_inject_k = np.array(self.pheromone_inject_k, np.float32)
        self.pheromone_width = phero_width
        
        # visualization
        self.visualize_img = np.zeros((height, width, 3), np.uint8)
        self.visualize_agent_color = {'prey': (135, 206, 235),
                                      'predator':(200, 0, 20)}
        self.visualize_obstacle_color = (125,125,125)
        
        self.cluster = {}
        
        # transfer data
        self.debug_data = {}
        
        # save data
        self.data_to_save = []
        
    
    def get_collide_agents(self, agent:Prey):
        a_ = []
        for a in self.alive_preys:
            if not a.collision:
                a_.append(a)

        co_ = [] # for collision
        co_theta = []
        for a in a_:
            if a is not agent:
                if distance(a.position, agent.position) <= (a.size + agent.size)*1.1:
                    co_.append(a)
                    co_theta.append(np.arctan2((a.position[1] - a.position[1]),
                                               (a.position[0] - a.position[0])))
        return co_, co_theta
    
    def get_near_agents(self, agent:Prey):
        a_ = []
        for a in self.alive_preys:
            if a.cluster_id is None:
                a_.append(a)
                
        cl_ = [] # for cluster
        for a in a_:
            if a is not agent:
                if distance(a.position, agent.position) <= (a.size + agent.size)*2:
                    cl_.append(a)
        return cl_
    
    def arange_cluster(self):
        c_avg = {}
        for ind, ags in self.cluster.items():
            if len(ags) >=2:
                avg_x = np.array([a.position[0] for a in ags]).mean()
                avg_y = np.array([a.position[1] for a in ags]).mean()
                c_avg.update({ind:[avg_x, avg_y]})

        for c_c in combinations(c_avg.keys(), 2):
            if distance(c_avg[c_c[0]], c_avg[c_c[1]]) <= 40:
                # merge cluster
                self.cluster[c_c[0]] += self.cluster[c_c[1]]
                for m_a in self.cluster[c_c[1]]:
                    m_a.cluster_id = c_c[0]
                self.cluster.pop(c_c[1])
    
    def arange_cluster_rectangle(self):
        rect = {}
        margin = 10
        for ind, ags in self.cluster.items():
            if len(ags) >=2:
                x = np.array([a.position[0] for a in ags])
                y = np.array([a.position[1] for a in ags])
                rect.update({ind:[x.min()-margin, x.max()+margin,
                                  y.min()-margin, y.max()+margin]})
        
        for c_c in combinations(rect.keys(), 2):
            r1l = rect[c_c[1]][0]
            r1r = rect[c_c[1]][1]
            r2l = rect[c_c[0]][0]
            r2r = rect[c_c[0]][1]
            r1b = rect[c_c[1]][2]
            r1t= rect[c_c[1]][3]
            r2b = rect[c_c[0]][2]
            r2t = rect[c_c[0]][3]

            if not ((r1l > r2r) or (r1t < r2b) or (r2l > r1r) or (r2t < r1b)):
                for m_a in self.cluster[c_c[1]]:
                    m_a.cluster_id = c_c[0]
            
        self.cluster = {}
        for b in self.alive_preys:
            if b.cluster_id in self.cluster.keys():
                self.cluster[b.cluster_id].append(b)
            elif b.cluster_id is not None:
                self.cluster.update({b.cluster_id:[b]})
                    
    def clear_cluster_collision_info(self):
        self.cluster = {}
        for b in self.alive_preys:
            b.cluster_id = None
            b.collision = False
    
    def visualization2d(self, show_agent=()):
        # pheromone
        self.visualize_img = np.clip(self.environment.pheromone.field,
                                     0, 255).astype(np.uint8)
        # boundary
        self.visualize_img = cv2.rectangle(self.visualize_img, (0, 0), 
                                           (self.environment.pheromone.width-1,
                                            self.environment.pheromone.height-1),
                                           (255, 255, 255),
                                           thickness=2)
        # preys
        for a in self.preys:
            p = world2image_coordinates_transfer(a.position,
                                    self.environment.boundary)
            if a.state != 'death':
                self.visualize_img = cv2.circle(self.visualize_img, p[::-1],
                                                a.size,
                                                self.visualize_agent_color['prey'],
                                                thickness=2)
                a_end = [int(a.size*np.cos(a.heading+np.pi/2) + p[0]),
                        int(a.size*np.sin(a.heading+np.pi/2) + p[1])]
                self.visualize_img = cv2.arrowedLine(self.visualize_img,
                                                    p[::-1],
                                                    a_end[::-1],
                                                    (229, 240, 16),
                                                    thickness=2)
                # state
                if a.id in show_agent:
                    cv2.imshow('{}_view'.format(a.id), cv2.cvtColor(a.view, cv2.COLOR_RGB2BGR))
                self.visualize_img = cv2.putText(self.visualize_img, str(a.cluster_id),
                                (p[1], p[0]-a.size*2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,255))
            else:
                self.visualize_img = cv2.drawMarker(self.visualize_img, p[::-1],
                                                    (100,100,100),
                                                    markerType=cv2.MARKER_CROSS,
                                                    markerSize=a.size)
        # predators
        for a in self.predators:
            p = world2image_coordinates_transfer(a.position,
                                                 self.environment.boundary)
            self.visualize_img = cv2.circle(self.visualize_img, p[::-1],
                                            a.size,
                                            (0,0,255),
                                            thickness=2)
            a_end = [int(a.size*np.cos(a.heading+np.pi/2) + p[0]),
                     int(a.size*np.sin(a.heading+np.pi/2) + p[1])]
            self.visualize_img = cv2.arrowedLine(self.visualize_img,
                                                 p[::-1],
                                                 a_end[::-1],
                                                 (16, 240, 229),
                                                 thickness=2)
 
    def run(self, save_data=False, filename='', visualization=False):
        self.time = 0
        end_condition = True if self.time_out is None else (self.time <= self.time_out*1000)
        # alive_agents = filter(lambda p:p.state != 'death', self.preys)
        # visualization parameter
        show_agents = (0, 1)
        while end_condition:
            self.step(save_data, filename)
                    # cv2-based 2D viualization
            if visualization:
                if (self.time % 100 <= self.dt):
                    self.visualization2d(show_agent=show_agents)
                    cv2.imshow('Simulation', cv2.cvtColor(self.visualize_img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            end_condition = True if self.time_out is None else (self.time <= self.time_out*1000)
        
    def step(self, save_data=False, filename=''):
        # update prey's state
        prey_pos = []
        energy = []
        dead_prey_num = 0
        self.clear_cluster_collision_info()
        for pe in self.alive_preys:
            # cluster
            if pe.cluster_id is None:
                # belong to a new cluster
                a = self.get_near_agents(pe)
                pe.cluster_id = len(self.cluster.keys()) + 1
                self.cluster.update({pe.cluster_id:a + [pe]})
                for a_ in a:
                    a_.cluster_id = pe.cluster_id
            self.arange_cluster_rectangle()
            pe.update(self.dt, self.environment.pheromone.field, self.environment.boundary)
            p = world2image_coordinates_transfer(pe.position,
                                                 self.environment.boundary)
            prey_pos.append(p)
            # update energy
            energy.append(pe.energy)
            if pe.state == 'death':
                if self.predators[-1].state == 'hunting':
                    self.predators[-1].energy += 0.4
                self.dead_preys.append(pe)
                self.alive_preys.remove(pe)
                dead_prey_num += 1
                # spawn a new prey
                h = np.random.vonmises(0, 100, 1)[0]
                pos = [np.random.uniform(30, self.environment.boundary[1] - self.environment.boundary[0] -30),
                        np.random.uniform(30, self.environment.boundary[3] - self.environment.boundary[2] -30)]
                t_prey = Prey(self.scene3d, self.render3d, len(self.preys) + 1, h, pos)
                # 1.evolving parameters setted as the average value of the group
                # t_prey.f_gather = np.mean([a.f_gather for a in self.alive_preys])
                # t_prey.f_avoid = np.mean([a.f_avoid for a in self.alive_preys])
                if np.random.uniform(0, 1) > 0.1:
                    # 2.evolving parameters setted with possiblities (agent with max energy has highest possibility)
                    # get alive preys' energy
                    temp_a_e = [(a, a.energy) for a in self.alive_preys]
                    temp_a_e = sorted(temp_a_e, key=lambda x:x[1])
                    sum_ = sum([a_[1] for a_ in temp_a_e])
                    partial_p = [a_[1]/sum_ for a_ in temp_a_e]
                    p_ = np.random.rand(1)[0]
                    ind = np.where(partial_p < p_)[0][-1] if len(np.where(partial_p < p_)[0]) > 0 else -1
                    # print(ind, len(temp_a_e), temp_a_e)
                    t_prey.f_gather = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_gather
                    t_prey.f_avoid = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_avoid
                else:
                    # 3. no evolving, random
                    t_prey.f_gather = np.random.uniform(0.02, 0.32)
                    t_prey.f_avoid = np.random.uniform(0.01, 1.01)
                t_prey.energy = np.mean(np.array([a.energy for a in self.alive_preys]))
                # add to the lists
                self.preys.append(t_prey)
                self.alive_preys.append(t_prey)

        # render pheromone
        predator_pos = []
        for pd in self.predators:
            pd.update(self.environment.boundary,
                        self.cluster, len(self.alive_preys))
            p = world2image_coordinates_transfer(pd.position,
                                                self.environment.boundary)
            predator_pos.append(p)
        self.environment.pheromone.update(predator_pos,
                                          self.pheromone_width,
                                          self.pheromone_inject_k)

        # debug data
        self.debug_data['energy'] = energy
        self.debug_data['f_avoid'] = [a.f_avoid for a in self.alive_preys]
        self.debug_data['f_gather'] = [a.f_gather for a in self.alive_preys]
        self.debug_data['pd_energy'] = self.predators[-1].energy
        self.debug_data['death_ratio'] = dead_prey_num/self.dt
        
        self.time += self.dt


class Model:
    def __init__(self) -> None:
        self.simulator = None
    
    def generate_simulator(self, arena_width, prey_num, phero_width, predator_speed):
        self.simulator = Simulator(None, [0, 0], arena_width, arena_width,
                                   prey_nums=prey_num, phero_width=phero_width,
                                   predator_speed=predator_speed)

if __name__ == "__main__":
    model = Model()
    model.generate_simulator(20, 20, 20, 4)