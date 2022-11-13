from threading import Thread
import time
from matplotlib.pyplot import show

import numpy as np
import cv2
import pyrender as pr

from agent import Prey, Predator, scene3d
from pheromone import Pheromone
from utils import world2image_coordinates_transfer
from utils import DataServer

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
            
class Simulator(Thread):
    def __init__(self, time_out, 
                 location, width, height, 
                 obstacles=None):
        
        Thread.__init__(self)
        
        self.time = 0
        self.dt = 30 # ms
        self.time_out = time_out
        boundary = [location[0], location[0] + width,
                    location[1], location[1] + height]
        self.environment = Environment(width, height,
                                       boundary, self.dt,
                                       obstacles)

        
        self.preys = []
        for i in range(32):
            h = np.random.vonmises(0, 100, 1)[0]
            pos = [np.random.randint(30, width-30), np.random.randint(30, height-30)] 
            self.preys.append(Prey(i, h, pos))
        
        self.alive_preys = self.preys.copy()
        self.dead_preys = []
        
        self.predators = []
        # predator releas pheromones
        self.pheromone_inject_k = []
        for i in range(1):
            h = np.random.vonmises(0, 100, 1)[0]
            pos = [width/2, height/2] 
            self.pheromone_inject_k.append([100, 0].copy())
            self.predators.append(Predator(i, h, pos))
        self.pheromone_inject_k = np.array(self.pheromone_inject_k, np.float32)
        
        # visualization
        self.visualize_img = np.zeros((height, width, 3), np.uint8)
        self.visualize_agent_color = {'prey': (135, 206, 235),
                                      'predator':(200, 0, 20)}
        self.visualize_obstacle_color = (125,125,125)
        
        # transfer data
        self.debug_data = {}
    
    def visualization2d(self, show_agent=()):
        # pheromone
        self.visualize_img = np.clip(self.environment.pheromone.field,
                                     0, 255).astype(np.uint8)
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
                    state_str = "{}: {}".format(a.id, a.state[0])
                # state_str = "{}: {}/{:.2f}".format(a.id, a.state[0], a.energy)
                    self.visualize_img = cv2.putText(self.visualize_img, state_str,
                                                    (p[1], p[0]-a.size*2), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.5, (255,255,255))
                    cv2.imshow('{}_view'.format(a.id), cv2.cvtColor(a.view, cv2.COLOR_RGB2BGR))
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
        cv2.imshow('Simulation', cv2.cvtColor(self.visualize_img, cv2.COLOR_RGB2BGR))
    
    def run(self):
        self.time = 0
        end_condition = True if self.time_out is None else (self.time <= self.time_out*1000)
        
        # visualization parameter
        show_agents = (0, 1)
        # alive_agents = filter(lambda p:p.state != 'death', self.preys)
        while end_condition:
            # update prey's state
            prey_pos = []
            energy = []
            for pe in self.alive_preys:
                pe.update(self.dt, self.environment.pheromone.field, self.environment.boundary)
                # print(pe.position, self.environment.boundary)
                p = world2image_coordinates_transfer(pe.position,
                                                     self.environment.boundary)
                prey_pos.append(p)
                # update energy
                energy.append(pe.energy)
                if pe.state == 'death':
                    self.dead_preys.append(pe)
                    self.alive_preys.remove(pe)
                    # spawn a new prey
                    h = np.random.vonmises(0, 100, 1)[0]
                    pos = [100, 100]
                    t_prey = Prey(len(self.preys) + 1, h, pos)
                    # 1.evolving parameters setted as the average value of the group
                    # t_prey.f_gather = np.mean([a.f_gather for a in self.alive_preys])
                    # t_prey.f_avoid = np.mean([a.f_avoid for a in self.alive_preys])

                    # 2.evolving parameters setted with possiblities (agent with max energy has highest possibility)
                    ## get alive preys' energy
                    temp_a_e = [(a, a.energy) for a in self.alive_preys]
                    temp_a_e = sorted(temp_a_e, key=lambda x:x[1])
                    sum_ = sum([a_[1] for a_ in temp_a_e])
                    partial_p = [a_[1]/sum_ for a_ in temp_a_e]
                    p_ = np.random.rand(1)[0]
                    ind = np.where(partial_p < p_)[0][-1] if len(np.where(partial_p < p_)[0]) > 0 else -1
                    # print(ind, len(temp_a_e), temp_a_e)
                    t_prey.f_gather = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_gather
                    t_prey.f_avoid = temp_a_e[min(ind + 1, len(temp_a_e)-1)][0].f_avoid
                    
                    # add to the lists
                    self.preys.append(t_prey)
                    self.alive_preys.append(t_prey)
                    
                    print(len(self.alive_preys), len(self.dead_preys), len(self.preys))

            # render pheromone
            predator_pos = []
            for pd in self.predators:
                pd.update(self.dt, self.environment.pheromone.field, self.environment.boundary)
                p = world2image_coordinates_transfer(pd.position,
                                                    self.environment.boundary)
                predator_pos.append(p)
            self.environment.pheromone.update(predator_pos, 20,
                                              self.pheromone_inject_k)
            
            # cv2-based 2D viualization
            if self.time % 100 < self.dt:
                self.visualization2d(show_agent=show_agents)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
            # debug data
            self.debug_data['energy'] = energy
            self.debug_data['f_avoid'] = [a.f_avoid for a in self.alive_preys]
            self.debug_data['f_gather'] = [a.f_gather for a in self.alive_preys]
            self.time += self.dt


if __name__ == "__main__":
    ARENA_WIDTH = 240
    ARENA_HEIGHT = 240
    ARENA_LOCATION = [0, 0]
    TIMEOUT = 10
    
    
    sim = Simulator(TIMEOUT, ARENA_LOCATION,
                    ARENA_WIDTH, ARENA_HEIGHT)
    
    
    data_server = DataServer(sim.debug_data)
    data_server_process = Thread(target=data_server.run)
    data_server_process.start()
    sim.run()
    data_server.stop = True
    data_server_process.join()
    # viewer3d = pr.Viewer(scene3d, run_in_thread=True)
    # sim.join()
    