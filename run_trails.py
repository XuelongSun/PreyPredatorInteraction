import os
import time

from simulator import Simulator

ARENA_WIDTH = 400
ARENA_HEIGHT = 400
ARENA_LOCATION = [0, 0]
TIMEOUT = 60

# simulation parameters
num_trails = 10
results_folder = './results/'
exp_type = 1.1

# 1. inter-specific
## 1.1 predator's velocity
if exp_type == 1.1:
    file_dir = results_folder + "inter_predator_velocity/"
    velocity = [1,2,4,6,8]
    for v in velocity:
        filename = file_dir + 'velocity_{}/'.format(v)
        if not os.path.exists(filename):
            os.makedirs(filename)
        for i in range(num_trails):
            print('*start {}th trail of predator velocity = {}......'.format(i, v),
                  end='')
            sim = Simulator(TIMEOUT, ARENA_LOCATION,
                            ARENA_WIDTH, ARENA_HEIGHT,
                            prey_nums=30)
            sim.predators[0].linear_velocity = v
            filename_t = filename + "{}.mat".format(i)
            st = time.time()
            sim.run(save_data=True, filename=filename_t)
            et = time.time()
            print(' time cost: {:.6f}s'.format(et - st))
## 1.2 predator's pheromone size (diffusion)
elif exp_type == 1.2:
    file_dir = results_folder + "inter_predator_phero/"
# 2. intra-specific
## 2.1 number of preys
elif exp_type == 2.1:
    file_dir = results_folder + "intra_prey_density/"
    numbers = range(10, 50, 5)
    for n in numbers:
        filename = file_dir + 'num_{}/'.format(n)
        if not os.path.exists(filename):
            os.makedirs(filename)
        for i in range(num_trails):
            print('*start {}th trail of prey number = {}......'.format(i, n),
                  end='')
            sim = Simulator(TIMEOUT, ARENA_LOCATION,
                            ARENA_WIDTH, ARENA_HEIGHT,
                            prey_nums=n)
            filename_t = filename + "{}.mat".format(i)
            st = time.time()
            sim.run(save_data=True, filename=filename_t)
            et = time.time()
            print(' time cost: {:.6f}s'.format(et - st))
## 2.2 prey's velocity
elif exp_type == 2.1:
    file_dir = results_folder + "intra_prey_density/"