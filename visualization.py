from threading import Thread
import socket
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d

from utils import DataReceiver

class ArenaView3D:
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
        self.margin = 0.5
        
        self.figure_config()
    
    def figure_config(self):
        # self.ax.set_axis_off()
        self.ax.set_xlabel('X/cm')
        self.ax.set_ylabel('Y/cm')
        self.ax.set_zlabel('Z/cm')
        
    
    def plot_arena(self, pos, width, height):
        arena = Rectangle(pos, width, height, facecolor=(.3,.3,.3, 0.3), edgecolor=(0,0,0))
        self.ax.add_patch(arena)
        art3d.pathpatch_2d_to_3d(arena, z=0, zdir="z")
        self.ax.set_xlim(pos[0]-self.margin, pos[0] + width + self.margin)
        self.ax.set_ylim(pos[1]-self.margin, pos[1] + height + self.margin)
        self.ax.set_box_aspect((width, height, 20))
        return self.fig, self.ax
    
    def plot_agents(self, agents):
        for a in agents:
            pass


if __name__ == "__main__":
    # connect socket
    data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        data_socket.connect(('127.0.0.1', 6666))
        print('Connected.')
    except TimeoutError:
        print('TCP time out')
        exit()
    
    data_receiver = DataReceiver(data_socket)
    data_receiver.socket_is_connected = True
    data_receiver.start()
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    # fig.tight_layout()
    
    x = range(32)
    # axes[0][0].bar(x, np.zeros(len(x)), edgecolor='skyblue',
    #             fill=False, ls='--', lw=1)
    e_bar = axes[0][0].bar(x, np.ones(len(x)),
                           color='skyblue', alpha=0.5)
    # axes[0][0].set_xlabel('Prey ID')
    axes[0][0].set_ylabel('Energy')
    e_total = []
    e_total_l, = axes[0][1].plot(0, 0, lw=2, color='skyblue')
    axes[0][1].set_title("Preys' total energy")
    # axes[0][1].set_xlabel("Time/step")
    axes[0][1].set_ylabel('Energy')
    axes[0][1].grid()

    f_avoid_bar = axes[1][0].bar(x, np.ones(len(x)),
                                 color='tomato', alpha=0.5)
    # axes[1][0].set_xlabel('Prey ID')
    axes[1][0].set_ylabel('F_avoid')
    
    f_avoid_mean = []
    # f_avoid_mean_l, = axes[1][1].plot(0, 0, lw=2, color='tomato')
    # axes[1][1].set_title("Preys' mean F_avoid")
    # axes[1][1].set_xlabel("Time/step")
    _, _, f_avoid_h = axes[1][1].hist(np.ones(len(x)), bins=20, range=(0, 0.7),
                                      facecolor='tomato')
    axes[1][1].set_ylabel('F_avoid')
    # axes[1][1].grid()
    
    f_gather_bar = axes[2][0].bar(x, np.ones(len(x)),
                                 color='cyan', alpha=0.5)
    axes[2][0].set_xlabel('Prey ID')
    axes[2][0].set_ylabel('F_gather')
    
    f_gather_mean = []
    # f_gather_mean_l, = axes[2][1].plot(0, 0, lw=2, color='cyan')
    # axes[2][1].set_title("Preys' mean F_gather")
    # axes[2][1].set_xlabel("Time/step")
    _, _, f_gather_h = axes[2][1].hist(np.ones(len(x)), bins=20, range=(6, 14),
                                       facecolor='cyan')
    axes[2][1].set_ylabel('F_gather')
    # axes[2][1].grid()
    
    if data_receiver.data is not None:
        data = pickle.loads(data_receiver.data)
        if 'f_avoid' in data.keys():
            f_avoid = data['f_avoid']
        axes[1][1].hist(f_avoid, bins=20, range=(0, 0.7),
                        edgecolor='gray', facecolor='none', alpha=0.5, ls='--', lw=2)
        if 'f_gather' in data.keys():
            f_avoid = data['f_gather']
        axes[2][1].hist(f_avoid, bins=20, range=(6, 14),
                        edgecolor='gray', facecolor='none', alpha=0.5, ls='--', lw=2)
            
    def update(frame):
        if data_receiver.data is not None:
            data = pickle.loads(data_receiver.data)
            
            # energy
            e_t = 0
            if 'energy' in data.keys():
                energy = data['energy']
                for rect, e in zip(e_bar, energy):
                    rect.set_height(e)
                    e_t += e
            axes[0][0].set_ylim(0, np.array(energy).max()+2)
            e_total.append(e_t)
            e_total_l.set_data(range(len(e_total)), e_total)
            axes[0][1].set_xlim([0, len(e_total)])
            axes[0][1].set_ylim([0, max(e_total)*1.2])
            # f_avoid
            fa_t = 0
            if 'f_avoid' in data.keys():
                f_avoid = data['f_avoid']
                for rect, fa in zip(f_avoid_bar, f_avoid):
                    rect.set_height(fa)
                    fa_t += fa
                t_d,_ = np.histogram(f_avoid, bins=len(f_avoid_h), range=(0, 0.7))
                for rect, d in zip(f_avoid_h, t_d):
                    rect.set_height(d)
            
            axes[1][0].set_ylim(0, np.array(f_avoid).max()+2)
            axes[1][1].set_ylim(0, max(t_d)+2)
            f_avoid_mean.append(fa_t/len(f_avoid))
            # f_avoid_mean_l.set_data(range(len(f_avoid_mean)), f_avoid_mean)
            # axes[1][1].set_xlim([0, len(f_avoid_mean)])
            # axes[1][1].set_ylim([min(f_avoid_mean)-0.1,
            #                      max(f_avoid_mean)*1.2])
            
            # f_gather
            fg_t = 0
            if 'f_gather' in data.keys():
                f_gather = data['f_gather']
                for rect, fg in zip(f_gather_bar, f_gather):
                    rect.set_height(fg)
                    fg_t += fg
                t_d, _ = np.histogram(f_gather, bins=len(f_gather_h), range=(6, 14))
                for rect, d in zip(f_gather_h, t_d):
                    rect.set_height(d)
            axes[2][0].set_ylim(0, np.array(f_gather).max()+2)
            f_gather_mean.append(fg_t/len(f_gather))
            axes[2][1].set_ylim(0, max(t_d)+2)
            # f_gather_mean_l.set_data(range(len(f_gather_mean)), f_gather_mean)
            # axes[2][1].set_xlim([0, len(f_gather_mean)])
            # axes[2][1].set_ylim([min(f_gather_mean)-0.1, max(f_gather_mean)*1.2])
                
        else:
            print("waiting data...")
    
    animation = FuncAnimation(fig, update,
                              repeat=True,
                              interval=2000)
    # v = ArenaView3D()
    # fig, ax = v.plot_arena((0, 0), 100, 200)
    plt.show()