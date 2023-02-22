import sys
import os
from viewer import Window
from model import Model
import glfw
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from OpenGL.GL import *

class SimulationRunner(QThread):
    msg = pyqtSignal(str)
    def __init__(self, simulator):
        super().__init__()
        self.pause = False
        self.stop_f = False
        self.simulator = simulator
        glfw.window_hint(glfw.VISIBLE, False)
        self.gl_window = glfw.create_window(640, 480, "pyrender", None, None)
        
    def run(self):
        glfw.make_context_current(self.gl_window)
        while not self.stop_f:
            if not self.pause:
                self.simulator.step()
                self.simulator.visualization2d()
                if self.simulator.time % 60 <= self.simulator.dt:
                    self.msg.emit('step_done')
        glfw.destroy_window(self.gl_window)
            
class Controller:
    def __init__(self) -> None:
        self.window = Window()
        self.model = Model()
        
        
        aw, num, pw, ps = self.construct_simulation()
        self.initial_paras = (aw, num, pw, ps)
        
        self.sim_data = {'energy':[], 'f_gather':[], 'f_avoid':[], 'pd_energy':[],
                         'death_ratio':[]}
        self.sim_data['prey_num'] = num
        

        self.sim_already_started = False
        
        self.window.pb_start.clicked.connect(self.start_simulation)
        self.window.pb_reset.clicked.connect(self.reset_simulation)
        self.window.pb_stop.clicked.connect(self.stop_simulation)
        
    def update_arena_image(self):
        # self.model.simulator.visualization2d()
        self.window.update_images_plot(self.model.simulator.visualize_img)
    
    def update_data_plots(self):
        f = False
        for k, v in self.model.simulator.debug_data.items():
            self.sim_data[k].append(v)
            f = True
        if f:
            self.window.update_data_plots(self.sim_data)

    def construct_simulation(self):
        num = self.window.s_prey_num.value()
        a_w = self.window.s_arena_width.value()
        p_w = self.window.s_phero_width.value()
        p_s = self.window.s_predator_speed.value()
        self.model.generate_simulator(a_w, num, p_w, p_s)
        return a_w, num, p_w, p_s
    
    def reset_simulation(self):
        aw, num, pw, ps = self.initial_paras
        self.window.s_arena_width.setValue(aw)
        self.window.s_prey_num.setValue(num)
        self.window.s_phero_width.setValue(pw)
        self.window.s_predator_speed.setValue(ps)
        self.window.system_logger("Simulation reset successfully.")
    
    def start_simulation(self):
        sender = self.window.sender().text()
        if sender == 'Start':
            if not self.sim_already_started:
                a_w, num, p_w, p_s = self.construct_simulation()
                self.sim_data['prey_num'] = num
                self.sim_thread = SimulationRunner(self.model.simulator)
                self.sim_thread.msg.connect(self.simulation_msg_process)
                self.sim_thread.start()
                self.window.pb_stop.setEnabled(True)
                self.sim_already_started = True
            else:
                self.sim_thread.pause = False
            self.window.pb_start.setText("Pause")
        elif sender == 'Pause':
            self.sim_thread.pause = True
            self.window.pb_start.setText("Start")
        else:
            pass
    
    def simulation_msg_process(self, msg):
        if msg == 'step_done':
            self.update_arena_image()
            self.update_data_plots()
        elif msg == 'error':
            pass
        
    def stop_simulation(self):
        self.sim_thread.stop_f = True
        self.sim_already_started = False
        self.window.pb_stop.setEnabled(False)
        self.sim_data = {'energy':[], 'f_gather':[], 'f_avoid':[], 'pd_energy':[],
                         'death_ratio':[]}
        num = self.window.s_prey_num.value()
        self.sim_data['prey_num'] = num
        self.window.pb_start.setText("Start")

if __name__ == '__main__':
    prey_predator_gui = QApplication(sys.argv)
    glfw.init()
    ctl = Controller()
    ctl.window.show()
    sys.exit(prey_predator_gui.exec())