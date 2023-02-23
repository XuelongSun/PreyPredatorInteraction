import threading
import time
import numpy as np
import cv2
import base64

import glfw

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from model import Simulator

thread = None
thread_lock = threading.Lock()

app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"
socketio = SocketIO(app)

simulation_parameters = {'arena_width':200,'prey_num':20,
                         'predator_speed':6, 'pheromone_width':30}

@app.route("/")
def home():
    return render_template("index.html")

@socketio.on("connect")
def on_connect():
    print("Client connected")

@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")

@socketio.on("start_simulation")
def start_simulation(parameters):
    for k in simulation_parameters.keys():
        simulation_parameters[k] = int(parameters[k])
    global thread
    thread = socketio.start_background_task(my_simulation_function)
    # Start the simulation in a separate thread
    # thread = threading.Thread(target=my_simulation_function)
    # thread.start()
    # Emit an event to the client indicating that the simulation has started
    emit("simulation_started")
    socketio.sleep(0.1)

@socketio.on("stop_simulation")
def stop_simulation():
    # Stop the simulation by setting a global flag
    global stop_simulation_flag
    stop_simulation_flag = True
    # Emit an event to the client indicating that the simulation has stopped
    emit("simulation_stopped")
    

def my_simulation_function():
    global stop_simulation_flag
    global x
    global socketio
    x = 0
    stop_simulation_flag = False

    
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(640, 480, "pyrender", None, None)
    glfw.make_context_current(window)

    simulation = Simulator(None, [0, 0],
                           simulation_parameters['arena_width'],
                           simulation_parameters['arena_width'],
                           prey_nums=simulation_parameters['prey_num'],
                           phero_width=simulation_parameters['pheromone_width'],
                           predator_speed=simulation_parameters['predator_speed'])
    # Emit an event to the client indicating that the simulation has started
    # socketio.emit("simulation_started")

    while not stop_simulation_flag:
        simulation.step()
        simulation.visualization2d()
        # Convert the OpenCV image to a base64-encoded string
        _, buffer = cv2.imencode('.jpg',
                                 cv2.cvtColor(simulation.visualize_img,
                                              cv2.COLOR_BGR2RGB)
                                 )
        img_str = base64.b64encode(buffer).decode()        
        socketio.emit("update_image", {'image': img_str})

        t_gather,_ = np.histogram(simulation.debug_data['f_gather'], bins=20, range=(0.02, 0.32))
        t_avoid,_ = np.histogram(simulation.debug_data['f_avoid'], bins=20, range=(0.01, 10.01))
        
        results = {"x":simulation.time,
                   "y11":sum(simulation.debug_data['energy'])/simulation_parameters['prey_num'], 
                   "y12":simulation.debug_data['pd_energy'],
                   'x_gather':np.linspace(0.02, 0.32, 20).tolist(),
                   'y_gather':t_gather.tolist(),
                   'x_avoid':np.linspace(0.01, 10.01, 20).tolist(),
                   'y_avoid':t_gather.tolist(),
                   }

        # Emit the results back to the client for visualization
        socketio.emit("update_plot", results)

        # Sleep for a short time before running the next step
        socketio.sleep(0.1)
    
    glfw.destroy_window(window)

if __name__ == "__main__":
    glfw.init()
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)
    # socketio.run(app, debug=True, host='192.168.31.141', port=8080)
    # socketio.run(app, debug=True)


