import datetime

import cv2
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QVector3D, QTextCursor
import cv2
import numpy as np
import pyqtgraph as pg

from Ui_main import Ui_mainWindow

class Window(QMainWindow, Ui_mainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.s_arena_width.valueChanged.connect(self.update_sliders)
        self.s_predator_speed.valueChanged.connect(self.update_sliders)
        self.s_phero_width.valueChanged.connect(self.update_sliders)
        self.s_prey_num.valueChanged.connect(self.update_sliders)
        self.s_prey_speed.valueChanged.connect(self.update_sliders)
        
        # self.label_arena.setScaledContents(True)
        self.logger_str_header = {'error': '--Err: ', 'info': '-Info: ', 'warning': '-Warn: '}
        self.logger_str_color = {'error': 'red', 'info': 'green', 'warning': 'orange'}
        
        # PLOTS
        self.data_plots = []
        for i in range(1, 5):
            exec("self.plot{}_vl = QVBoxLayout(self.wp_{})".format(i,i))
            exec("self.plot{} = pg.PlotWidget()".format(i))
            eval("self.plot{}.setBackground('k')".format(i))
            eval("self.plot{}.showGrid(x=True, y=True)".format(i))
            eval("self.plot{}_vl.addWidget(self.plot{})".format(i,i))
        
    
        item = pg.PlotCurveItem(symbol='d', size=10, pen=pg.mkPen('y'))
        self.data_plots.append(item)
        self.plot1.addItem(item)
        
        item = pg.BarGraphItem(x=range(20), height=np.zeros(20), width=0.3, brush='y')
        self.data_plots.append(item)
        self.plot2.addItem(item)
        
        item = pg.BarGraphItem(x=range(20), height=np.zeros(20), width=0.3, brush='y')
        self.data_plots.append(item)
        self.plot3.addItem(item)
        
        item = pg.PlotCurveItem(symbol='d', size=10, pen=pg.mkPen('r'))
        self.data_plots.append(item)
        self.plot4.addItem(item)
    
    def update_sliders(self):
        sender = self.sender()
        obj_s = sender.objectName()[2:]
        v = eval("self.s_{}.value()".format(obj_s))
        eval("self.lb_{}.setText(str(v))".format(obj_s))
    
    def transfer_CVimg2label(self, img):
        # transfer the cv2 image to PyQt image to display on the GUI
        if len(img.shape) == 3:
            # color image
            img = cv2.resize(np.swapaxes(img,1,0), dsize=None, fx=1, fy=1)
            qt_img = QImage(img.data,
                            img.shape[1],
                            img.shape[0],
                            img.shape[1] * 3,
                            QImage.Format_RGB888)
        else:
            # gray image
            img = cv2.resize(np.swapaxes(img,1,0), dsize=None, fx=1, fy=1)
            qt_img = QImage(img.data,
                            img.shape[1],
                            img.shape[0],
                            img.shape[1],
                            QImage.Format_Grayscale8)
            
        return qt_img
    
    def update_images_plot(self, img):
        qt_img = self.transfer_CVimg2label(img)
        self.label_arena.setPixmap(QPixmap.fromImage(qt_img))
    
    def update_data_plots(self, data):
        # plot 1
        if self.cb_p_1.currentText() == "Prey-AverageEnergy":
            self.data_plots[0].setData(x = np.array(range(len(data['energy']))),
                                       y = np.array(data['energy']).reshape([-1, data['prey_num']]).mean(axis=1))
        elif self.cb_p_1.currentText() == "Predator-Energy":
            self.data_plots[0].setData(x = np.array(range(len(data['pd_energy']))),
                                       y = np.array(data['pd_energy']))
        # plot 4
        if self.cb_p_4.currentText() == "Prey-AverageEnergy":
            self.data_plots[3].setData(x = np.array(range(len(data['energy']))),
                                       y = np.array(data['energy']).reshape([-1, data['prey_num']]).mean(axis=1))
        elif self.cb_p_4.currentText() == "Predator-Energy":
            self.data_plots[3].setData(x = np.array(range(len(data['pd_energy']))),
                                       y = np.array(data['pd_energy']))
        elif self.cb_p_4.currentText() == "Prey-DeathRatio":
            y = np.convolve(np.array(data['death_ratio']), np.ones(10) / 10, mode='valid')
            self.data_plots[3].setData(x = np.array(range(len(y))),
                                       y = y)
        elif self.cb_p_4.currentText() == "Predator-PheroRadius":
            self.data_plots[3].setData(x = np.array(range(len(data['pd_phero_ratio']))),
                                       y = np.array(data['pd_phero_ratio']))
        
        # plot 2
        if self.cb_p_2.currentText() == "Prey-FGatherDistribution":
            d_ = data['f_gather'][-1]
            t_d,_ = np.histogram(d_, bins=20, range=(0.02, 0.32))
            self.data_plots[1].setOpts(x=np.linspace(0.02, 0.32, 20), height=t_d, width=0.3/20)
            self.plot2.setRange(xRange=[0.01, 0.33])
        elif self.cb_p_2.currentText() == "Prey-FAvoidDistribution":
            d_ = data['f_avoid'][-1]
            t_d,_ = np.histogram(d_, bins=20, range=(0.01, 10.01))
            self.data_plots[1].setOpts(x=np.linspace(0.01, 10.01, 20), height=t_d, width=0.5)
        
        # plot 3
        if self.cb_p_3.currentText() == "Prey-FGatherDistribution":
            d_ = data['f_gather'][-1]
            t_d,_ = np.histogram(d_, bins=20, range=(0.02, 0.32))
            self.data_plots[2].setOpts(x=np.linspace(0.02, 0.32, 20),height=t_d, width=0.3/20)
            self.plot3.setRange(xRange=[0.01, 0.33])
        elif self.cb_p_3.currentText() == "Prey-FAvoidDistribution":
            d_ = data['f_avoid'][-1]
            t_d,_ = np.histogram(d_, bins=20, range=(0.01, 10.01))
            self.data_plots[2].setOpts(x=np.linspace(0.01, 10.01, 20),height=t_d, width=0.5)
                
    def system_logger(self, log, log_type='info'):
        time_e = datetime.datetime.now()
        time_e_s = datetime.datetime.strftime(time_e, '%Y-%m-%d %H:%M:%S')[-8:]

        if log_type in self.logger_str_header.keys():
            header = self.logger_str_header[log_type]
            color = self.logger_str_color[log_type]
        else:
            header = '^-^'
            color = 'black'

        log_str = time_e_s + header + log
        s = "<font color=%s>%s</font><br>" % (color, log_str)
        
        cursor = self.te_logger.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.te_logger.setTextCursor(cursor)
        self.te_logger.insertHtml(s)