from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget,QMainWindow, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtWidgets import QLabel, QFileDialog, QPushButton, QLineEdit, QPlainTextEdit, QSpinBox,QVBoxLayout,QSlider
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread,QMutex
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
from ModelLoader import ModelLoader
from DataLoader import DataLoader
import tensorflow as tf
import matplotlib
from scipy.signal import savgol_filter
matplotlib.use('Qt5Agg')
mutex =QMutex()
set_frame = 0
change = False
class MplCanvas(FigureCanvasQTAgg,):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        super(MplCanvas, self).__init__(fig)
        self.scatter_x=None
        self.scatter_y=None
        self.hline = None
        self.vline = None
    def scatter_values(self,scatter_x,scatter_y):
        self.scatter_y = scatter_y
        self.scatter_x = scatter_x
        self.err_plot = self.axes.plot(scatter_x,scatter_y)
        self.axes.set_ylim(0,np.max(scatter_y))
        self.axes.set_xlim(0,np.array(scatter_y).shape[0])
    def draw_v_line(self,value):

            if not(self.vline is None):
                for i,fig in  enumerate(self.axes.collections):
                    if fig == self.vline:
                        self.axes.collections.pop(i)
                        break
            top,bottom = self.axes.get_ylim()
            self.vline = self.axes.vlines([value],top,bottom,'black',"solid")

    def draw_h_line(self,value):
            if not(self.hline is None):
                for i,fig in enumerate(self.axes.collections):
                    if fig ==self.hline:
                        self.axes.collections.pop(i)
                        break
            left,right= self.axes.get_xlim()
            top,bottom = self.axes.get_ylim()
            line_pos = 0.01*value*(abs(top-bottom)) + min(top,bottom)
            self.hline = self.axes.hlines([line_pos],left,right,'black',"solid")
            return line_pos

        

class SliderThread(QThread):
    def __init__(self,slider):
        super().__init__()
        self.slider = slider
        self._run_flag = True
    def run(self):
        global set_frame
        while self._run_flag:
            mutex.lock()
            
            local_frame = set_frame
            mutex.unlock()
            self.slider.setValue(int(local_frame))
            
            self.msleep(1000)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(type([np.ndarray,float]))
    tick_signal = pyqtSignal(float)
    # Funkcja dodająca pasek menu do okna
    

    def __init__(self,video=None):
        super().__init__()
        self._run_flag = True
        self.video = video

        if not video is None:
            try:
                self.frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = 1/video.get(cv2.CAP_PROP_FPS)
            except:
                return
    def run(self):
        # capture from web cam
        
        global set_frame
        global change

        
        while self._run_flag:
            mutex.lock()
            local_set_frame = set_frame
            local_change = change
            mutex.unlock()
            if self.video is None:
                self.change_pixmap_signal.emit([np.zeros_like(np.empty((640,480,3))),0])
                continue
            mutex.lock()
            if local_change:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, int(set_frame*self.frame_count/10000))
                
                change = False
                
            else:
                set_frame =int(10000*self.video.get(cv2.CAP_PROP_POS_FRAMES)/self.frame_count)
            mutex.unlock()
            ret, cv_img = self.video.read()
            if ret:               
                t =[cv_img,(self.video.get(cv2.CAP_PROP_POS_FRAMES)/self.frame_count)] 
                # self.tick_signal.emit(self.video.get(cv2.CAP_PROP_POS_FRAMES)/self.frame_count)
                self.change_pixmap_signal.emit(t)
                self.msleep(int(self.duration*1000))
                
        # shut down capture system
        try:
            self.video.release()
        except:
                return
    def set_video(self,video):
        
        self._run_flag = False
        self.video.release()
        self._run_flag = True
        self.video = video

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QMainWindow):
    frame_signal = pyqtSignal(float)
    def __init__(self, *args, **kwargs):
        super(QMainWindow, self).__init__(*args, **kwargs)
        self.frame_count = 0
        self.setWindowTitle("Anomaly detection")
        self.setMinimumSize(4*480,640)
        self.setMaximumSize(4*480,640)
        self.disply_width = 480
        self.display_height = 640
        self.resize(self.disply_width,self.display_height)
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        self.joints_file = None
        self.model = None
        self.errors = np.empty(0)
        self.treshold = 0
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # self.thread.tick_signal.connect(self.tick_slider)
        # start the thread
        self.thread.start()
        self.createMenu()

        self.joints_file = None
        self.model = None
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        
        
        self.view = QtWidgets.QGridLayout()
        self.time_slider_label = QLabel()
        self.time_slider_label.setText("Time")
        self.time_slider = QSlider(orientation = Qt.Horizontal,parent=self.sc)
        self.time_slider.setMaximum(10000)
        self.time_slider.valueChanged.connect(self.onChange_time_slider)
        self.time_slider.sliderPressed.connect(self.slider_pressed)
        self.time_slider.sliderReleased.connect(self.slider_released)
        self.slider_is_pressed = False
        self.slider_ticker = SliderThread(self.time_slider)
    
        self.slider_ticker.start()
        
        self.treshold_slider = QSlider(orientation = Qt.Vertical,parent=self.sc)
        self.treshold_slider.sliderReleased.connect(self.treshold_slider_released)
        self.treshold_slider.sliderPressed.connect(self.treshold_slider_pressed)
        self.treshold_slider.valueChanged.connect(self.onChange_treshold_slider)
        self.treshold_slider_label = QLabel(parent=self.treshold_slider)
        self.treshold_slider_label.setText("Treshold")
        self.view.setColumnMinimumWidth(0,10)
        self.view.addWidget(self.sc,0,1)
        self.view.addWidget(self.treshold_slider_label,0,2)
        self.view.addWidget(self.treshold_slider,0,3)
        self.view.addWidget(self.time_slider_label,1,1)
        self.view.addWidget(self.time_slider,2,1)
        
        self.view.addWidget(self.image_label,0,0)
        # view.addWidget()
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(self.view)
        self.setCentralWidget(widget)

        self.show()
    def slider_pressed(self):
        self.slider_is_pressed = True
    def slider_released(self):
        self.slider_is_pressed = False

    def treshold_slider_pressed(self):
        self.treshold_slider_is_pressed = True
    def treshold_slider_released(self):
        self.treshold_slider_is_pressed = False

    def onChange_treshold_slider(self):
        if not self.errors.shape[0]==0 and not self.slider_is_pressed:
            self.treshold = self.sc.draw_h_line(self.treshold_slider.value())
            self.treshold_slider_label.setText("Treshold: {:.2f}".format(self.treshold))
            self.sc.draw()
            
    def onChange_time_slider(self):
        if not self.errors.shape[0]==0:
            frame = self.time_slider.value()/10000*self.errors.shape[0]
            self.time_slider_label.setText("Time: {}".format(int(frame)/100))
            self.sc.draw_v_line(frame)
            self.sc.draw()
        if  not self.thread.video is None and self.slider_is_pressed:
            frame = self.time_slider.value()
            global set_frame
            global change 
            mutex.lock()
            set_frame = frame
            change = True
            mutex.unlock()

    def createMenu(self):
        # Stworzenie paska menu
        self.menu = self.menuBar()
        # Dodanie do paska listy rozwijalnej o nazwie File
        self.FileMenu = self.menu.addMenu("File")
        self.FileMenu.addAction('Exit', self.close)
        self.Z1Menu = self.menu.addMenu("Wybierz")
        self.Z1Menu.addAction('Wybierz plik wideo', self.chose_video)
        self.Z1Menu.addAction('Wybierz model', self.chose_model)
        self.Z1Menu.addAction('Wybierz serię czasową', self.chose_joints)
    def chose_joints(self):
        fileName, selectedFilter = QFileDialog.getOpenFileName(self, "Wybierz serię czasową", ".\\Dataset\\Caren\\Connected\\normalized\\Annomaly", "All Files (*.csv);")
        if fileName:
            self.joints_file = pd.read_csv(fileName)
            if not self.model is None:
                s = self.model.input_shape
                loader_data = DataLoader(test_size=0)
                self.data = loader_data.production_load_data( self.joints_file,s[1])
                self.plot()

    def chose_model(self):
        fileName, selectedFilter = QFileDialog.getOpenFileName(self, "Wybierz plik modelu sieci",  ".\\Modele", "All Files (*.json);")
        if fileName:
            loader_model = ModelLoader()
            path =''
            for i in fileName.split('/')[:-1]:
                path+=i+'/'
            self.model = loader_model.Load_model(fileName.split('/')[-1][:5],path)
            loader_data = DataLoader(test_size=0)
            s = self.model.input_shape
            if not self.joints_file is None:
                self.data = loader_data.production_load_data( self.joints_file,s[1])
                self.plot()    
    def chose_video(self):
      global change
      global set_frame
      fileName, selectedFilter = QFileDialog.getOpenFileName(self, "Wybierz plik wideo",  ".\\Dataset\\Caren", "All Files (*);;Python Files (*.py);; PNG (*.png)")
      if fileName:
        self.thread.change_pixmap_signal.disconnect()
        # self.thread.tick_signal.disconnect()
        self.thread.stop()
        mutex.lock()
        change = False
        set_frame = 0
        mutex.unlock()
        self.thread = VideoThread(cv2.VideoCapture(fileName))
        self.thread.change_pixmap_signal.connect(self.update_image)
        # self.thread.tick_signal.connect(self.tick_slider)
        self.thread.start()


    def plot(self):
        self.sc.axes.clear()
        predicts = self.model.predict(self.data).reshape(-1,self.data.shape[-1])
        all_errors = (predicts-(self.data).reshape(-1,self.data.shape[-1]))**2
        self.errors = savgol_filter(np.array(tf.reduce_sum(all_errors,1)),60,1)
        # plot data
        self.sc.scatter_values(range(self.errors.shape[0]),self.errors)
        
        # refresh canvas
        self.sc.draw()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    # def resizeEvent(self, event):
    #     self.resize(self.width(), int((self.width())*640/480))
    #     self.image_label.resize(self.width()-200, int((self.width()-200)*640/480))
    #     # QWidget.resizeEvent(self, event)


    # @pyqtSlot((np.ndarray,int))
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    
    def convert_cv_qt(self, cv_tuple):
        """Convert from an opencv image to QPixmap"""
        # cv_img = cv2.resize(cv_img,(int(self.image_label.width()), int(self.image_label.height())),cv2.INTER_CUBIC)
        cv_img,frame = cv_tuple
        h, w, ch = cv_img.shape
        if not self.errors.shape[0]==0:
            f = int(frame*self.errors.shape[0])-1
            e = self.errors[np.max([f,0])]
            error = e>self.treshold
            if error == False:
                cv2.circle(cv_img,(80,80),13,(72,232,17),-1)
            else:
                cv2.circle(cv_img,(80,80),13,(69,21,237),-1)
        convert_to_Qt_format = QtGui.QImage(cv_img.data, w, h,QtGui.QImage.Format.Format_BGR888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    sys.exit(app.exec())