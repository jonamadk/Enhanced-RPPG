import cv2
import numpy as np
# from PyQt4.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from PyQt5 import QtTest

import pyqtgraph as pg
import sys
# import time
# from process import Process
# from webcam import Webcam
# from video import Video
# from interface import waitKey, plotXY

class GUI(QMainWindow, QThread):
    def __init__(self):
        super(GUI, self).__init__()
        self.initUI()

    def initUI(self):
    	

        # set font
        font = QFont()
        font.setPointSize(15)

        # widgets
        self.btnStart = QPushButton("Start", self)
        self.btnStart.move(50, 550)             #(y,x)
        self.btnStart.setFixedWidth(200)
        self.btnStart.setFixedHeight(50)
        self.btnStart.setFont(font)
#        self.btnStart.clicked.connect(self.run)
        self.btnStart.clicked.connect(self.selectInput)
        
        self.btnStop = QPushButton("Stop", self)
        self.btnStop.move(370, 550)
        self.btnStop.setFixedWidth(200)
        self.btnStop.setFixedHeight(50)
        self.btnStop.setFont(font)
#        self.btnStart.clicked.connect(self.stop)
        
        self.optionInput = QComboBox(self)
        self.optionInput.addItem("Webcam")
        self.optionInput.addItem("Video")
        self.optionInput.setCurrentIndex(0)
        self.optionInput.setFixedWidth(200)
        self.optionInput.setFixedHeight(50)
        self.optionInput.move(210, 620)
        self.optionInput.setFont(font)
#        self.optionInput.activated.connect(self.selectInput)

        #label to show notice	
        self.lbnotice = QLabel(self) 
        self.lbnotice.setGeometry(50, 10, 500, 30)
        self.lbnotice.setFont(font)
        self.lbnotice.setText("Please face toward the camera")
 
        #label to show stable HR       
        self.lblHR2 = QLabel(self) 
        self.lblHR2.setGeometry(690,70,300,40)
        self.lblHR2.setFont(font)
        self.lblHR2.setText("Heart rate: ")
        
        #label to show frame from camera
        self.lblDisplay = QLabel(self) 
        self.lblDisplay.setGeometry(10, 50, 640, 480)
        self.lblDisplay.setStyleSheet("background-color: #000000")
            
            
            
        #dynamic plot
        #signal plot
        # self.signal_Plt = pg.PlotWidget(self)
        # self.signal_Plt.move(660,220)
        # self.signal_Plt.resize(480,192)
        # self.signal_Plt.setLabel('bottom', "Signal")

        #config main window
        self.setGeometry(100, 100, 1000, 740)							#(xpos,ypos,width,height)
        #self.center()
        self.setWindowTitle("Heart rate monitor")
        self.show()		

    def selectInput(self):
        if self.optionInput.currentIndex() == 0:
            print("Input: webcam")
            a=0
            self.run(a)
        elif self.optionInput.currentIndex() == 1:
            print("Input: video")
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
            if fileName:
                print(fileName)
            #dialog box for opening the video file
         
        #    capture_value='C:/Users/Hukka/Desktop/test videos/videorec.mp4'
            if len(fileName) >0:           #check for empty string and to remove error 
                file_path=fileName.replace('/','\\')
                self.run(fileName)
        
        
    def run(self,catch):
        cap = cv2.VideoCapture(catch)
        
        while (True):
            # Capture frame-by-frame
            ret,frame = cap.read()
            if ret == True:
                print('camera is on')
                self.display_frame(frame,1)
                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Display the resulting frame
                cv2.imshow('frame', gray)
#                self.lblDisplay.setPixmap(self.show1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.lblDisplay.clear()
                    break
            else:
                print('Return not found')
                break
  
        # When everything done, release the capture
        self.lblDisplay.clear()
        cap.release()
        cv2.destroyAllWindows()
        
    def display_frame(self,frame,window=1):
        qformat = QImage.Format_Indexed8
        
        if len(frame.shape) == 3:
            if (frame.shape[2] == 4):
                qformat = QImage.Format_RGBA8888
                print('qimage')
            else:
                qformat =QImage.Format_RGB888
        # create QImage from image
        frame = QImage(frame,frame.shape[1],frame.shape[0],qformat)
        frame = frame.rgbSwapped()
        # show image in img_label
        self.lblDisplay.setPixmap(QPixmap.fromImage(frame))
#        self.lblDisplay.setAlignment(QtCore.Qt.AlignhCenter | Qtcore.Qt.AlignVCenter)
        
        
##void main
if __name__ == '__main__':
    app = QApplication(sys.argv)                #runnig application and find system os(win or linux)
    ex = GUI()                                  #Call GUi class to make Gui Structure
    # while ex.status == True:
    #     ex.main_loop()

    sys.exit(app.exec_())