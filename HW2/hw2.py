from PyQt5.QtWidgets import (QDialog, QApplication, QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('tkAgg')

import cv2
import numpy as np
import sys

class Window(QWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__()

        grid = QGridLayout()
        grid.addWidget(self.stereo(), 0, 0)
        grid.addWidget(self.bg_substraction(), 0, 1)
        grid.addWidget(self.feature_tracking(), 0, 2)
        grid.addWidget(self.augmented_reality(), 0, 3)
        self.setLayout(grid)

    def stereo(self):
        groupBox = QGroupBox("1. Stereo")
        push1 = QPushButton("1.1 Disparity", self)
        push1.clicked.connect(self.disparity)

        vbox = QVBoxLayout()
        vbox.addWidget(push1, 0)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def bg_substraction(self):
        groupBox = QGroupBox("2. Background Substraction")
        push1 = QPushButton("2.1 Background substraction")
        push1.clicked.connect(self.substraction)
        vbox = QVBoxLayout()
        vbox.addWidget(push1, 0)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    def feature_tracking(self):
        groupBox = QGroupBox("3. Feature Tracking")
        push1 = QPushButton("3.1 Preprocessing")
        push2 = QPushButton("3.2 Video Tracking")

        vbox = QVBoxLayout()
        vbox.addWidget(push1, 0)
        vbox.addWidget(push2, 1)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def augmented_reality(self):
        groupBox = QGroupBox("4. Augmented Reality")
        push1 = QPushButton("4.1 Augmented Reality")

        vbox = QVBoxLayout()
        vbox.addWidget(push1, 0)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    @pyqtSlot()

    def disparity(self):
        imgL = cv2.imread('imL.png', 0)
        imgR = cv2.imread('imR.png', 0)

        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
        disp = stereo.compute(imgL, imgR)
        plt.imshow(disp, 'gray')
        plt.show()

    def substraction(self):
        # read input video
        vid = cv2.VideoCapture('bgSub.mp4')

        # check if it's ready to be read
        while not vid.isOpened():
            vid = cv2.VideoCapture('bgSub.mp4')
            cv2.waitKey(1000)
            print("Wait for the header")
        
        # current frame
        pos_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

        # background subtractor object
        # backSub = cv2.createBackgroundSubtractorKNN()
        backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=192, detectShadows=False)
        # history: number of frames to be trained, varThreshold; threshold, detectShadow: detect shadow or not

        while(True):
            # read a frame
            ret, frame = vid.read()
            # if frame is ready
            if ret:
                # current frame
                pos_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

                # apply subtractor
                fgMask = backSub.apply(frame)
                # get the frame number and write it on the current frame
                cv2.rectangle(frame, (10, 2), (100, 2), (255, 255, 255), -1)
                cv2.putText(frame, str(pos_frame), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                # show in 2 windows
                cv2.imshow('Origin', frame)
                cv2.imshow('Mask', fgMask)
            # if frame is not ready
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                print("frame is not ready")
                cv2.waitKey(1000)

            if cv2.waitKey(37) & 0xFF == ord('q'):
                break

            # if the end of video, break
            if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                break
        
        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())