from PyQt5.QtWidgets import (QDialog, QApplication, QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage

from matplotlib import pyplot as plt

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
        vid = cv2.VideoCapture('bgSub.mp4')
        while not vid.isOpened():
            vid = cv2.VideoCapture('bgSub.mp4')
            cv2.waitKey(1000)
            print("Wait for the header")
        
        pos_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
        
        while(True):
            ret, frame = vid.read()
            if ret:
                cv2.imshow('Origin', frame)
                pos_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                vid.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                print("frame is not ready")
                cv2.waitKey(1000)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())