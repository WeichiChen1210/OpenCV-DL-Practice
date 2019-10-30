from PyQt5.QtWidgets import (QDialog, QApplication, QCheckBox, QGridLayout, QGroupBox, QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QLabel)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap
import cv2
import numpy as np
import sys

class Picture(QDialog):
    def __init__(self):
        super(Picture, self).__init__()
        self.img = np.ndarray(())
        self.setWindowTitle('picture')
        self.img = cv2.imread('./images/images/dog.bmp', -1)
        if self.img.size == 0:
            print('failed')
            return
        print(self.img.size)
        # label = QLabel(self)
        # pixmap = QPixmap('./images/images/dog.bmp')
        # label.setPixmap(pixmap)
        # self.resize(pixmap.width(), pixmap.height())
        # print("Height: ", pixmap.height())
        # print("Width: ", pixmap.width())


class Window(QWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__()

        grid = QGridLayout()
        grid.addWidget(self.imageProcessingGroup(), 0, 0)
        grid.addWidget(self.adaptiveThresholdGroup(), 1, 0)
        grid.addWidget(self.imageProcessingGroup(), 0, 1)
        grid.addWidget(self.convolutionGroup(), 0, 2)
        self.setLayout(grid)

        self.setFixedSize(640, 480)
        self.setWindowTitle("Image Processing")
        self.resize(400, 300)

    def imageProcessingGroup(self):
        groupBox = QGroupBox("1. Image Processing")
        push1 = QPushButton("1.1 Load Image")
        push1.clicked.connect(self.loadImage)
        push2 = QPushButton("1.2 Color Conversion")
        push3 = QPushButton("1.3 Image Flipping")
        push4 = QPushButton("1.4 Blending")

        vbox = QVBoxLayout()
        vbox.addWidget(push1)
        vbox.addWidget(push2)
        vbox.addWidget(push3)
        vbox.addWidget(push4)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def adaptiveThresholdGroup(self):
        groupBox = QGroupBox("2. Adaptive Threshold")
        push1 = QPushButton("2.1 Global Threshold")
        push2 = QPushButton("2.2 Local Threshold")

        vbox = QVBoxLayout()
        vbox.addWidget(push1)
        vbox.addWidget(push2)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def convolutionGroup(self):
        groupBox = QGroupBox("4. Convolution")
        push1 = QPushButton("4.1 Gaussian")
        push2 = QPushButton("4.2 Sobel X")
        push3 = QPushButton("4.3 Sobel Y")
        push4 = QPushButton("4.4 Magnitude")

        vbox = QVBoxLayout()
        vbox.addWidget(push1)
        vbox.addWidget(push2)
        vbox.addWidget(push3)
        vbox.addWidget(push4)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def imageTransformationGroup(self):
        groupBox = QGroupBox("3. imageTransformation")

        push1 = QPushButton("3.1 Rotation, scaling, translation")
        groupBox1 = QGroupBox("3.1 Rot, scale, translation")
        vbox1 = QVBoxLayout()
        vbox1.addWidget(push1)
        vbox1.addStretch(1)
        groupBox1.setLayout(vbox1)

        push2 = QPushButton("3.2 Perspective Transform")
        vbox2 = QVBoxLayout()
        vbox2.addWidget(groupBox1)
        vbox2.addWidget(push2)
        vbox2.addStretch(1)
        groupBox1.setLayout(vbox2)

        return groupBox

    

    @pyqtSlot()
    def on_click(self):
        print("clicked")
    
    def loadImage(self):
        self.nd = Picture()
        
        self.nd.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())