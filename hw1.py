from PyQt5.QtWidgets import (QDialog, QApplication, QCheckBox, QGridLayout, QGroupBox, QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QLabel)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
import numpy as np
import sys

def nothing(x):
    pass

# open a new dialog, read picture with opencv and show image with Qt
class showPicture(QDialog):
    def __init__(self):
        super(showPicture, self).__init__()
        self.img = np.ndarray(())
        self.setWindowTitle('picture')
        # dialog element setup
        self.image_frame = QLabel()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_frame)
        self.setLayout(self.layout)

        # read image with opencv
        self.img = cv2.imread('./images/images/dog.bmp', -1)
        height, width, bytesPerComponent = self.img.shape
        bytesPerLine = 3 * width
        print("Height = ", height)
        print("Width = ", width)

        # convert opencv image to Qt image and show
        self.img = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(self.img))

class colorConvert(QDialog):
    def __init__(self):
        super(colorConvert, self).__init__()
        self.img = np.ndarray(())
        self.setWindowTitle('picture')
        # dialog element setup
        self.image_frame = QLabel()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_frame)
        self.setLayout(self.layout)

        self.img = cv2.imread('./images/images/color.png', -1)
        temp = self.img.copy()
        # [:, :, 0] b [:, :, 1] g [:, :, 2] r
        self.img[:, :, 0] = temp[:, :, 1].copy()
        self.img[:, :, 1] = temp[:, :, 2].copy()
        self.img[:, :, 2] = temp[:, :, 0].copy()
        # read image with opencv
        height, width, bytesPerComponent = self.img.shape
        bytesPerLine = 3 * width

        # convert opencv image to Qt image and show
        self.img = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(self.img))

class imgFipping(QDialog):
    def __init__(self):
        super(imgFipping, self).__init__()
        self.img = np.ndarray(())
        self.setWindowTitle('picture')
        # dialog element setup
        self.image_frame = QLabel()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_frame)
        self.setLayout(self.layout)

        self.img = cv2.imread('./images/images/dog.bmp', -1)
        self.img = cv2.flip(self.img, 1)
        # read image with opencv
        height, width, bytesPerComponent = self.img.shape
        bytesPerLine = 3 * width

        # convert opencv image to Qt image and show
        self.img = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QPixmap.fromImage(self.img))



class Window(QWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__()

        grid = QGridLayout()
        grid.addWidget(self.image_processing_group(), 0, 0)
        grid.addWidget(self.adaptive_threshold_group(), 1, 0)
        grid.addWidget(self.image_processing_group(), 0, 1)
        grid.addWidget(self.convolution_group(), 0, 2)
        self.setLayout(grid)

        self.setFixedSize(640, 480)
        self.setWindowTitle("Image Processing")
        self.resize(400, 300)

    def image_processing_group(self):
        groupBox = QGroupBox("1. Image Processing")
        push1 = QPushButton("1.1 Load Image")
        push1.clicked.connect(self.load_image)
        push2 = QPushButton("1.2 Color Conversion")
        push2.clicked.connect(self.convert_color)
        push3 = QPushButton("1.3 Image Flipping")
        push3.clicked.connect(self.flip_image)
        push4 = QPushButton("1.4 Blending")
        push4.clicked.connect(self.blend_image)

        vbox = QVBoxLayout()
        vbox.addWidget(push1)
        vbox.addWidget(push2)
        vbox.addWidget(push3)
        vbox.addWidget(push4)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def adaptive_threshold_group(self):
        groupBox = QGroupBox("2. Adaptive Threshold")
        push1 = QPushButton("2.1 Global Threshold")
        push1.clicked.connect(self.global_threshold)
        push2 = QPushButton("2.2 Local Threshold")
        push2.clicked.connect(self.local_threshold)

        vbox = QVBoxLayout()
        vbox.addWidget(push1)
        vbox.addWidget(push2)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def convolution_group(self):
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
    
    def image_transformation_group(self):
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
    def load_image(self):
        # img = cv2.imread('./images/images/dog.bmp', -1)
        # cv2.imshow('picture', img)
        self.nd = showPicture()
        self.nd.show()
    
    def convert_color(self):
        # [:, :, 0] b [:, :, 1] g [:, :, 2] r
        # img = cv2.imread('./images/images/color.png', -1)
        # temp = img.copy()
        # img[:, :, 0] = temp[:, :, 1].copy()
        # img[:, :, 1] = temp[:, :, 2].copy()
        # img[:, :, 2] = temp[:, :, 0].copy()
        # cv2.imshow('picture', img)
        self.nd = colorConvert()
        self.nd.show()

    def flip_image(self):
        # img = cv2.imread('./images/images/dog.bmp', -1)
        # img = cv2.flip(img, 1)
        # cv2.imshow('picture', img)
        self.nd = imgFipping()
        self.nd.show()
    
    def blend_image(self):
        cv2.namedWindow('Blend')
        cv2.createTrackbar('blend', 'Blend', 0, 100, nothing)
        img_origin = cv2.imread('./images/images/dog.bmp', -1)
        img_flipped = cv2.flip(img_origin, 1)
        while(1):
            k = cv2.waitKey(1)
            if k == 27:
                break
            percentage = cv2.getTrackbarPos('blend', 'Blend') / 100
            dst = cv2.addWeighted(img_origin, 1-percentage, img_flipped, percentage, 0.0)
            cv2.imshow('Blend', dst)
            
        cv2.destroyAllWindows()

    def global_threshold(self):
        origin_img = cv2.imread('./images/images/QR.png', 0)
        cv2.imshow('Original image', origin_img)
        ret,thresh1 = cv2.threshold(origin_img, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('Threshold image', thresh1)

    def local_threshold(self):
        origin_img = cv2.imread('./images/images/QR.png', 0)
        cv2.imshow('Original image', origin_img)
        th1 = cv2.adaptiveThreshold(origin_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1)
        cv2.imshow('Threshold image', th1)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())