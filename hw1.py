from PyQt5.QtWidgets import (QDialog, QApplication, QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
import numpy as np
import sys
import math

center_x = 130
center_y = 125
mouseX = -1
mouseY = -1

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

def rgb_to_gray(origin_img):
    r_channel = origin_img[:, :, 0].copy()
    g_channel = origin_img[:, :, 0].copy()
    b_channel = origin_img[:, :, 0].copy()
    gray = 0.2126 * (r_channel/255) + 0.7152 * (g_channel/255) + 0.0722 * (b_channel/255)
    return gray

def convolution(gray, gaussian_kernel):
    result = np.zeros_like(gray)
    image_padded = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2))
    image_padded[1:-1, 1:-1] = gray
    for x in range(gray.shape[1]):
        for y in range(gray.shape[0]):
            result[y, x] = (gaussian_kernel * image_padded[y:y+3, x:x+3]).sum()
    return result

class Window(QWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__()

        grid = QGridLayout()
        grid.addWidget(self.image_processing_group(), 0, 0)
        grid.addWidget(self.adaptive_threshold_group(), 0, 1)
        grid.addWidget(self.image_transformation_group(), 0, 2)
        grid.addWidget(self.convolution_group(), 0, 3)
        self.setLayout(grid)

        self.setFixedSize(900, 450)
        self.setWindowTitle("Image Processing")

        self.points = []
        self.img = np.zeros((512,512,3), np.uint8)

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x, y), 10, (0, 0, 255), -1)
            # print(x, y)
            self.points.append((x, y))

    def image_processing_group(self):
        groupBox = QGroupBox("1. Image Processing")
        push1 = QPushButton("1.1 Load Image", self)
        # push1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        push1.clicked.connect(self.load_image)
        push2 = QPushButton("1.2 Color Conversion")
        # push2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        push2.clicked.connect(self.convert_color)
        push3 = QPushButton("1.3 Image Flipping")
        # push3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        push3.clicked.connect(self.flip_image)
        push4 = QPushButton("1.4 Blending")
        # push4.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        push4.clicked.connect(self.blend_image)

        vbox = QVBoxLayout()
        vbox.addWidget(push1, 0)
        vbox.addWidget(push2, 1)
        vbox.addWidget(push3, 2)
        vbox.addWidget(push4, 5)
        # vbox.addStretch(1)
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
        push1.clicked.connect(self.gaussian)
        push2 = QPushButton("4.2 Sobel X")
        push2.clicked.connect(self.sobel_x)
        push3 = QPushButton("4.3 Sobel Y")
        push3.clicked.connect(self.sobel_y)
        push4 = QPushButton("4.4 Magnitude")
        push4.clicked.connect(self.magnitude)

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
        groupBox1 = QGroupBox("3.1 Rotate, scale, translate")
        groupBox2 = QGroupBox("Parameters")
        
        # parameters
        label1 = QLabel()
        label1.setText('Angle (degree)')
        self.line1 = QLineEdit()
        label2 = QLabel()
        label2.setText('Scale')
        self.line2 = QLineEdit()
        label3 = QLabel()
        label3.setText('Tx (pixel)')
        self.line3 = QLineEdit()
        label4 = QLabel()
        label4.setText('Ty (pixel)')
        self.line4 = QLineEdit()

        vbox2 = QVBoxLayout()
        vbox2.addWidget(label1)
        vbox2.addWidget(self.line1)
        vbox2.addWidget(label2)
        vbox2.addWidget(self.line2)
        vbox2.addWidget(label3)
        vbox2.addWidget(self.line3)
        vbox2.addWidget(label4)
        vbox2.addWidget(self.line4)
        vbox2.addStretch(1)
        groupBox2.setLayout(vbox2)

        # 3.1
        self.push1 = QPushButton("3.1 Rotation, scaling, translation")
        self.push1.clicked.connect(self.transformation)
        vbox1 = QVBoxLayout()
        vbox1.addWidget(groupBox2)
        vbox1.addWidget(self.push1)
        vbox1.addStretch(1)
        groupBox1.setLayout(vbox1)

        # 3.2
        self.push2 = QPushButton("3.2 Perspective Transform")
        self.push2.clicked.connect(self.perspective)
        vbox = QVBoxLayout()
        vbox.addWidget(groupBox1)
        vbox.addWidget(self.push2)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)
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

    def transformation(self):
        angle = float(self.line1.text())
        scale = float(self.line2.text())
        tx = float(self.line3.text())
        ty = float(self.line4.text())

        origin_img = cv2.imread('./images/images/OriginalTransform.png')
        cv2.imshow('Original image', origin_img)
        img = origin_img.copy()
        rows, cols = img.shape[:2]
        
        # rotation and scaling
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
        result = cv2.warpAffine(img, M, (cols, rows))
        # translation
        H = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(result, H, (cols, rows))

        print(result.shape[:2])
        cv2.imshow('Rotation + Scaling + Translation', result)

    def perspective(self):
        self.img = cv2.imread('./images/images/OriginalPerspective.png')
        result = None
        pts1, pts2 = None, None
        img = self.img.copy()
        cv2.namedWindow('Original image')
        cv2.setMouseCallback('Original image', self.draw_circle)
        pts2 = np.float32([[20, 20], [450, 20], [450, 450], [20, 450]])
        flag = False

        while(1):
            cv2.imshow('Original image', self.img)
            if flag:
                break
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

            if len(self.points) == 4:
                pts1 = np.float32(self.points)
                flag = True
                # print(pts1)
                # print(pts2)
                
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (450, 450), flags=cv2.INTER_LINEAR)
        cv2.imshow('Perspective Result image', result)
        
    def gaussian(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)
        # convert to grayscale
        gray = rgb_to_gray(origin_img)
        # cv2.imshow('grayscale', gray)

        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        sum = gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel / sum
        # print(gaussian_kernel)

        # convolution
        result = convolution(gray, gaussian_kernel)
        cv2.imshow('Gaussian Smooth', result)

    def sobel_x(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)

        # convert to grayscale
        gray = rgb_to_gray(origin_img)

        # sobel operator x
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # print(Gx)

        # convolution
        result = convolution(gray, Gx)
        # print(result.shape)
        cv2.imshow('Sobel X', result)

    def sobel_y(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)
        # convert to grayscale
        gray = rgb_to_gray(origin_img)

        # sobel y
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # print(Gy)

        # convolution
        result = convolution(gray, Gy)
        # print(result.shape)
        cv2.imshow('Sobel Y', result)

    def magnitude(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)
        
        # convert to gray
        gray = rgb_to_gray(origin_img)

        # sobel
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        result_x = convolution(gray, Gx)
        result_y = convolution(gray, Gy)

        # mag: result array
        mag = np.zeros_like(gray)
        mag = np.sqrt(result_x ** 2 + result_y ** 2)
        cv2.imshow('Magnitude', mag)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())