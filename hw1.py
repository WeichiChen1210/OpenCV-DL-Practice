from PyQt5.QtWidgets import (QDialog, QApplication, QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage

import cv2
import numpy as np
import sys
import random

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('tkAgg')

PATH = "./Model/model.pth"
EPOCH = 5
BATCH_SIZE = 64
TRAIN_NUMS = 49000
PRINT_FREQ = 100
learning_rate = 0.001
log_interval = 10
optimizer = None
train_loader, test_loader = None, None

center_x = 130
center_y = 125
mouseX = -1
mouseY = -1

def nothing(x):
    pass

############################################## 
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

##############################################
# image operations
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

def gaussian_smooth():
    # 3*3 Gassian filter
    x, y = np.mgrid[-1:2, -1:2]
    gaussian_kernel = np.exp(-(x**2+y**2))
    sum = gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel / sum
    # print(gaussian_kernel)
    return gaussian_kernel

##############################################
# dataset preparations

data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# prepare train data
train_data = datasets.MNIST('./', train=True, download=True, transform=data_transform)
# global train_loader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader_all = DataLoader(train_data, batch_size=10000, shuffle=True)

# prepare test data
test_data = datasets.MNIST('./', train=False, download=True, transform=data_transform)
# global test_loader
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader_all = DataLoader(test_data, batch_size=10000, shuffle=True)

val_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(TRAIN_NUMS, 50000)))

def random_show_image():
    global train_loader
    pics = enumerate(train_loader_all)
    batch_idx, (data, labels) = next(pics)

    fig = plt.figure('10 Random Images')
    for i in range(10):
        index = random.randint(0, 10000)
        plt.subplot(1, 10, i+1)
        plt.tight_layout()
        plt.imshow(data[index][0], cmap='gray', interpolation='none')
        plt.title("{}".format(labels[index]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

##############################################
# Trainer class
class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.device = device
        self.accuracy = 0
        self.loss = 0
        self.loss_list = []
        self.iter_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_loop(self, model, train_loader, val_loader):
        self.train_acc_list.clear()
        self.loss_list.clear()

        for epoch in range(EPOCH):
            print("---------------- Epoch {} ----------------".format(epoch+1))
            self._training_step(model, train_loader, epoch)

            # validation
            self._validate(model, val_loader, epoch)

            # testing
            self._validate(model, test_loader, epoch, state="Testing")
    
    def test(self, model, test_loader):
            print("---------------- Testing ----------------")
            self._validate(model, test_loader, 0, state="Testing")
            
    def _training_step(self, model, loader, epoch):
        model.train()
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            N = X.shape[0]
            
            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            # save loss every iteration
            self.iter_loss_list.append(loss)

            if step >= 0 and (step % PRINT_FREQ == 0):
                self._state_logging(outs, y, loss, step, epoch, "Training")
            
            loss.backward()
            self.optimizer.step()
            
    def _validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]
                
                outs = model(X)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)
        
        # Save testing accuracy every epoch
        if state == "Testing":
            self.test_acc_list.append(self.accuracy)
        else:
            self.train_acc_list.append(self.accuracy)
            self.loss_list.append(self.loss)
        self.accuracy = 0
        self.loss = 0                
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCH, state, step, loss, acc))
        
        # save accuracy and loss every 100 iterations
        self.accuracy = acc
        self.loss = loss

    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc

##############################################
# LeNet5 CNN models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten(x):    
    # x = torch.flatten(x, start_dim=1)
    x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    return x

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1, self.conv2 = None, None
        self.fc1, self.fc2, self.fc3 = None, None, None
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3  = nn.Linear(84, 10)
    
    def forward(self, x):
        out = None
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        out = flatten(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

model = LeNet5()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=learning_rate, momentum=0.9)

##############################################
# GUI
class Window(QWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__()

        grid = QGridLayout()
        grid.addWidget(self.image_processing_group(), 0, 0)
        grid.addWidget(self.adaptive_threshold_group(), 0, 1)
        grid.addWidget(self.image_transformation_group(), 0, 2)
        grid.addWidget(self.convolution_group(), 0, 3)
        grid.addWidget(self.training_group(), 0, 4)
        self.setLayout(grid)

        self.setFixedSize(1000, 300)
        self.setWindowTitle("HW1")

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

    def image_transformation_group(self):
        groupBox = QGroupBox("3. imageTransformation")
        groupBox1 = QGroupBox("3.1 Rotate, scale, translate")
        groupBox2 = QGroupBox("Parameters")
        
        # parameters
        label1 = QLabel()
        label1.setText('Angle (degree): ')
        self.line1 = QLineEdit()
        label2 = QLabel()
        label2.setText('Scale:')
        self.line2 = QLineEdit()
        label3 = QLabel()
        label3.setText('Tx (pixel): ')
        self.line3 = QLineEdit()
        label4 = QLabel()
        label4.setText('Ty (pixel): ')
        self.line4 = QLineEdit()

        hbox1 = QHBoxLayout()
        hbox1.addWidget(label1)
        hbox1.addWidget(self.line1)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(label2)
        hbox2.addWidget(self.line2)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(label3)
        hbox3.addWidget(self.line3)
        hbox4 = QHBoxLayout()
        hbox4.addWidget(label4)
        hbox4.addWidget(self.line4)

        vbox2 = QVBoxLayout()
        vbox2.addLayout(hbox1)
        vbox2.addLayout(hbox2)
        vbox2.addLayout(hbox3)
        vbox2.addLayout(hbox4)
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

    def training_group(self):
        groupBox = QGroupBox("5. Training on LeNet5")
        push1 = QPushButton("5.1 Show train images")
        push1.clicked.connect(self.show_train_image)
        push2 = QPushButton("5.2 Show hyperparameters")
        push2.clicked.connect(self.show_hyperpara)
        push3 = QPushButton("5.3 Train 1 epoch")
        push3.clicked.connect(self.train_one_epoch)        
        push4 = QPushButton("5.4 Show training result")
        push4.clicked.connect(self.show_train_result)
        push5 = QPushButton("5.5 Inference")
        push5.clicked.connect(self.inference)
        label = QLabel()
        label.setText('Test Image Index: ')
        self.line1 = QLineEdit()
        self.line1.setPlaceholderText('0~9999')

        hbox = QHBoxLayout()
        hbox.addWidget(label)
        hbox.addWidget(self.line1)
        vbox = QVBoxLayout()
        vbox.addWidget(push1)
        vbox.addWidget(push2)
        vbox.addWidget(push3)
        vbox.addWidget(push4)
        vbox.addLayout(hbox)
        vbox.addWidget(push5)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    @pyqtSlot()
    # Problem 1
    def load_image(self):
        img = cv2.imread('./images/images/dog.bmp', -1)
        cv2.imshow('picture', img)
        # self.nd = showPicture()
        # self.nd.show()
    
    def convert_color(self):
        # [:, :, 0] b [:, :, 1] g [:, :, 2] r
        img = cv2.imread('./images/images/color.png', -1)
        temp = img.copy()
        img[:, :, 0] = temp[:, :, 1].copy()
        img[:, :, 1] = temp[:, :, 2].copy()
        img[:, :, 2] = temp[:, :, 0].copy()
        cv2.imshow('picture', img)
        # self.nd = colorConvert()
        # self.nd.show()

    def flip_image(self):
        img = cv2.imread('./images/images/dog.bmp', -1)
        img = cv2.flip(img, 1)
        cv2.imshow('picture', img)
        # self.nd = imgFipping()
        # self.nd.show()
    
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

    # Problem 2
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

    # Problem 3
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
    
    # Problem 4
    def gaussian(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)
        # convert to grayscale
        gray = rgb_to_gray(origin_img)

        # generate filter
        gaussian_kernel = gaussian_smooth()

        # convolution
        result = convolution(gray, gaussian_kernel)
        cv2.imshow('Gaussian Smooth', result)

    def sobel_x(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)

        # convert to grayscale
        gray = rgb_to_gray(origin_img)

        # generate filter
        gaussian_kernel = gaussian_smooth()
        # print(gaussian_kernel)

        # convolution for Gaussian Smooth
        smooth = convolution(gray, gaussian_kernel)

        # sobel operator x
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # convolution for Sobel X
        result = convolution(smooth, Gx)
        
        cv2.imshow('Sobel X', result)

    def sobel_y(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)
        # convert to grayscale
        gray = rgb_to_gray(origin_img)

        # generate filter
        gaussian_kernel = gaussian_smooth()

        # convolution for Gaussian Smooth
        smooth = convolution(gray, gaussian_kernel)

        # sobel y
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # convolution for Sobel Y
        result = convolution(smooth, Gy)
        
        cv2.imshow('Sobel Y', result)

    def magnitude(self):
        origin_img = cv2.imread('./images/images/School.jpg')
        cv2.imshow('Origin', origin_img)
        
        # convert to gray
        gray = rgb_to_gray(origin_img)

        # generate filter
        gaussian_kernel = gaussian_smooth()
        # print(gaussian_kernel)

        # convolution
        smooth = convolution(gray, gaussian_kernel)

        # sobel
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        result_x = convolution(smooth, Gx)
        result_y = convolution(smooth, Gy)

        # calculate magnitude
        mag = np.zeros_like(smooth)
        mag = np.sqrt(result_x ** 2 + result_y ** 2)
        # cv2.imshow('X', result_x)
        # cv2.imshow('Y', result_y)
        cv2.imshow('Magnitude', mag)

    # Problem 5    
    def show_train_image(self):
        # dataset_initializer()
        random_show_image()
    
    def show_hyperpara(self):
        print('hyperparameters:')
        print('batch size: ', BATCH_SIZE)
        print('learning rate: ', learning_rate)
        print('optimizer: SGD')

    def train_one_epoch(self):
        # training
        global EPOCH
        EPOCH = 1
        trainer = Trainer(criterion, optimizer, device)
        trainer.train_loop(model, train_loader, val_loader)
        # trainer.test(model, test_loader)

        loss_list = trainer.iter_loss_list
        num = len(loss_list)
        # get list of loss
        loss = []
        for item in loss_list:
            loss.append(item.item())
        trainer.iter_loss_list.clear()
        # plot loss
        plt.plot(range(num), loss)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title('Epoch[1/50]')
        plt.show()

    def train(self):
        global EPOCH
        global model
        EPOCH = 50
        
        # initialize trainer
        trainer = Trainer(criterion, optimizer, device)
        # start training
        trainer.train_loop(model, train_loader, val_loader)
        # trainer.test(model, test_loader)

        # get accuracy and loss lists
        tr_acc = trainer.train_acc_list
        te_acc = trainer.test_acc_list
        loss_list = trainer.loss_list
        loss = []
        train_acc = []
        test_acc = []
        for item in loss_list:
            loss.append(item.item())
        for item in tr_acc:
            train_acc.append(item.item())
        for item in te_acc:
            test_acc.append(item.item())
        
        # multiply by 100
        train_acc = [i * 100 for i in train_acc]
        test_acc = [i * 100 for i in test_acc]
        
        # plot
        plt.figure('Accuracy')
        plt.ylim(0, 100)
        plt.plot(range(1, EPOCH+1), train_acc, label='train')
        plt.plot(range(1, EPOCH+1), test_acc, label='test')
        plt.xlabel('EPOCH')
        plt.ylabel('%')
        plt.title('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig('./accuracy.png')

        plt.figure('Loss')
        plt.plot(range(1, EPOCH+1), loss)
        plt.xlabel('EPOCH')
        plt.ylabel('loss')
        plt.title('Loss')
        plt.savefig('./loss.png')
        # plt.show()

        # save model
        torch.save(model.state_dict(), PATH)
    
    def show_train_result(self):
        acc = cv2.imread('./accuracy.png')
        loss = cv2.imread('./loss.png')
        cv2.imshow('accuracy', acc)
        cv2.imshow('loss', loss)

    def inference(self):
        # get QLineEdit content
        num = int(self.line1.text())

        # load model
        loaded_model = LeNet5()
        loaded_model.load_state_dict(torch.load(PATH))
        loaded_model.cuda()
        loaded_model.eval()

        # setup test dataset
        batch_idx, (data, labels) = next(enumerate(test_loader_all))
        data = data.to(device)
        
        # run model
        with torch.no_grad():
            output = loaded_model(data)
        
        # convert to probability
        probability = F.softmax(output.data[num], dim=0).tolist()
        
        # plot 
        plt.figure(1)
        plt.imshow(data.cpu()[num][0], cmap='gray', interpolation='none')

        plt.figure(2)        
        plt.bar(range(0, 10), probability)
        plt.ylim(0, 1)
        plt.xlim(0, 9)
        plt.show()        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())