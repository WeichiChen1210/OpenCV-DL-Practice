from PyQt5.QtWidgets import (QDialog, QApplication, QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('tkAgg')

import cv2
import numpy as np
import sys
import time

def draw_pyramid(img, imgpts):
    # base
    img = cv2.line(img, (imgpts[1][0][0],imgpts[1][0][1]),(imgpts[2][0][0],imgpts[2][0][1]), (0, 0, 255), 6)
    img = cv2.line(img, (imgpts[2][0][0],imgpts[2][0][1]),(imgpts[3][0][0],imgpts[3][0][1]), (0, 0, 255), 6)
    img = cv2.line(img, (imgpts[3][0][0],imgpts[3][0][1]),(imgpts[4][0][0],imgpts[4][0][1]), (0, 0, 255), 6)
    img = cv2.line(img, (imgpts[4][0][0],imgpts[4][0][1]),(imgpts[1][0][0],imgpts[1][0][1]), (0, 0, 255), 6)

    # vertex
    for i in range(4):
        img = cv2.line(img, (imgpts[0][0][0],imgpts[0][0][1]),(imgpts[i+1][0][0],imgpts[i+1][0][1]), (0, 0, 255), 6)

    return img

def draw_square(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw red square ont the frame
        cv2.rectangle(param[0], (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
        # save the point
        param[1].append((x, y))

class Window(QWidget):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__()

        self.points = []
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
        push1.clicked.connect(self.preprocessing)
        push2 = QPushButton("3.2 Video Tracking")
        push2.clicked.connect(self.tracking)
        
        vbox = QVBoxLayout()
        vbox.addWidget(push1, 0)
        vbox.addWidget(push2, 1)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    
    def augmented_reality(self):
        groupBox = QGroupBox("4. Augmented Reality")
        push1 = QPushButton("4.1 Augmented Reality")
        push1.clicked.connect(self.augmented)

        vbox = QVBoxLayout()
        vbox.addWidget(push1, 0)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

    @pyqtSlot()

    def disparity(self):
        imgL = cv2.imread('imL.png')
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        imgR = cv2.imread('imR.png')
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

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
        backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=190, detectShadows=False)
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
                cv2.imshow('Foreground', fgMask)
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

    def preprocessing(self):
        video = cv2.VideoCapture('featureTracking.mp4')
        # take out the first frame
        ret, frame = video.read()

        # parameters to be passed to mouse callback function
        param = [frame, self.points]
        cv2.namedWindow('Preprocess')
        cv2.setMouseCallback('Preprocess', draw_square, param)

        while True:
            cv2.imshow('Preprocess', frame)

            # if 7 points, break
            if len(self.points) > 6:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(self.points)
        cv2.waitKey(1000)        
        video.release()
        cv2.destroyAllWindows()

    def tracking(self):
        if len(self.points) == 0:
            print("not preprocessed")
            return
        # turn the point list to np array with shape (n, 1, 2)
        p0 = np.zeros((len(self.points), 1, 2), dtype='f')  # if not turned to float32 may cause error
        p0[:, 0] = self.points[:]

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # LK params
        lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # read the first frame
        video = cv2.VideoCapture('featureTracking.mp4')
        ret, old_frame = video.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        mask = np.zeros_like(old_frame)

        while(1):
            ret, frame = video.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            # show frame
            cv2.imshow('Optical flow',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            
            if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
                print(video.get(cv2.CAP_PROP_FRAME_COUNT))
                break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
        
        cv2.waitKey(1000)
        video.release()
        cv2.destroyAllWindows()

    def augmented(self):
        # 3D points
        temp = [[3, 3, -4], [1, 1, 0], [1, 5, 0], [5, 5, 0], [5, 1, 0]]
        objp = np.array(temp, dtype='f')
        
        # intrinsic
        temp = [[2225.49585, 0, 1025.5459589],
                [0, 2225.18414074, 1038.58518846], 
                [0, 0, 1]]
        mtx = np.array(temp)
        
        # distortion
        temp = [[-0.12874225, 0.09057782, -0.00099125, 0.00000278, 0.0022925]]
        distortion = np.array(temp)
        
        # extrinsics
        extrinsic = []

        temp = [[-0.97157425, -0.01827487, 0.23602862,  6.81253889],
                [ 0.07148055, -0.97312723, 0.2188925,   3.37330384],
                [ 0.22568565,  0.22954177, 0.94677165, 16.71572319]]
        bmp1 = np.array(temp)
        extrinsic.append(bmp1)

        temp = [[-0.8884799, -0.14530922, -0.435303,    3.3925504 ],
                [0.07148066, -0.98078915,  0.18150248,  4.36149229],
                [-0.45331444, 0.13014556,  0.88179825, 22.15957429]]
        bmp2 = np.array(temp)
        extrinsic.append(bmp2)

        temp = [[-0.52390938,  0.22312793, 0.82202974,  2.68774801],
                [ 0.00530458, -0.96420621, 0.26510046,  4.70990021],
                [ 0.85175747,  0.14324914, 0.50397308, 12.98147662]]
        bmp3 = np.array(temp)
        extrinsic.append(bmp3)

        temp = [[-0.63108673,  0.53013053, 0.566296,    1.22781875],
                [ 0.13263301, -0.64553994, 0.75212145,  3.48023006],
                [ 0.76428923,  0.54976341, 0.33707888, 10.9840538 ]]
        bmp4 = np.array(temp)
        extrinsic.append(bmp4)

        temp = [[-0.87676843, -0.23020567,  0.42223508,  4.43641198],
                [ 0.19708207, -0.97286949, -0.12117596,  0.67177428],
                [ 0.43867502, -0.02302829,  0.89835067, 16.24069227]]
        bmp5 = np.array(temp)
        extrinsic.append(bmp5)

        # for each of 5 images
        for i in range(5):
            path = str(i+1) + '.bmp'
            img = cv2.imread(path, -1)

            # split rotation vector and translation vector
            rvec = extrinsic[i][:, :3]
            tvec = extrinsic[i][:, 3]

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(objp, rvec, tvec, mtx, distortion)

            # draw the pyramid
            img = draw_pyramid(img, imgpts)

            # resize
            resize = cv2.resize(img, (700, 700))
            cv2.imshow('Projection ' + str(i+1), resize)
            # time.sleep(0.5)
            cv2.waitKey(500)

        return

if __name__ == "__main__":
    count = 0
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())