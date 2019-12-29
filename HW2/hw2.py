from PyQt5.QtWidgets import (QDialog, QApplication, QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QImage

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('tkAgg')

import cv2
import numpy as np
import sys

def draw_square(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(param[0], (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
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
        ret, frame = video.read()

        param = [frame, self.points]
        cv2.namedWindow('Preprocess')
        cv2.setMouseCallback('Preprocess', draw_square, param)

        while True:
            cv2.imshow('Preprocess', frame)

            if len(self.points) > 6:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(self.points)
        cv2.waitKey(1500)        
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

if __name__ == "__main__":
    count = 0
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())