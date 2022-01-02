from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
from UI import showImage
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QTransform
from PIL import Image, ImageQt
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
import cv2

from digitalImageProcess import tongtaiFilter


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow_controller, self).__init__()
        self.ui = showImage.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ratio_value = 50  # 用于处理图像缩放扩大
        self.ui.slider_zoom.setProperty("value", 50)  # 设定一开始的放大缩小位置
        self.ratio_rate = 0  # 设定放大倍数
        self.WL_value = 50  # 设定窗位
        self.WW_value = 50  # 设定窗宽
        self.ui.windowLevel.setProperty("value", 50)
        self.ui.windowWidth.setProperty("value", 50)
        self.img_16 = []  # 存储16位的像素
        # file menu
        # open the file
        self.ui.actionopen.triggered.connect(self.read_file)
        # save the file
        self.ui.actionsave.triggered.connect(self.save_file)

        # edit menu
        # enlarge the picture
        self.ui.actionlarge.triggered.connect(self.enlarge_pic)
        # let the picture small
        self.ui.actionsmall.triggered.connect(self.small_pic)
        # make the picture more light
        self.ui.actionlight.triggered.connect(self.light_pic)
        # make the picture more dark
        self.ui.actiondark.triggered.connect(self.dark_pic)
        # rotate the picture
        self.ui.actionrotate.triggered.connect(self.rotate_pic)

        # 直方图统计部分
        # grayhistogram
        self.ui.actiongrayhistogram.triggered.connect(self.grayhistogram)

        # 图像增强部分
        # 非线性锐化(同态滤波)
        self.ui.actionnonlinearsharpen.triggered.connect(self.nonlinearsharpen)
        # 直方图均衡
        self.ui.actionhistogramequal.triggered.connect(self.histogramequal)

        # 滤波部分
        # 高通滤波
        self.ui.actionhighpass.triggered.connect(self.highpass)
        # 低通滤波
        self.ui.actionlowpass.triggered.connect(self.lowpass)

        # 图像变换部分
        # FFT
        self.ui.actionFFT.triggered.connect(self.FFT)

        # 图中的几个按钮操作
        # 图像放大
        self.ui.largePicture.clicked.connect(self.button_large)
        # 图像缩小
        self.ui.smallPicture.clicked.connect(self.button_small)
        # 图像旋转
        self.ui.rotatePicture_p.clicked.connect(self.button_rotate_pos)
        self.ui.rotatePicture_n.clicked.connect(self.button_rotate_neg)
        # 滑动条修改图像大小
        self.ui.slider_zoom.valueChanged.connect(self.getslidervalue)
        # 窗位滑动条
        self.ui.windowLevel.valueChanged.connect(self.getWLvalue)
        # 窗宽滑动条
        self.ui.windowWidth.valueChanged.connect(self.getWWvalue)
        # 窗位大小显示
        self.ui.showWL_label.setText(f"{int(40 * self.WL_value)} ")
        # 窗宽大小显示
        self.ui.showWW_label.setText(f"{int(30 * self.WW_value)} ")

    def read_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'open file *.raw', 'D:/DipFinalWork/', '*raw')
        if filename is '':
            return
        f = open(filename, 'rb')
        self.width = int.from_bytes(f.read(4)[::-1], 'big')
        self.height = int.from_bytes(f.read(4)[::-1], 'big')
        self.img_16 = []
        self.img = []
        img1 = []
        for i in range(self.height):
            one_line = []
            one_line2 = []
            for j in range(self.width):
                pixel = int.from_bytes(f.read(2)[::-1], 'big')
                one_line.append(pixel / 16)
                one_line2.append(pixel)
            img1.append(one_line)
            self.img_16.append(one_line2)
        self.img_16 = np.array(self.img_16)
        self.img = Image.fromarray(np.array(img1)).convert('L')
        self.imgHistogram = np.array(img1)
        self.qimg = ImageQt.toqimage(self.img)
        # self.ui.label.resize(self.width, self.height)
        self.orignal_qpixmap = QPixmap.fromImage(self.qimg)
        self.ui.label.setPixmap(self.orignal_qpixmap)

    def save_file(self):
        save_filename, _ = QFileDialog.getSaveFileName(self, "save file", 'D:/DipFinalWork/', '*raw')
        if save_filename == '':
            return
        save_filename=save_filename+'.raw'
        width = self.width
        height = self.height
        width_byte = width.to_bytes(4, byteorder='little', signed=False)
        height_byte = height.to_bytes(4, byteorder='little', signed=False)
        f = open(save_filename, 'wb')
        f.write(width_byte)
        f.write(height_byte)
        img=np.array(self.img)
        for i in range(height):
            for j in range(width):
                pixel=int(img[i][j]*16).to_bytes(2, byteorder='little', signed=False)
                f.write(pixel)

    def enlarge_pic(self):
        self.button_large()  # 与按钮逻辑一样

    def small_pic(self):
        self.button_small()  # 与按钮逻辑一样

    def light_pic(self):
        pix_light = 30
        imglight = self.imgHistogram + pix_light
        np.minimum(imglight, 255)
        img = Image.fromarray(imglight).convert('L')
        qimg = ImageQt.toqimage(img)
        self.ui.label.setPixmap(QPixmap.fromImage(qimg))

    def dark_pic(self):
        pix_dark = 90
        imgdark = self.imgHistogram - pix_dark
        np.maximum(imgdark, 0)
        img = Image.fromarray(imgdark).convert('L')
        qimg = ImageQt.toqimage(img)
        self.ui.label.setPixmap(QPixmap.fromImage(qimg))

    def rotate_pic(self):
        self.button_rotate_pos()  # 与按钮逻辑一样

    def grayhistogram(self):
        plt.hist(self.img_16.flatten(), 4096, facecolor='green', alpha=0.75)
        plt.show()

    def nonlinearsharpen(self):
        temp, temp16 = tongtaiFilter.test(self.height, self.width, self.img_16)
        self.img_16 = temp16
        self.fulsh_pic(temp)
        self.ui.label.setPixmap(self.orignal_qpixmap)

    def histogramequal(self):
        imgg = self.imgHistogram.astype(int)
        hist, bins = np.histogram(imgg.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf[imgg]
        img_hist = Image.fromarray(np.array(img2)).convert('L')
        self.fulsh_pic(img_hist)
        self.ui.label.setPixmap(self.orignal_qpixmap)
        # qimg = ImageQt.toqimage(img_hist)
        # qpixmap = QPixmap.fromImage(qimg)
        # self.ui.label.setPixmap(qpixmap)

    def highpass(self):
        imgg = self.imgHistogram
        dft = cv2.dft(np.float32(imgg), flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(dft)

        crow, ccol = int(self.height / 2), int(self.width / 2)
        mask = np.ones((self.height, self.width, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

        f = fshift * mask
        ishift = np.fft.ifftshift(f)
        iimg = cv2.idft(ishift)
        res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
        plt.imshow(res, 'gray')

    def lowpass(self):
        imgg = self.imgHistogram
        dft = cv2.dft(np.float32(imgg), flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(dft)

        crow, ccol = int(self.height / 2), int(self.width / 2)
        mask = np.zeros((self.height, self.width, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

        f = fshift * mask
        ishift = np.fft.ifftshift(f)
        iimg = cv2.idft(ishift)
        res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
        plt.imshow(res, 'gray')

    def FFT(self):
        img = self.imgHistogram
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        s1 = (np.abs(np.real(fshift)))
        plt.imshow(s1, 'gray')

    def button_large(self):
        self.ratio_value = min(100, self.ratio_value + 1)
        self.set_img_ratio()

    def button_small(self):
        self.ratio_value = max(0, self.ratio_value - 1)
        self.set_img_ratio()

    def button_rotate_pos(self):
        transform = QTransform()
        transform.rotate(90)
        self.qimg = self.qimg.transformed(transform)
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))

    def button_rotate_neg(self):
        transform = QTransform()
        transform.rotate(-90)
        self.qimg = self.qimg.transformed(transform)
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))

    # 处理滑动条的值
    def getslidervalue(self):
        self.setslidervalue(self.ui.slider_zoom.value() + 1)

    def setslidervalue(self, value):
        self.ratio_value = value
        self.set_img_ratio()

    def getWLvalue(self):
        self.setWLvalue(self.ui.windowLevel.value() + 1)

    def getWWvalue(self):
        self.setWWvalue(self.ui.windowWidth.value() + 1)

    def setWLvalue(self, value):
        self.WL_value = value
        self.set_image_window()

    def setWWvalue(self, value):
        self.WW_value = value
        self.set_image_window()

    # 把图像用到的函数进行封装
    def set_img_ratio(self):
        self.ratio_rate = pow(10, (self.ratio_value - 50) / 50)
        qpixmap_height = self.height * self.ratio_rate  # 放大或缩小的高度
        self.qpixmap = self.orignal_qpixmap.scaledToHeight(qpixmap_height)
        # self.ui.label.resize(self.width, self.height)
        self.ui.label.setPixmap(self.qpixmap)
        self.ui.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

    def set_image_window(self):
        windowlev = self.WL_value * 40
        windowwid = self.WW_value * 30
        gray_start = windowlev - windowwid / 2
        gray_end = windowlev + windowwid / 2
        temp = self.img_16.copy()
        temp = np.array(temp)
        temp[temp > gray_end] = 255
        temp[temp < gray_start] = 0
        temp = (temp - gray_start) / (gray_end - gray_start) * 255
        temp_img = Image.fromarray(temp).convert('L')
        self.fulsh_pic(temp_img)
        self.ui.label.setPixmap(self.orignal_qpixmap)
        self.ui.showWL_label.setText(f"{int(40 * self.WL_value)} ")
        self.ui.showWW_label.setText(f"{int(30 * self.WW_value)} ")

    def fulsh_pic(self, img):
        self.img = img
        self.qimg = ImageQt.toqimage(self.img)
        self.orignal_qpixmap = QPixmap.fromImage(self.qimg)
