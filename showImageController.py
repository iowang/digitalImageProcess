from PyQt5 import QtWidgets, QtGui, QtCore
from UI import helloWorld, showImage
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image, ImageQt
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = showImage.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # self.img_path = 'D:/DipFinalWork/lung.raw'
        self.ui.openPicture.clicked.connect(self.func_openPicture)
        # self.ui.largePicture.clicked.connect(self.func_largePicture)
        # self.display_img()

    def display_img(self,filename):
        f = open(filename, 'rb')
        width = int.from_bytes(f.read(4)[::-1], 'big')
        height = int.from_bytes(f.read(4)[::-1], 'big')
        img1 = []
        for i in range(height):
            one_line = []
            for j in range(width):
                pixel = int.from_bytes(f.read(2)[::-1], 'big')
                one_line.append(pixel / 16)
            img1.append(one_line)
        self.img = Image.fromarray(np.array(img1)).convert('L')
        print(self.img)
        self.qimg = ImageQt.toqimage(self.img)
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))

    def func_openPicture(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*raw')
        if filename is '':
            return
        self.display_img(filename)
