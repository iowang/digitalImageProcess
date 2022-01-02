from PyQt5 import QtWidgets
import sys
# from digitalImageProcess.showImageController import MainWindow_controller
from digitalImageProcess.imageProcessController import MainWindow_controller

if __name__ == '__main__':
    app=QtWidgets.QApplication(sys.argv)
    window=MainWindow_controller()
    window.show()
    sys.exit(app.exec_())