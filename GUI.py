import os
import sys
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from utils_cv2.utils_yolo import YOLO
from PyQt5.QtGui import QPixmap, QImage
from utils_cv2.utils import cvtColor, resize_image
from PyQt5.QtWidgets import QApplication, QMessageBox, QGraphicsScene, QGraphicsPixmapItem

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.init_slots()

        self.originalimage = None
        self.resultimage = None
        self.image_name = None

        self.max_width = 800
        self.zoomscale = 1
        self.recognize_flag = 0  # 识别标志位

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1394, 906)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_1.setStyleSheet("QPushButton{font-size:23px}")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_1.sizePolicy().hasHeightForWidth())
        self.pushButton_1.setSizePolicy(sizePolicy)
        self.pushButton_1.setObjectName("pushButton_1")
        self.verticalLayout.addWidget(self.pushButton_1)
        
        self.verticalLayout.setContentsMargins(50,-1,50,100)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setStyleSheet("QPushButton{font-size:23px}")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setStyleSheet("QPushButton{font-size:23px}")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setEnabled(True)
        self.textEdit.setStyleSheet("QPushButton{font-size:23px}")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setReadOnly(True)

        self.textEdit.setFontPointSize(15)
        self.horizontalLayout.addWidget(self.textEdit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1394, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate

        MainWindow.setWindowTitle(_translate("MainWindow", "焊缝缺陷检测系统"))
        self.pushButton_1.setText(_translate("MainWindow", "打开"))
        self.pushButton_2.setText(_translate("MainWindow", "识别"))
        self.pushButton_3.setText(_translate("MainWindow", "另存为"))

    def init_slots(self):
        self.pushButton_1.clicked.connect(self.button1_open)
        self.pushButton_2.clicked.connect(self.button2_open)
        self.pushButton_3.clicked.connect(self.button3_open)

    def button1_open(self):
        self.recognize_flag = 1  # 打开识别标志位

        self.graphicsView.setScene(None)  # 清空画布内容
        self.textEdit.setText(None) # 清空识别文本内容
        self.originalimage = None  # 清空原始图片临时变量
        self.resultimage = None  # 清空识别结果临时变量

        img_name, stauts = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", 'Image Files(*.png *.jpg *.jpeg)')

        if stauts:
            self.originalimage = Image.open(img_name)  # 打开图像
            height, width = np.array(np.shape(self.originalimage)[0:2])  # 获取图像尺寸

            # 限制图像大小
            if width > self.max_width:
                scale = self.max_width / width
                self.originalimage = resize_image(self.originalimage, (width * scale, height * scale))
                width, height = int(width * scale), int(height * scale)

            (self.image_name, _) = os.path.splitext(os.path.basename(img_name))   #获取选择的图片的图片名前缀

            img = self.originalimage  # 用于显示的图像
            img = cvtColor(img)  # 转化为RGB格式
            img = np.array(img)  # 转化为numpy格式

            frame = QImage(img, width, height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)

            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.item.setScale(self.zoomscale)
            self.scene = QGraphicsScene()  # 创建场景
            self.scene.addItem(self.item)

            self.graphicsView.setScene(self.scene)  # 将场景添加至视图
            self.textEdit.setText("您已打开 " + img_name + " 待检测")


    def button2_open(self):
        self.pushButton_1.setEnabled(False)  # 冻结打开按钮

        # 如果画布为空就点击识别-弹出警告（避免进入系统就点击识别）
        if self.graphicsView.scene() == None:       
            QMessageBox.critical(self, "错误", "请先选择焊缝图片进行检测！")
            self.pushButton_1.setEnabled(True)  # 解冻打开按钮
            return 0
        
        # 提示防止多次点击识别按钮
        if self.recognize_flag == 0: 
            QMessageBox.information(self, "注意", "请勿多次点击识别按钮", QMessageBox.Yes)
            return 0
        
        self.recognize_flag = 0  # 关闭识别按钮
        yolo = YOLO()
        img, detetxt= yolo.detect_image(self.originalimage)  # 根据原图临时变量进行识别检测，得到识别结果图像与文本
        self.resultimage = img  # 将最终检测结果添至缓存

        img = np.array(img)  # 将图像转化为numpy形式
        width = img.shape[1]  # 获取图像大小
        height = img.shape[0]
        frame = QImage(img, width, height, QImage.Format_RGB888)

        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.graphicsView.setScene(self.scene)  # 将场景添加至视图
        self.textEdit.setText(detetxt)  #展示识别文本结果

        self.pushButton_1.setEnabled(True)  # 解冻打开按钮

    def button3_open(self):
        if self.graphicsView.scene() == None:   #如果画布为空就点击识别-弹出警告（避免进入系统就点击另存为）
            QMessageBox.critical(self, "错误", "请先选择焊缝图片进行检测！")
            return 0

        if self.resultimage == None:   #避免还未进行识别就另存为
            QMessageBox.critical(self, "错误", "还未进行识别，请先识别后再保存结果！")
            return 0
               
        filename, status = QtWidgets.QFileDialog.getSaveFileName(self, "另存为", "", "PNG Files(*.png);;JPG Files(*.jpg);;JPEG Files(*.jpeg)", )
        if status:
            self.resultimage.save(filename)
            QMessageBox.information(self, "提示", "成功保存文件为" + filename , QMessageBox.Yes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())