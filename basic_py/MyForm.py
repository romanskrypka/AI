# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\CPO_V\OneDrive\Документы\GitHub\AI\basic_py\MyForm.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MyForm(object):
    def setupUi(self, MyForm):
        MyForm.setObjectName("MyForm")
        MyForm.resize(631, 358)
        self.pushButton = QtWidgets.QPushButton(MyForm)
        self.pushButton.setGeometry(QtCore.QRect(260, 170, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(MyForm)
        self.label.setGeometry(QtCore.QRect(250, 130, 101, 20))
        self.label.setObjectName("label")

        self.retranslateUi(MyForm)
        QtCore.QMetaObject.connectSlotsByName(MyForm)

    def retranslateUi(self, MyForm):
        _translate = QtCore.QCoreApplication.translate
        MyForm.setWindowTitle(_translate("MyForm", "Моя первая форма"))
        self.pushButton.setText(_translate("MyForm", "Приветствие"))
        self.label.setText(_translate("MyForm", "Здесь будет текст"))