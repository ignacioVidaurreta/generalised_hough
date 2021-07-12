import sys
from PyQt5 import sip
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QRadioButton,
    QMainWindow,
    QApplication,
    QPushButton,
    QWidget,
    QAction,
    QTabWidget,
    QVBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QLineEdit

)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QSize
import cv2
import numpy as np
from numpy.lib import histograms
from hough_guil import generalized_hough_guil
from hough_ballard import generalized_hough_ballard


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Generalized Hough'
        self.left, self.top = 0, 0
        self.width, self.height = 1000, 400
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.main_window = MainWindow(self)
        self.setCentralWidget(self.main_window)

        self.show()

class MainWindow(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        self.ref_image = None
        self.image = None
        self.setBasicLayout()

    def setBasicLayout(self):
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tabs.resize(300,200)

        # Add tabs
        self.tabs.addTab(self.tab1,"Loader")
        self.loaderTab()

        self.tab3 = QWidget()
        self.tabs.addTab(self.tab3, "GH Ballard")
        self.ballardTab()

        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "GH Guil")
        self.guilTab()

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def loaderTab(self):
        # Create first tab
        self.tab1.layout = QGridLayout(self)

        self.buttonSourceOpen = QPushButton("Upload Source Image")
        self.buttonSourceOpen.clicked.connect(self.uploadSourceImage)
        self.tab1.layout.addWidget(self.buttonSourceOpen, 0, 0)

        self.buttonRefOpen = QPushButton("Upload Reference Image")
        self.buttonRefOpen.clicked.connect(self.uploadRefImage)
        self.tab1.layout.addWidget(self.buttonRefOpen, 1, 0)

        self.tab1.setLayout(self.tab1.layout)

    def guilTab(self):
        self.tab2.layout = QGridLayout(self)

        self.min_dist_label, self.min_dist_input = QLabel('Min Distance'), QLineEdit()
        self.min_dist_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.min_dist_label, 0, 0)
        self.tab2.layout.addWidget(self.min_dist_input, 0, 1)
        self.min_dist_input.setText("200")

        self.levels_label, self.levels_input = QLabel('Levels'), QLineEdit()
        self.levels_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.levels_label, 1, 0)
        self.tab2.layout.addWidget(self.levels_input, 1, 1)
        self.levels_input.setText("360")

        self.min_angle_label, self.min_angle_input = QLabel('Min Angle'), QLineEdit()
        self.min_angle_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.min_angle_label, 2, 0)
        self.tab2.layout.addWidget(self.min_angle_input, 2, 1)
        self.min_angle_input.setText("0")


        self.max_angle_label, self.max_angle_input = QLabel('Max Angle'), QLineEdit()
        self.max_angle_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.max_angle_label, 2, 2)
        self.tab2.layout.addWidget(self.max_angle_input, 2, 3)
        self.max_angle_input.setText("360")

        self.angle_step_label, self.angle_step_input = QLabel('Angle Step'), QLineEdit()
        self.angle_step_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.angle_step_label, 2, 4)
        self.tab2.layout.addWidget(self.angle_step_input, 2, 5)
        self.angle_step_input.setText("1")

        self.angle_thresh_label, self.angle_thresh_input = QLabel('Angle Thresh'), QLineEdit()
        self.angle_thresh_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.angle_thresh_label, 2, 6)
        self.tab2.layout.addWidget(self.angle_thresh_input, 2, 7)
        self.angle_thresh_input.setText("100")

        self.min_scale_label, self.min_scale_input = QLabel('Min Scale'), QLineEdit()
        self.min_scale_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.min_scale_label, 3, 0)
        self.tab2.layout.addWidget(self.min_scale_input, 3, 1)
        self.min_scale_input.setText("1")

        self.max_scale_label, self.max_scale_input = QLabel('Max Scale'), QLineEdit()
        self.max_scale_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.max_scale_label, 3, 2)
        self.tab2.layout.addWidget(self.max_scale_input, 3, 3)
        self.max_scale_input.setText("1.3")

        self.scale_step_label, self.scale_step_input = QLabel('Scale Step'), QLineEdit()
        self.scale_step_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.scale_step_label, 3, 4)
        self.tab2.layout.addWidget(self.scale_step_input, 3, 5)
        self.scale_step_input.setText("0.05")

        self.scale_thresh_label, self.scale_thresh_input = QLabel('Scale Thresh'), QLineEdit()
        self.scale_thresh_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.scale_thresh_label, 3, 6)
        self.tab2.layout.addWidget(self.scale_thresh_input, 3, 7)
        self.scale_thresh_input.setText("100")

        self.position_thresh_label, self.position_thresh_input = QLabel('Position Thresh'), QLineEdit()
        self.position_thresh_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.position_thresh_label, 4, 0)
        self.tab2.layout.addWidget(self.position_thresh_input, 4, 1)
        self.position_thresh_input.setText("100")

        self.angle_epsilon_label, self.angle_epsilon_input = QLabel('Angle Epsilon'), QLineEdit()
        self.angle_epsilon_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.angle_epsilon_label, 4, 2)
        self.tab2.layout.addWidget(self.angle_epsilon_input, 4, 3)
        self.angle_epsilon_input.setText("1")

        self.xi_label, self.xi_input = QLabel('Xi'), QLineEdit()
        self.xi_label.setStyleSheet("background-color: #d6f9d6")
        self.tab2.layout.addWidget(self.xi_label, 4, 4)
        self.tab2.layout.addWidget(self.xi_input, 4, 5)
        self.xi_input.setText("90")

        self.houghButton = QPushButton("Execute")
        self.houghButton.clicked.connect(self.runGeneralizedHoughGuil)
        self.tab2.layout.addWidget(self.houghButton, 5, 0)

        self.tab2.setLayout(self.tab2.layout)

    def ballardTab(self):
        self.tab3.layout = QGridLayout(self)

        self.levels_label, self.levels_input = QLabel('Levels'), QLineEdit()
        self.levels_label.setStyleSheet("background-color: #d6f9d6")
        self.tab3.layout.addWidget(self.levels_label, 0, 0)
        self.tab3.layout.addWidget(self.levels_input, 0, 1)
        self.levels_input.setText("360")

        self.vote_threshold_label, self.vote_threshold_input = QLabel('Votes Threshold'), QLineEdit()
        self.vote_threshold_label.setStyleSheet("background-color: #d6f9d6")
        self.tab3.layout.addWidget(self.vote_threshold_label, 1, 0)
        self.tab3.layout.addWidget(self.vote_threshold_input, 1, 1)
        self.vote_threshold_input.setText("150")

        self.houghBallardBtn = QPushButton("Execute")
        self.houghBallardBtn.clicked.connect(self.runGeneralizedHoughBallard)
        self.tab3.layout.addWidget(self.houghBallardBtn, 2, 0)

        self.tab3.setLayout(self.tab3.layout)

    def uploadSourceImage(self):
        filename, image = self._uploadImage("Source Image")
        self.image = image
        if hasattr(self, "sourceImageFilename"): self.sourceImageFilename.hide()
        self.sourceImageFilename = QLabel(filename)
        self.tab1.layout.addWidget(self.sourceImageFilename, 0, 1)

    def uploadRefImage(self):
        filename, image = self._uploadImage("Reference Image")
        self.ref_image = image
        if hasattr(self, "refImageFilename"): self.refImageFilename.hide()
        self.refImageFilename = QLabel(filename)
        self.tab1.layout.addWidget(self.refImageFilename, 1, 1)

    def _uploadImage(self, img_title: str):
        # self.filenameError.hide()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        # Discards file_type since we are checking from extension
        file, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;PNG (*.png)",
            options=options
        )

        # This validation prevents the program from abortin when
        # user cancels file operation
        if file:
            img = cv2.imread(file)

            cv2.imshow(img_title, img)

        return file, img

    def runGeneralizedHoughGuil(self):
        args = dict()

        args["source_filename"] = self.sourceImageFilename.text()
        args["ref_filename"] = self.refImageFilename.text()
        args["min_dist"]  = int(self.min_dist_input.text())
        args["min_angle"] = int(self.min_angle_input.text())
        args["max_angle"] = int(self.max_angle_input.text())
        args["angle_step"] = int(self.angle_step_input.text())
        args["levels"] = int(self.levels_input.text())
        args["min_scale"] = float(self.min_scale_input.text())
        args["max_scale"] = float(self.max_scale_input.text())
        args["scale_step"] = float(self.scale_step_input.text())
        args["scale_thresh"] = int(self.scale_thresh_input.text())
        args["angle_thresh"] = int(self.angle_thresh_input.text())
        args["pos"] = int(self.position_thresh_input.text())
        args["angle_epsilon"] = int(self.angle_epsilon_input.text())
        args["xi"] = int(self.xi_input.text())

        print("Running Generalized Hough Transform")
        self.houghButton.setEnabled(False)
        self.houghButton.setDisabled(True)
        generalized_hough_guil(args)
        self.houghButton.setEnabled(True)
        self.houghButton.setDisabled(False)
        print("Finished Running")

    def runGeneralizedHoughBallard(self):
        args = dict()

        args["source_filename"] = self.sourceImageFilename.text()
        args["ref_filename"] = self.refImageFilename.text()

        args["levels"] = int(self.levels_input.text())
        args["vote_threshold"] = int(self.vote_threshold_input.text())

        print("Running Generalized Ballard Hough Transform")
        self.houghButton.setEnabled(False)
        self.houghButton.setDisabled(True)
        generalized_hough_ballard(args)
        self.houghButton.setEnabled(True)
        self.houghButton.setDisabled(False)
        print("Finished Running")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
