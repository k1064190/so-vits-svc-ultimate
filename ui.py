import sys
from functools import partial

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSlider, QComboBox, QCheckBox,
                             QTextEdit, QGroupBox, QFileDialog, QRadioButton, QInputDialog)
from PyQt5.QtCore import Qt, QUrl, QTimer
import pylab
import pyaudio

import torch
import sounddevice as sd
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import mplcursors
import pyqtgraph as pg
from pyqtgraph import Point
from pyqtgraph.Qt import QtCore


def get_available_devices():
    devices = []

    # Check for CPU
    devices.append(("CPU", "cpu"))

    # Check for CUDA devices
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        for i in range(cuda_count):
            cuda_name = torch.cuda.get_device_name(i)
            devices.append((f"{cuda_name}", f"cuda:{i}"))

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append(("Apple MPS", "mps"))

    return devices


def get_audio_devices():
    devices = sd.query_devices()
    input_devices = []
    output_devices = []

    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((device['name'], i))
        if device['max_output_channels'] > 0:
            output_devices.append((device['name'], i))

    return input_devices, output_devices

class AudioPlayer(pg.PlotWidget):
    def __init__(self, parent=None, csv=None, audio=None):
        super().__init__(parent)
        self.scatter = None
        self.dragPoint = None
        self.dragOffset = None

        # Load data
        assert csv is not None
        self.data = pd.read_csv(csv, header=None)
        self.x_data = np.arange(0, len(self.data[0].values))  # unit = 10ms
        self.y_data = self.data[1].values

        # Load audio
        self.player = QMediaPlayer()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(audio)))

        # Player parameters
        self.player_prev_position = self.player.position()

        # Player Event
        self.player.setNotifyInterval(50)  # 10ms
        self.player.positionChanged.connect(self.update_plot)

        # Plot
        self.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0), width=1)
        self.plot(
            self.x_data,
            self.y_data,
            pen=pen,
            symbol='o',
            size=10,
            brush="b",
        )
        # self.getViewBox().scaleBy((0.2, 1))

        self.playbar = self.addLine(x=0, pen=pg.mkPen(color=(0, 0, 0), width=2))

    def plot(self, x, y, pen, **kwargs):
        self.x_data = np.array(x)
        self.y_data = np.array(y)
        self.scatter = pg.ScatterPlotItem(x=self.x_data, y=self.y_data, **kwargs)
        self.addItem(self.scatter)
        # self.scatter.sigClicked.connect(self.clicked)

        # Add a line connecting the points
        pen = 'r' if pen is None else pen
        self.line = pg.PlotDataItem(x=self.x_data, y=self.y_data, pen=pen)
        self.addItem(self.line)

        # clicking empty space in the plot will move the position to that time
        self.plotItem.scene().sigMouseClicked.connect(self.moveTime)

        self.autoRange()

    def toggle_play_pause(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.position() == self.player.duration():
                self.player.setPosition(0)
            self.player.play()

    def moveTime(self, ev):
        vb = self.plotItem.vb
        scene_coords = ev.scenePos()
        if self.sceneBoundingRect().contains(scene_coords):
            mouse_point = vb.mapSceneToView(scene_coords)
            time = int(mouse_point.x() * 10)
            time = max(0, min(time, self.player.duration()))
            self.player.setPosition(time)

    def mousePressEvent(self, ev):
        pos = self.plotItem.vb.mapSceneToView(ev.pos())
        points = self.scatter.pointsAt(pos)
        if len(points) > 0:
            self.dragPoint = points[0]
            self.dragStartPos = self.dragPoint.pos()
        elif ev.button() == QtCore.Qt.LeftButton:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.dragPoint is not None:
            if self.sceneBoundingRect().contains(ev.pos()):
                pos = self.plotItem.vb.mapSceneToView(ev.pos())
                new_y = pos.y()
                self.updatePointPos(self.dragPoint, new_y)
                self.updateLine()
            else:
                # release the point
                self.dragPoint = None
                self.dragStartPos = None
                super().mouseReleaseEvent(ev)
        else:
            super().mouseMoveEvent(ev)

    def updatePointPos(self, point, new_y):
        index = point.index()
        self.y_data[index] = new_y
        self.scatter.setData(x=self.x_data, y=self.y_data)

    def updateLine(self):
        self.line.setData(x=self.x_data, y=self.y_data)

    def mouseReleaseEvent(self, ev):
        self.dragPoint = None
        self.dragStartPos = None
        super().mouseReleaseEvent(ev)

    def update_playbar(self, time):
        self.playbar.setValue(time)

    def update_plot(self):
        position = self.player.position()
        if position != self.player_prev_position:
            self.player_prev_position = position
            self.update_playbar(position // 10)

class VoiceChangerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("so-vits-svc-fork")
        self.setGeometry(100, 100, 1600, 1200)

        self.available = True
        self.realtime = False
        self.train = False

        '''
        Required components:
        - config_path
        - model_path
        - speaker
        - wave_input
        
        The program will automatically makes the following components:
        - wave_ppg
        - wave_hubert
        - wave_pitch
        
        Flowwork:
        - Load model
        - Load speaker
        - Load waveform (audio)
        - Makes ppg.npy with whisper
        - Makes vec.npy with HuBERT
        - Makes csv with f0 estimation
            - If pitch shift is not 0, shift the pitch
            - If checkbox is checked, you can pause here and modify f0 with graph manually
        - Run the model with the loaded components

        '''

        # Arguments
        self.model_path = None
        self.config_path = None
        self.cluster_model_path = None

        # Toggle
        self.retreival = False
        self.f0_modification = False
        self.realtime = False

        self.speaker = None
        self.silence_threshold = 0
        self.pitch = 0
        self.pitch_shift = 0

        self.f0_methods = ["crepe", "rmvpe", "fcpe"]
        self.f0 = 0

        self.retrieval_ratio = 0.5
        self.n_retrieval_vectors = 3

        # e.g) self.devices = [("CPU", "cpu"), ("NVIDIA GeForce RTX 3090", "cuda:0")]
        self.input_devices, self.output_devices = get_audio_devices()
        self.input_device = 0
        self.output_device = 0
        self.devices = get_available_devices()
        self.device = 0

        self.input_audio = None
        self.output_audio = None

        # Widget groups
        self.retrieval_widgets = []
        self.f0_modification_widgets = []

        def show_arguments(self):
            print(f"Model path: {self.model_path}")
            print(f"Config path: {self.config_path}")
            print(f"Cluster model path: {self.cluster_model_path}")
            print(f"Speaker: {self.speaker}")
            print(f"Silence threshold: {self.silence_threshold}")
            print(f"Pitch: {self.pitch}")
            print(f"Input audio path: {self.input_audio}")
            print(f"Output audio path: {self.output_audio}")
            print(f"F0 method: {self.f0_methods[self.f0]}")
            print(f"Device: {self.devices[self.device][0]}")
            print(f"Input device: {self.input_devices[self.input_device][0]}")
            print(f"Output device: {self.output_devices[self.output_device][0]}")
            # print(f"Crossfade seconds: {self.crossfade_seconds}")
            # print(f"Block seconds: {self.block_seconds}")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # TODO: Implement training mode
        # train_layout = QHBoxLayout(central_widget)

        # Left column and Right column should have the same width
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)

        # Left column
        left_column = QVBoxLayout()

        # Paths group
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.addWidget(self.create_path_input("Model path", "Select Model File", "model_path"))
        paths_layout.addWidget(self.create_path_input("Config path", "Select Config File", "config_path"))
        retrieval_widget, retrieval_checkbox = self.create_check_box("Retreival")
        paths_layout.addWidget(retrieval_widget)
        cluster_model_widget = self.create_path_input("Cluster model path (Optional)", "Select Cluster Model File", "cluster_model_path")
        paths_layout.addWidget(cluster_model_widget)
        self.retrieval_widgets.append(cluster_model_widget)

        left_column.addWidget(paths_group)

        # Common group
        common_group = QGroupBox("Common")
        common_layout = QVBoxLayout(common_group)
        # TODO: speaker should be a combo box with available speakers
        common_layout.addWidget(QLabel("Speaker"))
        common_layout.addWidget(QComboBox())
        common_layout.addWidget(self.create_slider("Silence threshold", -35.0, -100.0, 0.0, 1.0, "silence_threshold"))
        common_layout.addWidget(self.create_slider("Pitch", 12, -24, 24, 1, "pitch"))

        common_layout.addWidget(self.create_radio_button("F0 method", self.f0_methods, "f0"))

        # retreival groups
        retrieval_ratio = self.create_slider("Retreival ratio", 0.5, 0.0, 1.0, 0.01, "retrieval_ratio")
        common_layout.addWidget(retrieval_ratio)
        self.retrieval_widgets.append(retrieval_ratio)
        n_retrieval_vectors = self.create_slider("Number of retrieval vectors", 3, 1, 10, 1, "n_retrieval_vectors")
        common_layout.addWidget(n_retrieval_vectors)
        self.retrieval_widgets.append(n_retrieval_vectors)

        retrieval_checkbox.stateChanged.connect(lambda: self.hide_widgets(self.retrieval_widgets, not retrieval_checkbox.isChecked()))
        self.hide_widgets(self.retrieval_widgets, True)
        retrieval_checkbox.setChecked(False)

        # List of available GPUs
        # Check if cuda is available on torch and if so, make a list of available GPUs
        if self.devices:
            common_layout.addWidget(self.create_combo_box("Device", [device[0] for device in self.devices], "device"))
        else:
            # Error message
            common_layout.addWidget(QLabel("No suitable devices found."))
            self.available = False

        realtime_widget, realtime_checkbox = self.create_check_box("Realtime")
        common_layout.addWidget(realtime_widget)

        left_column.addWidget(common_group)

        main_layout.addLayout(left_column)

        # Right column
        right_column = QVBoxLayout()

        # File group
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.addWidget(self.create_path_input("Input audio path", "Select Input Audio File", "input_audio"))
        file_layout.addWidget(self.create_path_input("Output audio path", "Select Output Audio File", "output_audio"))

        self.graph = AudioPlayer(self, "test.csv", "test.wav")
        file_layout.addWidget(self.graph)
        play_button = QPushButton("Play/Pause")
        file_layout.addWidget(play_button)
        play_button.clicked.connect(self.graph.toggle_play_pause)

        # self.timer = QTimer()
        # self.timer.timeout.connect(self.update_plot)
        # self.timer.start(10)  # Update every 10ms

        right_column.addWidget(file_group)

        # Realtime group
        realtime_group = QGroupBox("Realtime")
        realtime_layout = QVBoxLayout(realtime_group)
        realtime_layout.addWidget(self.create_slider("Crossfade seconds", 0.05))
        realtime_layout.addWidget(self.create_slider("Block seconds", 0.35))

        realtime_layout.addWidget(self.create_combo_box("Input device", [device[0] for device in self.input_devices], "input_device"))
        realtime_layout.addWidget(self.create_combo_box("Output device", [device[0] for device in self.output_devices], "output_device"))
        right_column.addWidget(realtime_group)

        # Run arguments group
        run_arguments_group = QGroupBox("Run Arguments")
        run_arguments_layout = QHBoxLayout(run_arguments_group)
        # Horizontal Checkbox
        f0_modification_widget, f0_modification_checkbox = self.create_check_box("F0 Modification")
        run_arguments_layout.addWidget(f0_modification_widget)
        button = self.create_button("Run", lambda: show_arguments(self))
        run_arguments_layout.addWidget(button)
        right_column.addWidget(run_arguments_group)


        def realtime_checkbox_state_changed():
            self.realtime = realtime_checkbox.isChecked()
            if self.realtime:
                # hide file group and show realtime group
                file_group.hide()
                realtime_group.show()
            else:
                # hide realtime group and show file group
                realtime_group.hide()
                file_group.show()

        realtime_checkbox.stateChanged.connect(realtime_checkbox_state_changed)
        realtime_checkbox_state_changed()

        main_layout.addLayout(right_column)

    def create_button(self, label, callback):
        button = QPushButton(label)
        button.clicked.connect(callback)
        return button

    def create_path_input(self, label, dialog_title, var_name=None):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        line_edit = QLineEdit()
        layout.addWidget(line_edit)
        browse_button = QPushButton("Browse")
        layout.addWidget(browse_button)

        def open_file_dialog():
            file_path, _ = QFileDialog.getOpenFileName(self, dialog_title, "", "All Files (*)")
            if file_path:
                line_edit.setText(file_path)

        browse_button.clicked.connect(open_file_dialog)
        if var_name is not None:
            line_edit.textChanged.connect(lambda: setattr(self, var_name, line_edit.text()))

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def create_combo_box(self, label, items, var_name=None):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        combo_box = QComboBox()
        combo_box.addItems(items)
        if var_name is not None:
            combo_box.currentIndexChanged.connect(lambda: setattr(self, var_name, combo_box.currentIndex()))
            setattr(self, var_name, combo_box.currentIndex())
        layout.addWidget(combo_box)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def create_check_box(self, label, var_name=None):
        layout = QVBoxLayout()
        check_box = QCheckBox(label)
        if var_name is not None:
            check_box.stateChanged.connect(lambda: setattr(self, var_name, check_box.isChecked()))
            setattr(self, var_name, check_box.isChecked())
        layout.addWidget(check_box)
        widget = QWidget()
        widget.setLayout(layout)
        return widget, check_box

    def create_radio_button(self, label, items, var_name=None):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        for idx, item in enumerate(items):
            radio_button = QRadioButton(item)
            if var_name is not None:
                if getattr(self, var_name) == idx:
                    radio_button.setChecked(True)
                radio_button.toggled.connect(partial(lambda idx, var_name, checked: setattr(self, var_name, idx) if checked else None, idx, var_name))
            layout.addWidget(radio_button)

        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def create_slider(self, label, value, min=0.0, max=100.0, step=0.5, var_name=None):
        min = int(min * 10)
        max = int(max * 10)
        value = int(value * 10)
        step = int(step * 10)
        layout = QVBoxLayout()
        qlabel = QLabel(f"{label}: {value / 10}")
        layout.addWidget(qlabel)
        slider = QSlider(Qt.Horizontal)
        slider.setMaximum(max)
        slider.setMinimum(min)
        slider.setSingleStep(step)
        slider.setPageStep(step*2)
        slider.setValue(int(value))  # Assuming slider range 0-100

        def update():
            value = slider.value()
            qlabel.setText(f"{label}: {value / 10}")
            if var_name is not None:
                setattr(self, var_name, value / 10)

        slider.valueChanged.connect(update)
        update()

        layout.addWidget(slider)
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def hide_widgets(self, widgets, hide=True):
        for widget in widgets:
            if hide:
                widget.hide()
            else:
                widget.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceChangerGUI()
    window.show()
    sys.exit(app.exec_())
