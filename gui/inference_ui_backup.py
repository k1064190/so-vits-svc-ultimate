import json
import os
from functools import partial
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QLabel, QComboBox, QPushButton, \
    QLineEdit, QSlider, QRadioButton, QCheckBox, QFileDialog

from audio_player import AudioPlayer
from ui_utils import get_audio_devices, get_available_devices, load_json_file, save_json_file
from widgets_visibility_manager import WidgetVisibilityManager


class InferenceTab(QWidget):
    def __init__(self, parent=None):
        super(InferenceTab, self).__init__(parent)
        self.f0_modes = ["crepe", "rmvpe", "fcpe", "none"]
        self.speech_encoder_modes = ["contentvec", "whisper_pgg", "hubert_soft", "hubert_discrete"]
        self.svc_modes = ["so-vits"]
        self.post_processing_modes = ["shallow_diffusion"]
        # e.g) self.devices = [("CPU", "cpu"), ("NVIDIA GeForce RTX 3090", "cuda:0")]
        self.input_devices, self.output_devices = get_audio_devices()
        self.devices = get_available_devices()
        self.preset_folder = "presets"
        self.presets = load_json_file(self.preset_folder)
        # Arguments that the preset saves
        self.common_arguments = {
            "model_path": None,
            "config_path": None,
            "cluster_model_path": None,
            "speaker": None,
            "silence_threshold": 0,
            "pitch_shift": 0,
            "noise_scale": 0,
            "pad_seconds": 0,
            "chunk_seconds": 0,
            "max_chunk_seconds": 0,
            "absolute_thresh": False,
            "f0_modification": False,
            "auto_predict_f0": False,
            "f0": 0,
            "speech_encoder": [False] * len(self.speech_encoder_modes),
            "svc": 0,
            "post_processing": [False] * len(self.post_processing_modes),
        }
        self.realtime_arguments = {
            "realtime": False,
            "crossfade_seconds": 0,
            "block_seconds": 0,
            "additional_infer_before_seconds": 0,
            "additional_infer_after_seconds": 0,
        }
        self.retrieval_arguments = {
            "retrieval": False,
            "retrieval_ratio": 0.5,
            "n_retrieval_vectors": 3,
        }

        # arguments that the preset doesn't save
        self.run_arguments = {
            "f0_modification": False,
        }
        self.file_arguments = {
            "input_audio": None,
            "output_audio": None,
        }
        self.device_arguments = {
            "device": 0,
            "input_device": 0,
            "output_device": 0,
        }

        # Widget groups
        self.retrieval_widgets = []
        self.f0_modification_widgets = []
        self.created_widgets = []

        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # Left column and Right column should have the same width
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)

        # Left column
        left_column = QVBoxLayout()

        # Paths group
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout(paths_group)
        paths_layout.addWidget(self.create_path_input(
            "Model path", "Select Model File",
            arguments_dict=self.common_arguments, var_name="model_path"))
        paths_layout.addWidget(self.create_path_input(
            "Config path", "Select Config File",
            arguments_dict=self.common_arguments, var_name="config_path"))
        cluster_model_widget = (
            self.create_path_input("Cluster model path (Optional)", "Select Cluster Model File",
                                   arguments_dict=self.common_arguments, var_name="cluster_model_path"))
        paths_layout.addWidget(cluster_model_widget)

        left_column.addWidget(paths_group)

        # Common group is scrollable
        common_scroll_area = QScrollArea()
        common_scroll_area.setWidgetResizable(True)
        common_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        common_group = QGroupBox("Common")
        common_layout = QVBoxLayout(common_group)
        # TODO: speaker should be a combo box with available speakers
        common_layout.addWidget(QLabel("Speaker"))
        common_layout.addWidget(QComboBox())
        common_layout.addWidget(
            self.create_slider("Silence threshold", -35.0, -100.0, 0.0, 1.0,
                               arguments_dict=self.common_arguments, var_name="silence_threshold"))
        common_layout.addWidget(
            self.create_slider("Pitch", 12, -24, 36, 1,
                               arguments_dict=self.common_arguments, var_name="pitch_shift"))
        common_layout.addWidget(
            self.create_slider("Noise scale", 0.40, 0.0, 1.0, 0.01,
                               arguments_dict=self.common_arguments, var_name="noise_scale"))
        common_layout.addWidget(
            self.create_slider("Pad seconds", 0.10, 0.0, 1.0, 0.01,
                               arguments_dict=self.common_arguments, var_name="pad_seconds"))
        common_layout.addWidget(
            self.create_slider("Chunk seconds", 0.50, 0.0, 1.0, 0.01,
                               arguments_dict=self.common_arguments, var_name="chunk_seconds"))
        common_layout.addWidget(
            self.create_slider("Max chunk seconds", 40.0, 0.0, 200.0, 1,
                               arguments_dict=self.common_arguments, var_name="max_chunk_seconds"))
        common_layout.addWidget(
            self.create_check_box("Absolute threshold", arguments_dict=self.common_arguments,
                                  var_name="absolute_thresh"))

        common_layout.addWidget(self.create_radio_button("F0 method", self.f0_modes,
                                                         arguments_dict=self.common_arguments, var_name="f0"))

        # retreival groups
        retrieval_widget = self.create_check_box("Retreival",
                                                 arguments_dict=self.retrieval_arguments, var_name="retrieval")
        retrieval_checkbox = retrieval_widget.layout().itemAt(0).widget()
        common_layout.addWidget(retrieval_widget)
        retrieval_ratio = (
            self.create_slider("Retreival ratio", 0.5, 0.0, 1.0, 0.01,
                               arguments_dict=self.retrieval_arguments, var_name="retrieval_ratio"))
        common_layout.addWidget(retrieval_ratio)
        self.retrieval_widgets.append(retrieval_ratio)
        n_retrieval_vectors = (
            self.create_slider("Number of retrieval vectors", 3, 1, 10, 1,
                               arguments_dict=self.retrieval_arguments, var_name="n_retrieval_vectors"))
        common_layout.addWidget(n_retrieval_vectors)
        self.retrieval_widgets.append(n_retrieval_vectors)

        WidgetVisibilityManager.toggle_widgets_visibility(self.retrieval_widgets, True)
        retrieval_checkbox.stateChanged.connect(
            lambda state: WidgetVisibilityManager.toggle_widgets_visibility(
                self.retrieval_widgets, not state))
        retrieval_checkbox.setChecked(False)

        # List of available GPUs
        # Check if cuda is available on torch and if so, make a list of available GPUs
        if self.devices:
            common_layout.addWidget(
                self.create_combo_box("Device", [device[0] for device in self.devices],
                                      arguments_dict=self.device_arguments, var_name="device"))
        else:
            # Error message
            common_layout.addWidget(QLabel("No suitable devices found."))
            self.available = False

        realtime_widget = (
            self.create_check_box("Realtime", arguments_dict=self.realtime_arguments, var_name="realtime"))
        realtime_checkbox = realtime_widget.layout().itemAt(0).widget()
        common_layout.addWidget(realtime_widget)

        common_scroll_area.setWidget(common_group)
        left_column.addWidget(common_scroll_area, stretch=1)

        main_layout.addLayout(left_column)

        # Right column
        right_column = QVBoxLayout()

        # File group
        file_column = []
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.addWidget(self.create_path_input("Input audio path", "Select Input Audio File",
                                                     arguments_dict=self.file_arguments, var_name="input_audio"))
        file_layout.addWidget(self.create_path_input("Output audio path", "Select Output Audio File",
                                                     arguments_dict=self.file_arguments, var_name="output_audio"))

        self.graph = AudioPlayer()
        file_layout.addWidget(self.graph)
        play_button = QPushButton("Play/Pause")
        file_layout.addWidget(play_button)
        play_button.clicked.connect(self.graph.toggle_play_pause)
        right_column.addWidget(file_group, stretch=1)
        file_column.append(file_group)

        # Run arguments group
        run_arguments_group = QGroupBox("Run Arguments")
        run_arguments_layout = QHBoxLayout(run_arguments_group)
        # Horizontal Checkbox
        f0_modification_widget = self.create_check_box("F0 Modification")
        f0_modification_checkbox = f0_modification_widget.layout().itemAt(0).widget()
        run_arguments_layout.addWidget(f0_modification_widget)
        button = self.create_button("Run", lambda: {})
        run_arguments_layout.addWidget(button)
        right_column.addWidget(run_arguments_group)
        file_column.append(run_arguments_group)

        # Realtime group
        realtime_column = []
        realtime_group = QGroupBox("Realtime")
        realtime_layout = QVBoxLayout(realtime_group)
        realtime_layout.addWidget(self.create_slider("Crossfade seconds", 0.05,
                                                     arguments_dict=self.realtime_arguments,
                                                     var_name="crossfade_seconds"))
        realtime_layout.addWidget(self.create_slider("Block seconds", 0.35,
                                                     arguments_dict=self.realtime_arguments, var_name="block_seconds"))

        realtime_layout.addWidget(
            self.create_combo_box("Input device", [device[0] for device in self.input_devices],
                                  arguments_dict=self.device_arguments, var_name="input_device"))
        realtime_layout.addWidget(
            self.create_combo_box("Output device", [device[0] for device in self.output_devices],
                                  arguments_dict=self.device_arguments, var_name="output_device"))
        right_column.addWidget(realtime_group, stretch=1)
        realtime_column.append(realtime_group)

        # Record arguments
        record_group = QGroupBox("Record")
        record_layout = QHBoxLayout(record_group)
        record_button = self.create_button("Record", lambda: {})
        record_layout.addWidget(record_button)
        right_column.addWidget(record_group)
        realtime_column.append(record_group)

        def realtime_checkbox_state_changed():
            self.realtime_arguments['realtime'] = realtime_checkbox.isChecked()
            if self.realtime_arguments['realtime']:
                # hide file group and show realtime group
                WidgetVisibilityManager.swap_widgets_visibility(file_column, realtime_column)
            else:
                # hide realtime group and show file group
                WidgetVisibilityManager.swap_widgets_visibility(realtime_column, file_column)

        realtime_checkbox.stateChanged.connect(realtime_checkbox_state_changed)
        realtime_checkbox_state_changed()

        # presets
        preset_group = QGroupBox("Presets")

        preset_layout = QVBoxLayout(preset_group)
        preset_select_delete = QHBoxLayout()
        preset_select = self.create_combo_box("Select preset", list(self.presets.keys()))
        preset_combo_box = preset_select.layout().itemAt(1).widget()
        preset_combo_box.currentIndexChanged.connect(lambda: self.load_gui_preset(preset_combo_box.currentText()))
        preset_select_delete.addWidget(preset_select, stretch=4)
        preset_delete = QPushButton("Delete")
        preset_delete.clicked.connect(lambda: self.delete_current_preset(preset_combo_box))
        preset_select_delete.addWidget(preset_delete, stretch=1)

        preset_name_add = QHBoxLayout()
        preset_name_add.addWidget(QLabel("Preset name"))
        self.preset_name = QLineEdit()
        self.preset_name.setPlaceholderText("Preset name")
        preset_name_add.addWidget(self.preset_name)
        preset_button = QPushButton("Save")
        preset_button.clicked.connect(lambda: self.save_gui_preset(self.preset_name.text(), preset_combo_box))
        preset_name_add.addWidget(preset_button)

        preset_layout.addLayout(preset_select_delete)
        preset_layout.addLayout(preset_name_add)
        right_column.addWidget(preset_group, stretch=0)

        main_layout.addLayout(right_column)

    def create_button(self, label, callback):
        button = QPushButton(label)
        button.clicked.connect(callback)
        return button

    def create_path_input(self, label, dialog_title, arguments_dict = None, var_name=None):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        line_edit = QLineEdit()
        layout.addWidget(line_edit)
        browse_button = QPushButton("Browse")
        layout.addWidget(browse_button)
        widget = QWidget()

        def open_file_dialog():
            file_path, _ = QFileDialog.getOpenFileName(self, dialog_title, "", "All Files (*)")
            if file_path:
                line_edit.setText(file_path)

        browse_button.clicked.connect(open_file_dialog)
        if arguments_dict is not None and var_name is not None:
            line_edit.textChanged.connect(lambda: arguments_dict.__setitem__(var_name, line_edit.text()))
            def update_line_edit():
                text = arguments_dict[var_name]
                if text is not None:
                    line_edit.setText(text)
            setattr(widget, "renew", update_line_edit)

        widget.setLayout(layout)

        self.created_widgets.append(widget)

        return widget

    def create_combo_box(self, label, items, arguments_dict=None, var_name=None):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label))
        combo_box = QComboBox()
        combo_box.addItems(items)
        if arguments_dict is not None and var_name is not None:
            combo_box.currentIndexChanged.connect(lambda: arguments_dict.__setitem__(var_name, combo_box.currentIndex()))
            arguments_dict.__setitem__(var_name, combo_box.currentIndex())
            def update_combo_box():
                combo_idx = arguments_dict[var_name]
                if combo_idx is not None:
                    combo_box.setCurrentIndex(combo_idx)
            setattr(widget, "renew", update_combo_box)

        layout.addWidget(combo_box)
        widget.setLayout(layout)

        self.created_widgets.append(widget)

        return widget   # widget > layout > label, combo_box

    def clear_combo_box(self, combo_box, items, name_or_idx: Optional[str or int]):
        combo_box.clear()
        combo_box.addItems(items)
        if name_or_idx is not None:
            if isinstance(name_or_idx, int):
                combo_box.setCurrentIndex(name_or_idx)
            else:
                combo_box.setCurrentIndex(combo_box.findText(name_or_idx))

    def create_check_box(self, label, arguments_dict=None, var_name=None):
        widget = QWidget()
        layout = QVBoxLayout()
        check_box = QCheckBox(label)
        if arguments_dict is not None and var_name is not None:
            check_box.stateChanged.connect(lambda: arguments_dict.__setitem__(var_name, check_box.isChecked()))
            arguments_dict.__setitem__(var_name, check_box.isChecked())
            def update_check_box():
                checked = arguments_dict[var_name]
                if checked is not None:
                    check_box.setChecked(checked)
            setattr(widget, "renew", update_check_box)
        layout.addWidget(check_box)
        widget.setLayout(layout)

        self.created_widgets.append(widget)

        # widget > layout > check_box
        return widget

    def create_radio_button(self, label, items, arguments_dict=None, var_name=None):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        for idx, item in enumerate(items):
            radio_button = QRadioButton(item)
            if arguments_dict is not None and var_name is not None:
                if arguments_dict[var_name] == idx:
                    radio_button.setChecked(True)
                radio_button.toggled.connect(partial(lambda idx, var_name, checked: arguments_dict.__setitem__(var_name, idx) if checked else None, idx, var_name))
            layout.addWidget(radio_button)

        widget.setLayout(layout)

        if arguments_dict is not None and var_name is not None:
            def update_radio_buttons():
                idx = arguments_dict[var_name]
                if idx is not None:
                    layout.itemAt(idx + 1).widget().setChecked(True)
            setattr(widget, "renew", update_radio_buttons)

        self.created_widgets.append(widget)

        return widget

    def create_slider(self, label, value, min=0.0, max=100.0, step=0.5, arguments_dict=None, var_name=None):
        widget = QWidget()
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
            if arguments_dict is not None and var_name is not None:
                arguments_dict[var_name] = value / 10

                def update_slider():
                    slider_value = arguments_dict[var_name]
                    if slider_value is not None:
                        slider.setValue(int(slider_value * 10))
                        qlabel.setText(f"{label}: {slider_value}")

                setattr(widget, "renew", update_slider)

        slider.valueChanged.connect(update)
        update()

        layout.addWidget(slider)
        widget.setLayout(layout)

        self.created_widgets.append(widget)
        return widget

    def load_gui_preset(self, preset_name):
        # Preset is dictionary with keys as the variable names
        preset = self.presets[preset_name]
        # common, realtime, retrieval arguments
        for key, value in preset["common"].items():
            self.common_arguments[key] = value
        for key, value in preset["realtime"].items():
            self.realtime_arguments[key] = value
        for key, value in preset["retrieval"].items():
            self.retrieval_arguments[key] = value

        # We should update sliders, checkboxes, etc. with the new values
        for widget in self.created_widgets:
            if hasattr(widget, "renew"):
                widget.renew()
        self.preset_name.setText(preset_name)


    def save_gui_preset(self, preset_name, combo_box=None):
        if not preset_name:
            return
        preset = {}
        common = {}
        realtime = {}
        retrieval = {}
        # common, realtime, file, run_arguments, retrieval
        for key, value in self.common_arguments.items():
            common[key] = value
        for key, value in self.realtime_arguments.items():
            realtime[key] = value
        for key, value in self.retrieval_arguments.items():
            retrieval[key] = value
        preset["common"] = common
        preset["realtime"] = realtime
        preset["retrieval"] = retrieval
        self.presets[preset_name] = preset

        save_json_file(self.preset_folder, preset_name, preset)

        # combo_box_widget > layout > label, combo_box
        if combo_box is not None:
            self.clear_combo_box(combo_box, list(self.presets.keys()), preset_name)

    def delete_current_preset(self, combo_box):
        preset_name = combo_box.currentText()
        idx = combo_box.currentIndex()

        del self.presets[preset_name]
        # Delte the file, if exists
        os.remove(f"presets/{preset_name}.json")

        # Set the combo box to the previous index
        self.clear_combo_box(combo_box, list(self.presets.keys()), idx-1)

    def closeEvent(self, ev):
        ev.accept()
