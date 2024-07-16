import json

import torch
import sounddevice as sd
from PyQt5.QtWidgets import QDesktopWidget


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

def calculateWindowSize():
    # 현재 스크린의 가용 영역 얻기
    screen = QDesktopWidget().availableGeometry()

    # set 2/3 of the screen size as the default size
    target_width = int(screen.width() * 2 / 3)
    target_height = int(screen.height() * 2 / 3)

    # min size (y: 800, x: 1200)
    width = max(target_width, 1200)
    height = max(target_height, 800)

    # Locate the window in the center of the screen
    left = int((screen.width() - width) / 2)
    top = int((screen.height() - height) / 2)

    return (left, top, width, height)

def save_json_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
