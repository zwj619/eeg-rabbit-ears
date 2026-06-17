"""
实时 Alpha 阈值分类 + Arduino 兔耳朵
Alpha高(闭眼) → 放下('D'), Alpha低(睁眼) → 竖起('U')
"""
import time
import numpy as np
import pickle
import serial
from collections import deque
from scipy.signal import welch
import EEGAcquisitionSDK as eeg

MODEL_PATH = r"D:\temp\EEGlab\training_data\model.pkl"
ARDUINO_PORT = "COM9"
ARDUINO_BAUD = 115200
SMOOTH_WINDOW = 5

print("加载模型...")
with open(MODEL_PATH, "rb") as f:
    pkg = pickle.load(f)
threshold = pkg["threshold"]
direction = pkg["direction"]
o_idx = pkg["occipital_idx"]
CHANNELS = pkg["channels"]
SAMPLE_RATE = pkg["sample_rate"]
WINDOW_SEC = pkg["window_sec"]
ALPHA_LOW = pkg["alpha_low"]
ALPHA_HIGH = pkg["alpha_high"]

occ_names = [CHANNELS[i] for i in o_idx]
print(f"枕叶通道: {occ_names}, 阈值: {threshold:.2f}")
if direction == 1:
    print("规则: Alpha > 阈值 → 闭眼 → 放下")
else:
    print("规则: Alpha > 阈值 → 睁眼 → 竖起")

print(f"连接 Arduino ({ARDUINO_PORT})...")
arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

print("连接脑电...")
sdk = eeg.EEGAcquisitionSDK(port="COM10", filtered_channels=CHANNELS)
sdk.start_acquisition()

predictions = deque(maxlen=SMOOTH_WINDOW)
current_state = None
win_samples = int(WINDOW_SEC * SAMPLE_RATE)
buffer = []

print("\n实时分类运行中... 试着睁眼/闭眼！(Ctrl+C 退出)\n")

try:
    while True:
        data = sdk.pop_data(n_samples=62, is_blocking_mode=False)
        if data is None or data.shape[0] == 0:
            time.sleep(0.02)
            continue

        eeg_ch = data[:, :-1]
        buffer.extend(eeg_ch.tolist())
        while len(buffer) > win_samples:
            buffer.pop(0)

        if len(buffer) >= win_samples:
            window = np.array(buffer[-win_samples:])

            # 计算枕叶 Alpha 功率
            freqs, psd = welch(window.T, SAMPLE_RATE,
                               nperseg=min(SAMPLE_RATE, window.shape[0]), axis=-1)
            mask = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
            occ_alpha = np.mean(np.sum(psd[o_idx][:, mask], axis=-1))

            # 分类
            pred = 1 if (occ_alpha - threshold) * direction > 0 else 0
            predictions.append(pred)

            if len(predictions) >= SMOOTH_WINDOW:
                smoothed = round(np.mean(predictions))
                if smoothed != current_state:
                    current_state = smoothed
                    cmd = '0' if current_state == 0 else '1'
                    arduino.write(cmd.encode())
                    label = "睁眼→放下" if current_state == 0 else "闭眼→竖起"
                    print(f"[{time.strftime('%H:%M:%S')}] {label}  (Alpha={occ_alpha:.2f})")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n退出...")
finally:
    arduino.write(b'0')
    sdk.stop_acquisition()
    arduino.close()
    print("已停止")
