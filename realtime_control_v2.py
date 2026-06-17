"""
实时分类 v2 — 滤波管线 + Arduino 控制
"""
import time
import numpy as np
import pickle
import serial
from collections import deque
from scipy.signal import butter, filtfilt, welch
import EEGAcquisitionSDK as eeg

MODEL_PATH = r"D:\temp\EEGlab\training_data\model_v2.pkl"
ARDUINO_PORT = "COM9"
ARDUINO_BAUD = 9600
SMOOTH_WINDOW = 5


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def butter_bandstop(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandstop")
    return b, a


def alpha_ratio(window, sf, alpha_low, alpha_high, hp_cut):
    freqs, psd = welch(window.T, sf, nperseg=min(sf, window.shape[0]), axis=-1)
    mask_alpha = (freqs >= alpha_low) & (freqs <= alpha_high)
    mask_total = (freqs >= hp_cut) & (freqs <= 45)
    alpha_pow = psd[:, mask_alpha].sum(axis=-1)
    total_pow = psd[:, mask_total].sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_pow > 0, alpha_pow / total_pow, 0)
    return ratio


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
HP_CUT = pkg.get("hp_cut", 4)

occ_names = [CHANNELS[i] for i in o_idx]
print(f"通道: {occ_names}")
print(f"阈值: {threshold:.6f}, HP={HP_CUT}Hz")
print(f"方向: {'闭眼 Alpha 更高' if direction == 1 else '睁眼 Alpha 更高'}")

# 预设计滤波器
b_hp, a_hp = butter_bandpass(HP_CUT, 45, SAMPLE_RATE)
b_notch, a_notch = butter_bandstop(48, 52, SAMPLE_RATE)

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

print("\n实时分类运行中 (Ctrl+C 退出)\n")

try:
    while True:
        data = sdk.pop_data(n_samples=62, is_blocking_mode=False)
        if data is None or data.shape[0] == 0:
            time.sleep(0.02)
            continue

        eeg_ch = data[:, :-1]
        buffer.extend(eeg_ch.tolist())
        while len(buffer) > win_samples * 2:
            buffer = buffer[-win_samples * 2:]

        if len(buffer) >= win_samples:
            window = np.array(buffer[-win_samples:])

            # 滤波
            w_filt = filtfilt(b_hp, a_hp, window, axis=0)
            w_filt = filtfilt(b_notch, a_notch, w_filt, axis=0)

            # Alpha 占比
            ratios = alpha_ratio(w_filt, SAMPLE_RATE, ALPHA_LOW, ALPHA_HIGH, HP_CUT)
            occ_ratio = np.mean(ratios[o_idx])

            # 分类
            pred = 1 if (occ_ratio - threshold) * direction > 0 else 0
            predictions.append(pred)

            if len(predictions) >= SMOOTH_WINDOW:
                smoothed = round(np.mean(predictions))
                if smoothed != current_state:
                    current_state = smoothed
                    cmd = "U" if current_state == 0 else "D"
                    arduino.write(cmd.encode())
                    label = "睁眼 -> 竖起" if current_state == 0 else "闭眼 -> 放下"
                    print(f"[{time.strftime('%H:%M:%S')}] {label}  (ratio={occ_ratio:.6f})")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n退出...")
finally:
    arduino.write(b"D")
    sdk.stop_acquisition()
    arduino.close()
    print("已停止")
