"""
实时分类 v3 — PCA9685 双舵机, 串口 115200
睁眼 -> '1' (竖起), 闭眼 -> '0' (放下)
"""
import time
import numpy as np
import pickle
import serial
from collections import deque
from scipy.signal import butter, filtfilt, welch, detrend
import EEGAcquisitionSDK as eeg

MODEL_PATH = r"D:\temp\EEGlab\training_data\model_v3.pkl"
ARDUINO_PORT = "COM9"
ARDUINO_BAUD = 115200
SMOOTH_WINDOW = 5


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def alpha_ratio(window, sf, alpha_low, alpha_high):
    freqs, psd = welch(window.T, sf, nperseg=min(sf, window.shape[0]), axis=-1)
    mask_alpha = (freqs >= alpha_low) & (freqs <= alpha_high)
    mask_total = (freqs >= 4) & (freqs <= 45)
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
CHANNELS = pkg["occipital_channels"]
# 现在只请求枕叶通道, o_idx 重新映射为 0..N-1
o_idx = list(range(len(CHANNELS)))
SAMPLE_RATE = pkg["sample_rate"]
WINDOW_SEC = pkg["window_sec"]
ALPHA_LOW = pkg["alpha_low"]
ALPHA_HIGH = pkg["alpha_high"]

occ_names = [CHANNELS[i] for i in o_idx]
print(f"通道: {occ_names}")
print(f"阈值: {threshold:.6f}, 窗口: {WINDOW_SEC}s")
print(f"方向: {'闭眼 Alpha 更高' if direction == 1 else '睁眼 Alpha 更高'}")

b_bp, a_bp = butter_bandpass(8, 30, SAMPLE_RATE)

print(f"连接 Arduino ({ARDUINO_PORT} @ {ARDUINO_BAUD})...")
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

        if len(buffer) >= win_samples:
            # 取一个完整非重叠窗口，清空 buffer，匹配训练
            window = np.array(buffer[:win_samples]).copy()
            buffer = buffer[win_samples:]

            for ch in range(window.shape[1]):
                window[:, ch] = detrend(window[:, ch])

            w_filt = filtfilt(b_bp, a_bp, window, axis=0)

            ratios = alpha_ratio(w_filt, SAMPLE_RATE, ALPHA_LOW, ALPHA_HIGH)
            occ_ratio = np.mean(ratios[o_idx])

            pred = 1 if (occ_ratio - threshold) * direction > 0 else 0
            predictions.append(pred)
            # 每窗都输出，确认 ratio 范围
            state_name = "闭眼" if pred == 1 else "睁眼"
            print(f"  ratio={occ_ratio:.6f} thresh={threshold:.6f} -> {state_name}  ({len(predictions)}/5)")

            if len(predictions) >= SMOOTH_WINDOW:
                smoothed = round(np.mean(predictions))
                if smoothed != current_state:
                    current_state = smoothed
                    cmd = b'1' if current_state == 0 else b'0'
                    arduino.write(cmd)
                    label = "睁眼 -> 竖起" if current_state == 0 else "闭眼 -> 放下"
                    print(f"[{time.strftime('%H:%M:%S')}] {label}  (ratio={occ_ratio:.6f})")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n退出...")
finally:
    arduino.write(b'0')
    sdk.stop_acquisition()
    arduino.close()
    print("已停止")
