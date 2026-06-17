"""
实时分类 v4 — 动态自适应基线, 无需校准阶段
"""
import time
import numpy as np
import serial
from collections import deque
from scipy.signal import butter, filtfilt, welch, detrend
import EEGAcquisitionSDK as eeg

ARDUINO_PORT = "COM9"
ARDUINO_BAUD = 115200
SAMPLE_RATE = 250
WINDOW_SEC = 2.0
SMOOTH_WINDOW = 5
THRESHOLD_MULT = 2.0
ALPHA_LOW, ALPHA_HIGH = 8, 13
EMA_ALPHA = 0.10             # 基线更新速度
BASELINE_HISTORY = 15         # 用最近N个睁眼值算基线

POSTERIOR = ["T5", "TP7", "O1", "P3", "PZ", "OZ", "P4", "O2"]


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


print("连接脑电...")
sdk = eeg.EEGAcquisitionSDK(port="COM10", filtered_channels=POSTERIOR)
sdk.start_acquisition()

b_bp, a_bp = butter_bandpass(8, 13, SAMPLE_RATE)
win_samples = int(WINDOW_SEC * SAMPLE_RATE)
buffer = []

print(f"连接 Arduino ({ARDUINO_PORT} @ {ARDUINO_BAUD})...")
arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

predictions = deque(maxlen=SMOOTH_WINDOW)
current_state = None
baseline_history = deque(maxlen=BASELINE_HISTORY)
baseline = None  # 动态学习

print(f"\n{'='*50}")
print("  动态自适应模式")
print("  前30秒学习基线, 然后自动检测闭眼")
print(f"{'='*50}\n")

try:
    while True:
        data = sdk.pop_data(n_samples=62, is_blocking_mode=False)
        if data is None or data.shape[0] == 0:
            time.sleep(0.02)
            continue

        eeg_ch = data[:, :-1]
        buffer.extend(eeg_ch.tolist())

        if len(buffer) >= win_samples:
            window = np.array(buffer[:win_samples]).copy()
            buffer = buffer[win_samples:]

            for ch in range(window.shape[1]):
                window[:, ch] = detrend(window[:, ch])

            w_filt = filtfilt(b_bp, a_bp, window, axis=0)

            freqs, psd = welch(w_filt.T, SAMPLE_RATE, nperseg=250, axis=-1)
            mask = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
            alpha_pow = psd[:, mask].sum(axis=-1)
            alpha = np.mean(alpha_pow)

            # ---- 动态基线 ----
            if baseline is None:
                # 初始化: 累积前几个窗口
                baseline_history.append(alpha)
                if len(baseline_history) >= 5:
                    baseline = np.median(baseline_history)
            else:
                # 如果当前判断为睁眼, 更新基线
                if alpha < baseline * THRESHOLD_MULT * 0.8:
                    baseline_history.append(alpha)
                    baseline = np.median(baseline_history)

            if baseline is None:
                print(f"  alpha={alpha:.1f}  (学习基线... {len(baseline_history)}/5)")
                continue

            threshold = baseline * THRESHOLD_MULT
            pred = 1 if alpha > threshold else 0
            predictions.append(pred)

            state_name = "闭眼↑↑" if pred == 1 else "睁眼  "
            bar_len = int(min(alpha / (baseline * 4), 1) * 30)
            bar = "█" * bar_len
            print(f"  α={alpha:.1f} 基线={baseline:.1f} 阈值={threshold:.1f} |{bar:<30}| {state_name}")

            if len(predictions) >= SMOOTH_WINDOW:
                smoothed = round(np.mean(predictions))
                if smoothed != current_state:
                    current_state = smoothed
                    cmd = b'1' if current_state == 1 else b'0'
                    arduino.write(cmd)
                    label = "闭眼(高α) -> 竖起" if current_state == 1 else "睁眼(低α) -> 放下"
                    print(f"\n>>> [{time.strftime('%H:%M:%S')}] {label} <<<\n")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n退出...")
finally:
    arduino.write(b'0')
    sdk.stop_acquisition()
    arduino.close()
    print("已停止")
