"""
实时分类 v5 — 总信号功率法 (RMS)，不依赖 Alpha 频段
高阻抗环境下 Alpha 不可靠，改用枕叶信号总能量差异
睁眼: 微眼动+眨眼 → 信号方差大
闭眼: 更安静 → 信号方差小
"""
import time
import numpy as np
import serial
from collections import deque
from scipy.signal import butter, filtfilt, detrend
import EEGAcquisitionSDK as eeg

ARDUINO_PORT = "COM9"
ARDUINO_BAUD = 115200
SAMPLE_RATE = 250
WINDOW_SEC = 2.0
SMOOTH_WINDOW = 3
THRESHOLD_MULT = 1.3       # 当前RMS > 基线*倍数 → 睁眼
EMA_ALPHA = 0.15           # 基线更新速度 (0.15=较快)
BASELINE_HISTORY = 15

# 枕叶+后部通道
POSTERIOR = ["T5", "TP7", "O1", "P3", "PZ", "OZ", "P4", "O2"]


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


print("连接脑电...")
sdk = eeg.EEGAcquisitionSDK(port="COM10", filtered_channels=POSTERIOR)
sdk.start_acquisition()

# 1-30 Hz 带通，保留眼动伪迹所在的频段
b_bp, a_bp = butter_bandpass(1, 30, SAMPLE_RATE, order=3)
win_samples = int(WINDOW_SEC * SAMPLE_RATE)
buffer = []

print(f"连接 Arduino ({ARDUINO_PORT} @ {ARDUINO_BAUD})...")
arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
time.sleep(2)
arduino.reset_input_buffer()

predictions = deque(maxlen=SMOOTH_WINDOW)
baseline_rms_history = deque(maxlen=BASELINE_HISTORY)
baseline_rms = None
current_state = None  # None=未知, 0=睁眼, 1=闭眼

print("=" * 55)
print("  v5 总功率模式 | 自适应基线")
print("  前10秒学习安静基线, 请睁眼保持不动")
print("=" * 55)

try:
    while True:
        data = sdk.pop_data(n_samples=62, is_blocking_mode=False)
        if data is None or data.shape[0] == 0:
            time.sleep(0.02)
            continue

        eeg_ch = data[:, :-1].copy()
        buffer.extend(eeg_ch.tolist())

        if len(buffer) >= win_samples:
            window = np.array(buffer[:win_samples]).copy()
            buffer = buffer[win_samples:]

            # 去趋势 + 带通滤波
            for ch in range(window.shape[1]):
                window[:, ch] = detrend(window[:, ch])
            w_filt = filtfilt(b_bp, a_bp, window, axis=0)

            # 计算枕叶通道RMS (均方根 = 总功率的平方根)
            ch_rms = np.sqrt(np.mean(w_filt ** 2, axis=0))
            total_rms = np.mean(ch_rms)       # 所有枕叶通道平均RMS

            # --- 动态基线 ---
            if baseline_rms is None:
                baseline_rms_history.append(total_rms)
                if len(baseline_rms_history) >= 5:
                    baseline_rms = np.median(baseline_rms_history)
                    print(f"> 基线建立: {baseline_rms:.3f}")
                else:
                    print(f"  学习基线... {len(baseline_rms_history)}/5  RMS={total_rms:.3f}")
                    time.sleep(0.02)
                    continue

            # 指数移动平均更新基线 (只在闭眼时更新)
            if total_rms < baseline_rms * THRESHOLD_MULT * 0.9:
                baseline_rms = EMA_ALPHA * total_rms + (1 - EMA_ALPHA) * baseline_rms

            # 分类: RMS高 → 睁眼(0), RMS低 → 闭眼(1)
            threshold = baseline_rms * THRESHOLD_MULT
            pred = 0 if total_rms > threshold else 1   # 注意: 高RMS=睁眼
            predictions.append(pred)

            state_name = "睁眼(高RMS)" if pred == 0 else "闭眼(低RMS)"
            bar = "#" * int(min(total_rms / (baseline_rms * 3), 1) * 30)
            print(f"  RMS={total_rms:.3f} base={baseline_rms:.3f} thr={threshold:.3f} |{bar:<30}| {state_name}")

            if len(predictions) >= SMOOTH_WINDOW:
                smoothed = round(np.mean(predictions))
                if smoothed != current_state:
                    current_state = smoothed
                    cmd = b'0' if current_state == 0 else b'1'  # 0=放下(睁眼) 1=竖起(闭眼)
                    arduino.write(cmd)
                    label = "睁眼 -> 放下" if current_state == 0 else "闭眼 -> 竖起"
                    print(f">>> [{time.strftime('%H:%M:%S')}] {label} <<<")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n退出...")
finally:
    arduino.write(b'1')  # 退出时竖起
    sdk.stop_acquisition()
    arduino.close()
    print("已停止")
