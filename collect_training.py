"""
采集训练数据：睁眼 (label=0) vs 闭眼 (label=1)
全通道 + 更长采集时长
"""
import os
import time
import numpy as np
import EEGAcquisitionSDK as eeg

SAVE_DIR = r"D:\temp\EEGlab\training_data"
DURATION = 20          # 每段采集秒数 (加长到20秒)
ROUNDS = 3             # 每种状态轮数
REST_TIME = 5          # 切换休息秒数
SAMPLE_RATE = 250

os.makedirs(SAVE_DIR, exist_ok=True)

sdk = eeg.EEGAcquisitionSDK(port="COM10", filtered_channels="all")
sdk.start_acquisition()

all_data = []
all_labels = []

def countdown(msg, seconds):
    for i in range(seconds, 0, -1):
        print(f"  {msg}... {i}s ", end="\r")
        time.sleep(1)
    print(" " * 40, end="\r")

print("=" * 50)
print("  睁眼/闭眼 训练数据采集 (全通道)")
print(f"  每种状态 {ROUNDS} 轮, 每轮 {DURATION}s")
print("=" * 50)

try:
    for round_num in range(1, ROUNDS + 1):
        # --- 睁眼 ---
        print(f"\n{'='*40}")
        print(f"  第 {round_num}/{ROUNDS} 轮 — 睁眼")
        print(f"{'='*40}")
        countdown("准备睁眼", 3)
        print("  >>> 请保持睁眼，减少眨眼和身体移动 <<<")
        sdk.clear_data_buffer()

        segments = []
        t0 = time.time()
        while time.time() - t0 < DURATION:
            data = sdk.pop_data(n_samples=250, is_blocking_mode=False)
            if data is not None and data.shape[0] > 0:
                segments.append(data[:, :-1])
            elapsed = time.time() - t0
            print(f"  睁眼... {elapsed:.0f}/{DURATION}s", end="\r")
            time.sleep(0.1)

        eye_open = np.vstack(segments)
        all_data.append(eye_open)
        all_labels.append(np.zeros(len(eye_open)))
        print(f"  睁眼: {len(eye_open)} 样本 ({len(eye_open)/SAMPLE_RATE:.1f}s)")

        # --- 闭眼 ---
        print(f"\n{'='*40}")
        print(f"  第 {round_num}/{ROUNDS} 轮 — 闭眼")
        print(f"{'='*40}")
        countdown("准备闭眼", 3)
        print("  >>> 请闭眼放松，不要睡着 <<<")
        sdk.clear_data_buffer()

        segments = []
        t0 = time.time()
        while time.time() - t0 < DURATION:
            data = sdk.pop_data(n_samples=250, is_blocking_mode=False)
            if data is not None and data.shape[0] > 0:
                segments.append(data[:, :-1])
            elapsed = time.time() - t0
            print(f"  闭眼... {elapsed:.0f}/{DURATION}s", end="\r")
            time.sleep(0.1)

        eye_closed = np.vstack(segments)
        all_data.append(eye_closed)
        all_labels.append(np.ones(len(eye_closed)))
        print(f"  闭眼: {len(eye_closed)} 样本 ({len(eye_closed)/SAMPLE_RATE:.1f}s)")

        if round_num < ROUNDS:
            countdown("休息", REST_TIME)

finally:
    sdk.stop_acquisition()

X = np.vstack(all_data)
y = np.concatenate(all_labels)

save_path = os.path.join(SAVE_DIR, "training_data.npz")
# 通道名 (去掉event后只剩32个通道)
ch_names = ["T5","TP7","O1","P3","CP3","C3","CZ","CPZ","PZ","OZ",
            "C4","CP4","P4","O2","TP8","T6","T3","FT7","F7","FC3",
            "F3","FP1","FZ","FCZ","FP2","F4","FC4","VEOU","F8","FT8","T4","VEOL"]
np.savez(save_path, X=X, y=y, channels=ch_names, sample_rate=SAMPLE_RATE)
print(f"\n{'='*50}")
print(f"  训练数据已保存: {save_path}")
print(f"  总样本: {len(X)}, 睁眼: {int(np.sum(y==0))}, 闭眼: {int(np.sum(y==1))}")
print(f"{'='*50}")
