"""
训练分类器 v3 — detrend去直流 + 8-30Hz带通, 训练实时一致
"""
import numpy as np
from scipy.signal import butter, filtfilt, welch, detrend
import pickle

DATA_PATH = r"D:\temp\EEGlab\training_data\training_data.npz"
MODEL_PATH = r"D:\temp\EEGlab\training_data\model_v3.pkl"
SAMPLE_RATE = 250
WINDOW_SEC = 2.0
ALPHA_LOW, ALPHA_HIGH = 8, 13


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a


def alpha_ratio(window, sf):
    freqs, psd = welch(window.T, sf, nperseg=min(sf, window.shape[0]), axis=-1)
    mask_alpha = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
    mask_total = (freqs >= 4) & (freqs <= 45)
    alpha_pow = psd[:, mask_alpha].sum(axis=-1)
    total_pow = psd[:, mask_total].sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_pow > 0, alpha_pow / total_pow, 0)
    return ratio


print("加载训练数据...")
data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]
channels = list(data["channels"])
print(f"原始数据: {X.shape}, 睁眼={(y == 0).sum()}, 闭眼={(y == 1).sum()}")

occ_channels = ["T5", "TP7", "O1", "P3", "PZ", "OZ", "P4", "O2"]
available_occ = [(i, ch) for i, ch in enumerate(channels) if ch in occ_channels]
occ_idx = [i for i, ch in available_occ]
occ_names = [ch for i, ch in available_occ]
print(f"枕顶叶通道: {occ_names}")

# 8-30Hz 带通 (detrend 先去直流, 无需高通)
b_bp, a_bp = butter_bandpass(8, 30, SAMPLE_RATE)

win_samples = int(WINDOW_SEC * SAMPLE_RATE)
n_windows = len(X) // win_samples

open_ratios = []
closed_ratios = []

for i in range(n_windows):
    seg = X[i * win_samples : (i + 1) * win_samples].copy()

    # detrend 去直流 (每通道独立, 训练实时行为一致)
    for ch in range(seg.shape[1]):
        seg[:, ch] = detrend(seg[:, ch])

    # 8-30Hz 带通
    seg_filt = filtfilt(b_bp, a_bp, seg, axis=0)

    ratios = alpha_ratio(seg_filt, SAMPLE_RATE)
    occ_r = np.mean(ratios[occ_idx])

    if y[i * win_samples] == 0:
        open_ratios.append(occ_r)
    else:
        closed_ratios.append(occ_r)

open_ratios = np.array(open_ratios)
closed_ratios = np.array(closed_ratios)

print(f"窗口总数: {n_windows}, 睁眼={len(open_ratios)}, 闭眼={len(closed_ratios)}")
print(f"睁眼 Alpha占比: mean={open_ratios.mean():.6f} std={open_ratios.std():.6f}")
print(f"闭眼 Alpha占比: mean={closed_ratios.mean():.6f} std={closed_ratios.std():.6f}")

threshold = (open_ratios.mean() + closed_ratios.mean()) / 2.0

if closed_ratios.mean() > open_ratios.mean():
    direction = 1
    print("方向: 闭眼 Alpha 占比更高 (正常)")
else:
    direction = -1
    print("方向: 睁眼 Alpha 占比更高 (异常)")

correct = 0
for r in open_ratios:
    pred = 1 if (r - threshold) * direction > 0 else 0
    if pred == 0:
        correct += 1
for r in closed_ratios:
    pred = 1 if (r - threshold) * direction > 0 else 0
    if pred == 1:
        correct += 1
acc = correct / (len(open_ratios) + len(closed_ratios))
print(f"阈值: {threshold:.6f}, 准确率: {acc:.1%}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "threshold": threshold,
        "direction": direction,
        "occipital_channels": occ_names,
        "occipital_idx": occ_idx,
        "channels": channels,
        "sample_rate": SAMPLE_RATE,
        "window_sec": WINDOW_SEC,
        "alpha_low": ALPHA_LOW,
        "alpha_high": ALPHA_HIGH,
    }, f)
print(f"模型已保存: {MODEL_PATH}")
