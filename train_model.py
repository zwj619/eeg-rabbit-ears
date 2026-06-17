"""
简单 Alpha 功率阈值分类器 (O1/O2/OZ)
比 LDA 更鲁棒，不易过拟合
"""
import numpy as np
from scipy.signal import welch
import pickle

DATA_PATH = r"D:\temp\EEGlab\training_data\training_data.npz"
MODEL_PATH = r"D:\temp\EEGlab\training_data\model.pkl"
SAMPLE_RATE = 250
WINDOW_SEC = 1.0
ALPHA_LOW, ALPHA_HIGH = 8, 13


def alpha_power(window, sf):
    freqs, psd = welch(window.T, sf, nperseg=min(sf, window.shape[0]), axis=-1)
    mask = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
    return np.sum(psd[:, mask], axis=-1)  # 每通道的 Alpha 绝对功率


print("加载训练数据...")
data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]
channels = list(data["channels"])
print(f"数据: {X.shape}, 睁眼={(y==0).sum()}, 闭眼={(y==1).sum()}")

# 只用 O1, O2, OZ 的 Alpha 功率之和
o_idx = [i for i, ch in enumerate(channels) if ch in ["O1", "O2", "OZ"]]
print(f"枕叶通道索引: {[channels[i] for i in o_idx]}")

win_samples = int(WINDOW_SEC * SAMPLE_RATE)
n_windows = len(X) // win_samples

open_powers = []
closed_powers = []
for i in range(n_windows):
    seg = X[i * win_samples : (i + 1) * win_samples]
    ap = alpha_power(seg, SAMPLE_RATE)
    occ_alpha = np.mean(ap[o_idx])  # O1/O2/OZ 平均 Alpha 功率
    if y[i * win_samples] == 0:
        open_powers.append(occ_alpha)
    else:
        closed_powers.append(occ_alpha)

open_powers = np.array(open_powers)
closed_powers = np.array(closed_powers)

print(f"睁眼 Alpha 功率: mean={open_powers.mean():.2f} std={open_powers.std():.2f}")
print(f"闭眼 Alpha 功率: mean={closed_powers.mean():.2f} std={closed_powers.std():.2f}")

# 阈值 = 两类均值的中点
threshold = (open_powers.mean() + closed_powers.mean()) / 2.0

# 如果闭眼 Alpha 更高（正常情况），则高于阈值为闭眼
# 如果睁眼 Alpha 更高（异常），反转方向
if closed_powers.mean() > open_powers.mean():
    direction = 1  # 高于阈值 → 闭眼
else:
    direction = -1  # 高于阈值 → 睁眼

# 计算准确率
correct = 0
for ap in open_powers:
    pred = 1 if (ap - threshold) * direction > 0 else 0
    if pred == 0:
        correct += 1
for ap in closed_powers:
    pred = 1 if (ap - threshold) * direction > 0 else 0
    if pred == 1:
        correct += 1
acc = correct / (len(open_powers) + len(closed_powers))
print(f"阈值: {threshold:.2f}, 方向: {'闭眼>阈值' if direction == 1 else '睁眼>阈值'}")
print(f"准确率: {acc:.1%}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "threshold": threshold,
        "direction": direction,
        "occipital_channels": ["O1", "O2", "OZ"],
        "occipital_idx": o_idx,
        "channels": channels,
        "sample_rate": SAMPLE_RATE,
        "window_sec": WINDOW_SEC,
        "alpha_low": ALPHA_LOW,
        "alpha_high": ALPHA_HIGH,
    }, f)
print(f"模型已保存: {MODEL_PATH}")
