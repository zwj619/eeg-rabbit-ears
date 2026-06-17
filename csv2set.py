import numpy as np
import pandas as pd
from scipy.io import savemat
import os

csv_path = r"D:\temp\EEGlab\EEGData\recording_data_2026-06-03_20-58-35_240.csv"

# 读取 CSV
df = pd.read_csv(csv_path)
ch_names = list(df.columns[:-1])  # 前32列是通道名, 最后一列是event
event = df["event"].values
data = df.iloc[:, :-1].values.T  # 转置为 channels × samples (32 × 7490)

# 构建 EEGLAB 结构体
EEG = {}
EEG["setname"] = os.path.splitext(os.path.basename(csv_path))[0]
EEG["filename"] = EEG["setname"] + ".set"
EEG["filepath"] = os.path.dirname(csv_path)
EEG["data"] = data                    # channels × samples
EEG["srate"] = 250.0
EEG["pnts"] = data.shape[1]
EEG["nbchan"] = data.shape[0]
EEG["trials"] = 1
EEG["xmin"] = 0.0
EEG["xmax"] = (data.shape[1] - 1) / 250.0
EEG["times"] = np.arange(data.shape[1]) / 250.0
EEG["ref"] = "common"
EEG["icawinv"] = []
EEG["icaweights"] = []
EEG["icasphere"] = []
EEG["event"] = np.zeros((1,), dtype=[("type", "O"), ("latency", "O"), ("urevent", "O")])

# 通道位置 (标准 EEGLAB 字段)
chanlocs = np.zeros(data.shape[0], dtype=[
    ("labels", "O"),
    ("theta", "O"),
    ("radius", "O"),
    ("X", "O"),
    ("Y", "O"),
    ("Z", "O"),
    ("sph_theta", "O"),
    ("sph_phi", "O"),
    ("sph_radius", "O"),
    ("type", "O"),
    ("urchan", "O"),
])
for i, ch in enumerate(ch_names):
    chanlocs[i] = (ch, 0, 0, 0, 0, 0, 0, 0, 0, "", i + 1)
EEG["chanlocs"] = chanlocs

# 保存为 .set (实际是 MATLAB .mat 格式)
out_path = csv_path.replace(".csv", ".set")
savemat(out_path, {"EEG": EEG}, do_compression=True)
print(f"已转换: {out_path}")
print(f"通道: {data.shape[0]}, 采样点: {data.shape[1]}, 采样率: 250Hz, 时长: {data.shape[1]/250:.1f}s")
