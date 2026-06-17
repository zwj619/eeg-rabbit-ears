"""
采集训练数据 v2 — 交互式"开始"控制
睁眼 (label=0) vs 闭眼 (label=1)
"""
import os
import time
import numpy as np
import EEGAcquisitionSDK as eeg

SAVE_DIR = r"D:\temp\EEGlab\training_data"
DURATION = 20
ROUNDS = 3
REST_TIME = 5
SAMPLE_RATE = 250

os.makedirs(SAVE_DIR, exist_ok=True)

sdk = eeg.EEGAcquisitionSDK(port="COM10", filtered_channels="all")
sdk.start_acquisition()

all_data = []
all_labels = []


def wait_for_start(prompt="输入 '开始' 启动"):
    """等待用户输入开始指令"""
    print(f"\n  {prompt}")
    while True:
        cmd = input("  >>> ").strip()
        if cmd == "开始":
            break
        else:
            print(f"  请输入 '开始' 来启动")


def record_segment(duration, label_name):
    """录制一段数据，返回 numpy 数组"""
    sdk.clear_data_buffer()
    segments = []
    t0 = time.time()
    while time.time() - t0 < duration:
        data = sdk.pop_data(n_samples=250, is_blocking_mode=False)
        if data is not None and data.shape[0] > 0:
            segments.append(data[:, :-1])
        elapsed = time.time() - t0
        print(f"  {label_name}... {elapsed:.0f}/{duration}s", end="\r")
        time.sleep(0.1)
    return np.vstack(segments)


# ============ 阻抗检查 ============
print("=" * 50)
print("  电极阻抗检查")
print("=" * 50)
try:
    sdk.measure_impedance()
    time.sleep(3)
    imp = sdk.impedance
    if imp is not None:
        print(f"  当前阻抗: {imp}")
        high = [v for v in imp if v > 10]
        if high:
            print(f"  !! 警告: {len(high)} 个通道阻抗 > 10 kΩ, 请补导电膏!")
        else:
            print("  阻抗 OK，可以开始采集。")
    else:
        print("  无法读取阻抗，跳过检查")
except Exception as ex:
    print(f"  阻抗测量跳过: {ex}")
    try:
        sdk.stop_measure_impedance()
    except Exception:
        pass

wait_for_start("输入 '开始' 启动采集流程")

# ============ 采集 ============
print("\n" + "=" * 50)
print(f"  睁眼/闭眼 训练数据采集")
print(f"  每种状态 {ROUNDS} 轮, 每轮 {DURATION}s")
print("=" * 50)

try:
    for round_num in range(1, ROUNDS + 1):
        # --- 睁眼 ---
        print(f"\n{'='*40}")
        print(f"  第 {round_num}/{ROUNDS} 轮 — 睁眼")
        print(f"{'='*40}")
        wait_for_start(f"第{round_num}轮睁眼 — 输入 '开始' 启动")
        print("  >>> 请保持睁眼，减少眨眼 <<<")

        eye_open = record_segment(DURATION, "睁眼")
        all_data.append(eye_open)
        all_labels.append(np.zeros(len(eye_open)))
        print(f"\n  睁眼: {len(eye_open)} 样本 ({len(eye_open)/SAMPLE_RATE:.1f}s)")

        # --- 闭眼 ---
        print(f"\n{'='*40}")
        print(f"  第 {round_num}/{ROUNDS} 轮 — 闭眼")
        print(f"{'='*40}")
        wait_for_start(f"第{round_num}轮闭眼 — 输入 '开始' 启动")
        print("  >>> 请闭眼放松 <<<")

        eye_closed = record_segment(DURATION, "闭眼")
        all_data.append(eye_closed)
        all_labels.append(np.ones(len(eye_closed)))
        print(f"\n  闭眼: {len(eye_closed)} 样本 ({len(eye_closed)/SAMPLE_RATE:.1f}s)")

        if round_num < ROUNDS:
            print(f"\n  --- 休息 {REST_TIME}s ---")
            time.sleep(REST_TIME)

finally:
    sdk.stop_acquisition()

X = np.vstack(all_data)
y = np.concatenate(all_labels)

save_path = os.path.join(SAVE_DIR, "training_data.npz")
ch_names = ["T5","TP7","O1","P3","CP3","C3","CZ","CPZ","PZ","OZ",
            "C4","CP4","P4","O2","TP8","T6","T3","FT7","F7","FC3",
            "F3","FP1","FZ","FCZ","FP2","F4","FC4","VEOU","F8","FT8","T4","VEOL"]
np.savez(save_path, X=X, y=y, channels=ch_names, sample_rate=SAMPLE_RATE)
print(f"\n{'='*50}")
print(f"  训练数据已保存: {save_path}")
print(f"  总样本: {len(X)}, 睁眼: {int(np.sum(y==0))}, 闭眼: {int(np.sum(y==1))}")
print(f"{'='*50}")
