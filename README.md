# EEG 脑控兔耳朵 🐰🧠

基于 EEG 脑电信号的实时兔耳朵舵机控制系统。通过检测**睁眼/闭眼**状态下枕叶 Alpha 波 (8-13Hz) 功率变化，驱动 Arduino 控制的舵机兔耳朵。

- **闭眼** → Alpha 功率升高 → 耳朵**竖起** 🐰
- **睁眼** → Alpha 被阻断，功率降低 → 耳朵**放下**

---

## 硬件连接

| 设备 | 端口 | 芯片 | 用途 |
|:---|:---|:---|:---|
| EEG 放大器 | COM10 | CH343 | 32通道脑电采集 |
| Arduino | COM9 | CH340 | PCA9685 舵机驱动 |
| 兔耳朵 | — | 2×舵机 | 竖起/放下动作 |

### 电极布局 (国际 10-20)

```
       FP1──FP2
    F7─F3─FZ─F4─F8
    T3─C3─CZ─C4─T4
    T5─P3─PZ─P4─T6
       O1─OZ─O2      ← ★ 枕叶 (Alpha 最强)
```

---

## 环境要求

- **Python 3.12** (SDK wheel 是 cp312，3.13 不兼容)
- Windows 10/11
- Arduino IDE (用于烧录固件)

```powershell
# 安装 EEG SDK
pip install eegacquisitionsdk-0.1.0-cp312-cp312-win_amd64.whl

# 安装依赖
pip install numpy scipy pyserial
```

---

## 项目结构

```
D:\Arduino\
├── rabbit_ears/
│   └── rabbit_ears.ino      # Arduino 固件 (PCA9685 双舵机)
│
├── measure_impedance.py      # 电极阻抗检测
├── collect_training.py       # 训练数据采集 (自动倒计时)
├── collect_training_v2.py    # 训练数据采集 (交互式)
├── train_model.py            # Alpha 阈值模型训练
├── realtime_control.py       # 实时脑控 v1 (需预训练模型)
├── realtime_control_v4.py    # 实时脑控 v4 (自适应基线, 推荐)
├── realtime_control_v5.py    # 实时脑控 v5 (RMS 功率法备选)
│
├── record_eeg.py             # 通用 CSV 录制
├── csv2set.py                # CSV → EEGLAB .set 转换
├── test_acquisition.py       # SDK 连通性测试
│
└── eegacquisitionsdk-0.1.0-cp312-cp312-win_amd64.whl
```

---

## 实验流程

### 1. 检查阻抗

```bash
py -3.12 measure_impedance.py
```

所有通道阻抗应 < 10 kΩ。若偏高（特别是 O1/OZ/O2），需补充导电膏或生理盐水。

### 2. 采集训练数据

```bash
py -3.12 collect_training.py
```

睁眼/闭眼各 3 轮，每轮 20 秒。跟随倒计时提示操作。数据保存至 `D:\temp\EEGlab\training_data\training_data.npz`。

### 3. 训练模型

```bash
py -3.12 train_model.py
```

输出 Alpha 功率阈值和准确率。模型保存至 `D:\temp\EEGlab\training_data\model.pkl`。

### 4. 启动实时脑控 (推荐 v4)

```bash
py -3.12 realtime_control_v4.py
```

v4 使用**自适应动态基线**，无需预训练模型。前 10 秒自动学习基线，之后实时检测睁眼/闭眼并控制 Arduino。

---

## 核心算法

### 信号处理链

```
原始 EEG (32通道 × 250Hz)
    │
    ▼
[枕叶通道选取]  O1, OZ, O2  (Alpha 最强的视觉皮层区域)
    │
    ▼
[去趋势]  detrend 去除直流漂移
    │
    ▼
[带通滤波]  Butterworth 8-13Hz, 4阶  (只保留 Alpha 频段)
    │
    ▼
[功率谱估计]  Welch 方法 (汉宁窗 → FFT → |·|² → PSD)
    │
    ▼
[频段积分]  Σ PSD[8-13Hz] → 每通道 Alpha 功率
    │
    ▼
[通道平均]  mean(O1, OZ, O2) → 单一特征值
    │
    ▼
[动态基线]  基线 = median(最近15个睁眼窗口的Alpha值)
    │
    ▼
[阈值判定]  α > 基线 × 2.0 → 闭眼, 否则 → 睁眼
    │
    ▼
[滑动投票]  5窗多数投票 (消除瞬态误判)
    │
    ▼
[串口输出]  闭眼→'1'(竖起), 睁眼→'0'(放下)
```

### 核心参数

| 参数 | 值 | 说明 |
|:---|:---|:---|
| 采样率 | 250 Hz | SDK 固定 |
| 分析窗口 | 2 秒 (500点) | 提供 0.5 Hz 频率分辨率 |
| Alpha 频段 | 8-13 Hz | 标准 Alpha 定义 |
| 枕叶通道 | O1, OZ, O2 | 视觉皮层 |
| 带通滤波 | Butterworth 4阶 | 8-13 Hz |
| Welch nperseg | 250 点 | 3段平均降低方差 |
| 阈值倍数 | 2.0× 基线 | 自适应动态调整 |
| 平滑窗口 | 5 次 | 约 2.5 秒决策延迟 |

### 生理基础: Alpha 阻断 (Berger 效应)

- **闭眼**: 无视觉输入 → 视觉皮层神经元同步放电 → Alpha 功率 **升高**
- **睁眼**: 大量视觉刺激 → 神经元去同步化 → Alpha 功率 **降低**

这是 EEG 领域最古老 (1929年发现) 也是最可靠的信号特征。

---

## Arduino 固件

位于 `rabbit_ears/rabbit_ears.ino`，使用 Arduino IDE 烧录。

- 芯片: PCA9685 (I²C 舵机驱动板)
- 舵机频率: 50 Hz (PWM 范围 150-400)
- 串口: 115200 baud
- 指令: `'1'` = 竖起, `'0'` = 放下

---

## 版本说明

| 版本 | 方法 | 特点 |
|:---|:---|:---|
| v1 | 固定阈值 Alpha | 需预训练模型，简单直接 |
| v4 ★ | 自适应基线 Alpha | Butterworth 滤波 + 动态基线，推荐使用 |
| v5 | RMS 总功率 | 不依赖 Alpha，高阻抗备选方案 |

---

## 常见问题

**信号质量差 / 准确率低？**
用 `measure_impedance.py` 检查阻抗。正常应 < 10 kΩ。若 > 100 kΩ，导电膏已干，需全部重新打。

**COM 口不通？**
- EEG: 设备管理器确认 CH343 端口号
- Arduino: 关闭 Arduino IDE 串口监视器释放端口

**SDK 导入失败？**
必须用 Python 3.12。检查: `py -3.12 --version`
