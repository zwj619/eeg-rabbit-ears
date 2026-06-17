"""
测量所有通道阻抗，重点标注枕叶通道 O1/O2/OZ
"""
import time
import EEGAcquisitionSDK as eeg

CH_NAMES = ["T5","TP7","O1","P3","CP3","C3","CZ","CPZ","PZ","OZ",
            "C4","CP4","P4","O2","TP8","T6","T3","FT7","F7","FC3",
            "F3","FP1","FZ","FCZ","FP2","F4","FC4","VEOU","F8","FT8","T4","VEOL"]

CRITICAL = {"O1", "O2", "OZ"}
OK_THRESHOLD = 10  # kΩ

print("=" * 55)
print("  电极阻抗测量")
print("=" * 55)

sdk = eeg.EEGAcquisitionSDK(port="COM10", filtered_channels="all")

try:
    sdk.measure_impedance()
    print("  测量中...")
    time.sleep(3)
    imp = sdk.impedance

    if imp is None:
        print("  !! 无法读取阻抗数据")
    else:
        print(f"\n  {'通道':<8} {'阻抗 (kΩ)':>10}  状态")
        print("  " + "-" * 40)

        bad_count = 0
        bad_critical = []

        for i, (ch, val) in enumerate(zip(CH_NAMES, imp)):
            marker = "  !! 过高!" if val > OK_THRESHOLD else "  OK"
            if val > OK_THRESHOLD:
                bad_count += 1
                if ch in CRITICAL:
                    bad_critical.append(ch)

            flag = " ★" if ch in CRITICAL else ""
            print(f"  {ch:<8}{flag} {val:>8.1f} kΩ  {marker}")

        print("  " + "-" * 40)
        print(f"\n  总通道: {len(imp)}, 异常(>{OK_THRESHOLD}kΩ): {bad_count}")

        if bad_critical:
            print(f"  !! 枕叶关键通道异常: {', '.join(bad_critical)}")
            print(f"  !! 请给这些电极补充导电膏/盐水后重测!")
        elif bad_count == 0:
            print(f"  全部通道 OK，可以开始采集训练数据。")
        else:
            print(f"  非关键通道有异常，可以尝试采集。")

finally:
    try:
        sdk.stop_measure_impedance()
    except Exception:
        pass
    sdk.stop_acquisition()

print("=" * 55)
