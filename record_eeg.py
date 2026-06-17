import time
import EEGAcquisitionSDK as eeg

save_path = r"D:\temp\EEGlab\EEGData"
duration = 30  # 采集时长（秒）

sdk = eeg.EEGAcquisitionSDK(port="auto", filtered_channels="all")
sdk.start_acquisition()
sdk.start_recording_eeg_data(eeg_data_directory=save_path)

print(f"正在录制到: {save_path}")
print(f"采集时长: {duration} 秒")

start = time.time()
while time.time() - start < duration:
    remaining = duration - (time.time() - start)
    data = sdk.read_latest_data(n_samples=250)
    print(f"剩余 {remaining:.0f}s  |  数据: {data.shape}", end="\r")
    time.sleep(0.1)

print("\n采集完成。")
sdk.stop_recording_eeg_data()
sdk.stop_acquisition()
print("已停止。")
