import EEGAcquisitionSDK as eeg

# port="auto" 会自动搜索 CH34/CH343 串口设备
sdk = eeg.EEGAcquisitionSDK(port="auto", filtered_channels="all")
print("正在启动采集...")
sdk.start_acquisition()
print("采集已启动！按 Ctrl+C 停止\n")

try:
    while True:
        data = sdk.read_latest_data(n_samples=250)
        print(f"数据形状: {data.shape}", end="\r")
except KeyboardInterrupt:
    print("\n\n停止采集...")
finally:
    sdk.stop_acquisition()
    print("采集已停止。")
