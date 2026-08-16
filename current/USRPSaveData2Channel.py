import numpy as np
import uhd
import threading
import time
import os
from datetime import datetime

# ... (其他参数保持不变) ...
SDR_ARGS = "serial=321D889,num_recv_frames=1024"
SAMPLE_RATE = 30e6  # 30 MHz 采样率（当天如需改 25MHz，把这一行改成 25e6 即可，下面文件夹名会自动跟着变）
CENTER_FREQ = 1.89e9  # 1890 MHz 频段
GAIN = 30
NUM_CHANNELS = 2
FILENAME = "2ch_iq_data.bin"

# sc16 存盘格式：每个复数样点 = 2 个 int16 (re, im)，4 字节。
# UHD 内部 fc32<->sc16 定标系数固定为 1/32767 (host/lib/include/uhdlib/transport/rx_streamer_impl.hpp)，
# 读取时须用同样系数换算回浮点，否则幅度/能量会算错。
SC16_DTYPE = np.dtype([('re', np.int16), ('im', np.int16)])
SC16_SCALE = 32767.0

# 实验文件夹名称：带时间戳，验证脚本可通过「最新 folder」自动用本次采集
# 格式: experiment_{采样率}MHz_static_YYYYMMDD_HHMMSS
# 采样率标签从 SAMPLE_RATE 自动生成（之前是手打的固定字符串 "10MHz"，跟实际 SAMPLE_RATE
# 早就对不上了——下游脚本靠文件夹名反推采样率标签来命名图/GIF，手打字符串很容易和真实值脱节）。
EXPERIMENT_FOLDER = f"experiment_{SAMPLE_RATE/1e6:.0f}MHz_static"

# 采集时长 (秒) - None 表示手动停止（Ctrl+C）
ACQUISITION_DURATION = 10.0  # 10 秒

def create_experiment_folder(folder_name):
    """创建实验文件夹并返回完整路径"""
    # 添加时间戳避免重名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_folder_name = f"{folder_name}_{timestamp}"
    
    # 创建文件夹
    os.makedirs(full_folder_name, exist_ok=True)
    print(f"创建实验文件夹: {full_folder_name}")
    
    return full_folder_name

def save_acquisition_parameters(folder_path, sample_rate, center_freq, gain, 
                               num_channels, acquisition_duration, sdr_args):
    """保存采集参数到JSON文件"""
    import json
    
    params = {
        "sample_rate_hz": sample_rate,
        "sample_rate_mhz": sample_rate / 1e6,
        "center_freq_hz": center_freq,
        "center_freq_ghz": center_freq / 1e9,
        "gain_db": gain,
        "num_channels": num_channels,
        "acquisition_duration_sec": acquisition_duration,
        "sdr_args": sdr_args,
        "timestamp": datetime.now().isoformat(),
        "data_format": "interleaved_sc16",
        "sc16_scale": SC16_SCALE,
        "file_structure": "ch0[0](re,im int16), ch1[0](re,im int16), ch0[1], ch1[1], ..."
    }
    
    params_file = os.path.join(folder_path, "acquisition_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"采集参数已保存到: {params_file}")
    return params_file

def acquire_and_save(stop_event, folder_path, acquisition_duration=None):
    """主采集函数：配置USRP，进行定时启动，并以二进制交错格式连续保存双通道数据"""
    print(f"开始双通道数据采集，采样率: {SAMPLE_RATE/1e6} MHz, 中心频率: {CENTER_FREQ/1e9} GHz")
    if acquisition_duration is not None:
        print(f"采集时长: {acquisition_duration} 秒（将自动停止）")
    else:
        print("采集模式: 连续采集（按 Ctrl+C 手动停止）")

    total_samps_written = 0
    acquisition_start_time = None
    duration_reached = False  # 异常发生在进入循环前时 finally 会引用，必须先初始化

    # <<< MODIFICATION: 将 uhd 对象声明为 None，以便在 finally 中检查 >>>
    usrp = None
    streamer = None

    file_path = os.path.join(folder_path, FILENAME)
    with open(file_path, "ab") as f:
        try:
            usrp = uhd.usrp.MultiUSRP(SDR_ARGS)
            # ... (其他配置保持不变)
            usrp.set_rx_rate(SAMPLE_RATE)
            
            # 显式设置带宽（避免带外干扰）
            usrp.set_rx_bandwidth(SAMPLE_RATE, 0)  # 通道0：带宽 = 采样率
            usrp.set_rx_bandwidth(SAMPLE_RATE, 1)  # 通道1：带宽 = 采样率
            
            # 打印实际设置的采样率和带宽（确认UHD接受了什么）
            print(f"实际采样率 ch0: {usrp.get_rx_rate(0)/1e6:.6f} MHz")
            print(f"实际带宽   ch0: {usrp.get_rx_bandwidth(0)/1e6:.6f} MHz")
            print(f"实际采样率 ch1: {usrp.get_rx_rate(1)/1e6:.6f} MHz")
            print(f"实际带宽   ch1: {usrp.get_rx_bandwidth(1)/1e6:.6f} MHz")
            
            usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ))
            usrp.set_rx_gain(GAIN, 0)
            # 天线配置：Channel 0 使用 TX/RX 端口（双用途端口）
            # 注意：确保物理天线连接到 Channel 0 的 TX/RX 端口
            usrp.set_rx_antenna("RX2", 0)
            usrp.set_rx_gain(GAIN, 1)
            # 天线配置：Channel 1 也使用 TX/RX 端口（双用途端口）
            # 注意：确保物理天线连接到 Channel 1 的 TX/RX 端口
            # 两个通道都使用TX/RX端口，适合双通道接收配置
            usrp.set_rx_antenna("RX2", 1)
            # cpu_format 也用 sc16：wire format 本来就是 sc16 (16-bit 整数 I/Q)，
            # 之前用 fc32 只是让 UHD 在写盘前多转一次 float，白白让文件大一倍、精度没有任何提升。
            st_args = uhd.usrp.StreamArgs("sc16", "sc16")
            st_args.channels = list(range(NUM_CHANNELS))
            streamer = usrp.get_rx_stream(st_args)
            recv_buffer = np.zeros((NUM_CHANNELS, 8192), dtype=SC16_DTYPE)
            metadata = uhd.types.RXMetadata()
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            stream_cmd.stream_now = False
            start_time = usrp.get_time_now() + uhd.libpyuhd.types.time_spec(0.2)
            stream_cmd.time_spec = start_time
            streamer.issue_stream_cmd(stream_cmd)
            print("正在预热数据流...")
            streamer.recv(recv_buffer, metadata, timeout=0.5)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                 print(f"预热时出现一个预料内的错误: {metadata.strerror()} (可忽略)")

            if acquisition_duration is None:
                print(f"串流已稳定！开始写入文件... 按 Ctrl+C 停止。")
            else:
                print(f"串流已稳定！开始写入文件... 将自动采集 {acquisition_duration} 秒。")
            
            # 数据采集循环（静默模式：不打印任何信息，避免影响数据采集稳定性）
            start_collection_time = time.time()
            while not stop_event.is_set():
                # 检查是否达到预设采集时长（静默检查，不打印）
                if acquisition_duration is not None:
                    elapsed_time = time.time() - start_collection_time
                    if elapsed_time >= acquisition_duration:
                        duration_reached = True
                        break
                
                samps = streamer.recv(recv_buffer, metadata)
                
                if acquisition_start_time is None and samps > 0:
                    acquisition_start_time = time.monotonic()
                
                total_samps_written += samps

                # 注意：不在采集循环中打印，避免影响数据采集稳定性
                # 只在严重错误时记录（但不打印）
                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    # 静默处理错误，避免打印影响采集
                    pass
                
                # 修复：只写入有效的样本数，避免写入旧数据
                if samps > 0:
                    f.write(recv_buffer[:, :samps].T.tobytes())

        except Exception as e:
            print(f"线程内发生异常: {e}")
        finally:
            # --- 关键修复：手动、有序地停止和销毁UHD对象 ---
            print("\n正在安全关闭数据流和设备...")
            
            # 1. 停止数据流
            if streamer:
                stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                streamer.issue_stream_cmd(stop_cmd)
            
            # 采集结束后打印信息（如果达到预设时长）
            if duration_reached:
                print(f"\n⏰ 达到预设采集时长 {acquisition_duration} 秒，自动停止采集")
            
            # 打印统计信息
            if acquisition_start_time is not None:
                acquisition_end_time = time.monotonic()
                measured_duration = acquisition_end_time - acquisition_start_time
                theoretical_duration = total_samps_written / SAMPLE_RATE
                
                print("\n--- 采集统计 ---")
                print(f"总计采集样本数 (每通道): {total_samps_written}")
                print(f"实际采集时间: {measured_duration:.6f} 秒")
                print(f"理论采集时间: {theoretical_duration:.6f} 秒")
                print(f"数据文件大小: {os.path.getsize(file_path)/1024/1024:.2f} MB")
                print("------------------")

            # 2. 显式删除对象，触发C++析构函数以释放硬件
            #    这个操作对于防止C++底层崩溃至关重要
            if streamer:
                del streamer
            if usrp:
                del usrp
            
            print("设备已安全关闭。")

def main(acquisition_duration=None):
    """主函数。acquisition_duration 不传时用模块级 ACQUISITION_DURATION（独立运行本文件时的默认值）；
    调用方（如 validate_capture_and_caf.py）可传入自己的时长，一处改动即可同步采集+验证。"""
    if acquisition_duration is None:
        acquisition_duration = ACQUISITION_DURATION
    print("=== USRP B210 双通道数据采集 ===")
    print(f"实验文件夹: {EXPERIMENT_FOLDER}")
    print(f"采样率: {SAMPLE_RATE/1e6:.1f} MHz")
    print(f"中心频率: {CENTER_FREQ/1e9:.3f} GHz")
    print(f"采集时长: {acquisition_duration}s" if acquisition_duration is not None else "采集时长: 手动停止(Ctrl+C)")
    print("=" * 50)

    # 创建实验文件夹
    folder_path = create_experiment_folder(EXPERIMENT_FOLDER)

    # 保存采集参数
    save_acquisition_parameters(
        folder_path=folder_path,
        sample_rate=SAMPLE_RATE,
        center_freq=CENTER_FREQ,
        gain=GAIN,
        num_channels=NUM_CHANNELS,
        acquisition_duration=acquisition_duration,
        sdr_args=SDR_ARGS
    )

    # 创建空的数据文件
    file_path = os.path.join(folder_path, FILENAME)
    with open(file_path, "wb") as f:
        pass

    # 启动采集线程
    stop_event = threading.Event()
    acq_thread = threading.Thread(target=acquire_and_save, args=(stop_event, folder_path, acquisition_duration))
    print("启动采集线程...")
    acq_thread.start()
    try:
        acq_thread.join()
    except KeyboardInterrupt:
        print("\n主线程收到中断信号，正在停止采集线程...")
        stop_event.set()
    acq_thread.join()
    
    print(f"\n数据已保存到: {file_path}")
    print(f"本次采集即「最新文件夹」: {folder_path}")
    print("跑多普勒验证: validate_capture_and_caf.py 设 run_acquisition=True 将先采数再验证；False 则仅用上方最新文件夹验证。")
    print("程序已完全退出。")

if __name__ == "__main__":
    main()