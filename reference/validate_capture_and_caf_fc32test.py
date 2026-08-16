#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAF 逻辑验证脚本 —— fc32(旧存盘格式)版本

只用来回归测试：validate_capture_and_caf.py 把存盘格式从 fc32 换成 sc16、并重写内存优化
(decim 提前到转换前)之后，CAF/PDP 的算法逻辑本身有没有被改坏。做法是用同一套 CAF/PDP/
spectrogram 代码，去读一份已知是旧 fc32 格式的历史数据(experiment_30MHz_static_20260227_150212，
data_format=interleaved_complex64)，跟 sc16 版本对比结果是否一致。

跟 validate_capture_and_caf.py 的唯一区别：read_2ch_iq_memmap / sc16_to_complex64(此处改成
直接返回 fc32 memmap 切片，不做定标换算) / check_data_format / check_iq_magnitude_abort_memmap
这几个"数据格式相关"的函数改成认 fc32；CAF/PDP/GIF 的算法逻辑一字未动。

- Ref=ch0(Tx), Sensing=ch1。两通道 |IQ|>1.4 检查；Tw=0.4s, step=0.02s。
- PDP: delay -3~+3，选最强的 delay 做 CAF。相位: spectrogram mode='complex'，去旋转 exp(-j*2*pi*f*t)，折叠 0 单独、+f×(-f)。
"""

import os
import glob
import re
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

# 数据读取与 doppler_3d_time_delay.py 一致：本文件内 read_2ch_iq_memmap，不整盘加载。
# 内存优化：memmap 打开后，饱和检查(check_iq_magnitude_abort_memmap)分块扫描，
# PDP(compute_pdp_delay_memmap)只读 0.1s 小窗口，CAF(build_decimated_channel)先按 decim=16
# 跳采样切片再转 complex64——全程不把整段采集数据摊在内存里。

# 旧 fc32 存盘格式：每个复数样点 = 2 个 float32 (re, im)，8 字节，memmap 本身已经是
# complex64，不需要 sc16 那套 int16/32767 定标换算。
FC32_DTYPE = np.complex64

MAGNITUDE_ABORT_THRESHOLD = 1.4
MAGNITUDE_ABORT_COUNT = 100
TW_SEC = 0.4
STEP_SEC = 0.02
DECIM = 16
PDP_WINDOW_SEC = 0.1
PDP_DELAY_NEG = 3   # PDP 范围 -3..+3，选最强的 delay
PDP_DELAY_POS = 3
# GIF 用：delay 从 -3 扫到 +3，每帧 0.5s，距离标注
DELAY_MIN_GIF = -3
DELAY_MAX_GIF = 3
GIF_FRAME_DURATION_SEC = 0.5
C_LIGHT = 3e8  # m/s
# max_duration_sec=None 时默认只读 60 秒，避免整盘加载崩溃
DEFAULT_MAX_DURATION_SEC = 60.0


def range_step_for_sample_rate(sample_rate_hz):
    """delay(单位: 未降采样的原始 sample)每差 1 对应的距离 = c / (2*sample_rate)（往返）。
    之前是硬编码 RANGE_FIRST_BIN_MAX_M=2.5 / RANGE_STEP_M=5.0（正好是 30MHz 时的值），
    换成 25MHz 采样率而没改这两个数，GIF 上标的距离会是错的，所以改成跟着实际采样率算。"""
    step_m = C_LIGHT / (2.0 * sample_rate_hz)
    first_bin_max_m = step_m / 2.0
    return first_bin_max_m, step_m


def read_2ch_iq_memmap(filename: str):
    """memmap 读双通道 IQ (旧 fc32 存盘：每复数样点 8 字节 float32 I/Q)，不整盘加载。"""
    try:
        file_size = os.path.getsize(filename)
        total_complex = file_size // 8
        total_per_ch = total_complex // 2
        data_memmap = np.memmap(filename, dtype=FC32_DTYPE, mode='r', shape=(total_complex,))
        ch0_memmap = data_memmap[0::2]
        ch1_memmap = data_memmap[1::2]
        return ch0_memmap, ch1_memmap, total_per_ch
    except Exception as e:
        print(f"Error creating memory map for {filename}: {e}")
        return None, None, 0


def sc16_to_complex64(fc32_arr):
    """fc32 版本：memmap 切片本身已经是 complex64，不需要任何定标换算，只是物化成普通
    ndarray。函数名沿用 sc16 版本，是为了让 build_decimated_channel/compute_pdp_delay_memmap
    这些下游函数完全不用改——这样才能验证「换了存盘格式之后 CAF/PDP 算法本身没被改坏」。"""
    return np.array(fc32_arr, dtype=np.complex64)


def check_data_format(params, filename):
    """本 fc32 验证脚本只用来处理旧的 interleaved_complex64 数据，跟 sc16 主脚本反过来。"""
    data_format = params.get("data_format") if params else None
    if data_format != "interleaved_complex64":
        print(f"错误: {filename} 的 data_format={data_format!r}，本 fc32 验证脚本只处理 "
              f"interleaved_complex64 格式的旧数据，已中止。")
        return False
    return True


def find_latest_experiment_folder():
    experiment_folders = glob.glob("experiment_*")
    if not experiment_folders:
        return None
    pattern = r'experiment_.*_(\d{8})_(\d{6})'
    folders_with_timestamp = []
    for folder in experiment_folders:
        match = re.search(pattern, folder)
        if match:
            date_str, time_str = match.group(1), match.group(2)
            try:
                ts = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                folders_with_timestamp.append((ts, folder))
            except ValueError:
                folders_with_timestamp.append((datetime.fromtimestamp(os.path.getctime(folder)), folder))
        else:
            folders_with_timestamp.append((datetime.fromtimestamp(os.path.getctime(folder)), folder))
    return max(folders_with_timestamp, key=lambda x: x[0])[1]


def load_acquisition_parameters(folder_path):
    path = os.path.join(folder_path, "acquisition_parameters.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def run_step1_acquisition():
    import USRPSaveData2Channel
    USRPSaveData2Channel.main()


def compute_pdp_delay_memmap(ch0_memmap, ch1_memmap, n_total, sample_rate_hz,
                              delay_neg=PDP_DELAY_NEG, delay_pos=PDP_DELAY_POS, window_sec=PDP_WINDOW_SEC):
    """PDP 只需要开头 window_sec(默认 0.1s) 的一小段原始采样率数据，直接从 memmap 切这一小段
    再转 complex64，不需要（也不应该）把整段采集数据转出来才能算 PDP。"""
    K_neg, K_pos = delay_neg, delay_pos
    delay_axis = np.arange(-K_neg, K_pos + 1, dtype=np.int32)
    n_pdp = min(int(sample_rate_hz * window_sec), n_total)
    n_need = max(K_neg, K_pos) + 1
    if n_pdp <= n_need:
        return 0, np.zeros(len(delay_axis), dtype=np.float64), delay_axis
    ch0_seg = sc16_to_complex64(ch0_memmap[:n_pdp])
    ch1_seg = sc16_to_complex64(ch1_memmap[:n_pdp])
    pdp_amp = np.zeros(len(delay_axis), dtype=np.float64)
    for i, d in enumerate(delay_axis):
        if d >= 0:
            pdp_amp[i] = np.abs(np.sum(np.conj(ch0_seg[: n_pdp - d]) * ch1_seg[d:n_pdp]))
        else:
            pdp_amp[i] = np.abs(np.sum(np.conj(ch0_seg[-d:n_pdp]) * ch1_seg[: n_pdp + d]))
    delay_max = int(delay_axis[np.argmax(pdp_amp)])
    return delay_max, pdp_amp, delay_axis


MAGNITUDE_CHECK_CHUNK = 5_000_000  # 每块样点数，分块扫描，避免整段转 complex64


def check_iq_magnitude_abort_memmap(ch0_memmap, ch1_memmap, n_total,
                                     threshold=MAGNITUDE_ABORT_THRESHOLD, max_count=MAGNITUDE_ABORT_COUNT,
                                     chunk_size=MAGNITUDE_CHECK_CHUNK):
    """fc32 版本：memmap 已经是 complex64，分块直接 abs() 比较，不需要 sc16 那套整数运算。"""
    n0 = n1 = 0
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        n0 += int(np.sum(np.abs(ch0_memmap[start:end]) > threshold))
        n1 += int(np.sum(np.abs(ch1_memmap[start:end]) > threshold))
        if n0 + n1 > max_count:
            break
    return (n0 + n1) <= max_count, n0, n1


def build_decimated_channel(ch_memmap, offset, n_total, decim):
    """先在原始 sc16 memmap 上按 decim 跳采样切片，再转 complex64——
    切片后数组只有 n_total/decim 个点，比先转 complex64 整段再切片小 decim 倍(默认 16x)。
    这是本文件里能把 10s/4GB 降到几百 MB 的关键一步。"""
    sc16_slice = ch_memmap[offset:n_total:decim]
    return sc16_to_complex64(sc16_slice)


def fast_caf_spectrogram(ch0_memmap, ch1_memmap, n_total, sample_rate_hz, Tw=TW_SEC, step=STEP_SEC, decim=DECIM, freq_range=(-600, 600),
                         use_pdp=True, pdp_delay_neg=PDP_DELAY_NEG, pdp_delay_pos=PDP_DELAY_POS, pdp_window_sec=PDP_WINDOW_SEC):
    """ch0_memmap/ch1_memmap 是原始 sc16 memmap（未转 complex、未截断），n_total 是两通道
    共同可用的样点数。PDP 只读一小段窗口；CAF 数据在 build_decimated_channel 里先 decim
    再转 complex64，全程不会把整段采集数据以 complex 形式摊开在内存里。"""
    pdp_amp, pdp_delay_axis = None, None
    if use_pdp:
        delay_max, pdp_amp, pdp_delay_axis = compute_pdp_delay_memmap(ch0_memmap, ch1_memmap, n_total, sample_rate_hz, delay_neg=pdp_delay_neg, delay_pos=pdp_delay_pos, window_sec=pdp_window_sec)
    else:
        delay_max = 0
    if delay_max >= 0:
        ch0_ds = build_decimated_channel(ch0_memmap, 0, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, delay_max, n_total, decim)
    else:
        ch0_ds = build_decimated_channel(ch0_memmap, -delay_max, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, 0, n_total, decim)
    n_ds = min(len(ch0_ds), len(ch1_ds))
    ch0_ds, ch1_ds = ch0_ds[:n_ds], ch1_ds[:n_ds]
    effective_fs = sample_rate_hz / decim
    prod_stream = np.conj(ch0_ds) * ch1_ds
    nperseg = int(effective_fs * Tw)
    noverlap = int(effective_fs * (Tw - step))
    win = windows.blackman(nperseg)
    f, t, Sxx = spectrogram(prod_stream, fs=effective_fs, window=win, nperseg=nperseg, noverlap=noverlap, return_onesided=False, mode="complex")
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
    fmin, fmax = freq_range
    mask = (f >= fmin) & (f <= fmax)
    f_axis = f[mask]
    Sxx_db = Sxx_db[mask, :]
    Sxx_complex = Sxx[mask, :]
    return f_axis, t, Sxx_db, Sxx_complex, effective_fs, delay_max, pdp_amp, pdp_delay_axis


def fast_caf_spectrogram_at_delay(ch0_memmap, ch1_memmap, delay_bin, n_total, sample_rate_hz, Tw=TW_SEC, step=STEP_SEC, decim=DECIM, freq_range=(-600, 600)):
    """固定 delay_bin 的 CAF spectrogram，返回 f_axis, t, Sxx_db。ch0_memmap/ch1_memmap 是原始
    sc16 memmap，这里直接按 delay_bin 偏移 + decim 跳采样切片再转 complex64，不会先把整段
    数据转成 complex64。"""
    if delay_bin >= 0:
        ch0_ds = build_decimated_channel(ch0_memmap, 0, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, delay_bin, n_total, decim)
    else:
        ch0_ds = build_decimated_channel(ch0_memmap, -delay_bin, n_total, decim)
        ch1_ds = build_decimated_channel(ch1_memmap, 0, n_total, decim)
    n_ds = min(len(ch0_ds), len(ch1_ds))
    ch0_ds, ch1_ds = ch0_ds[:n_ds], ch1_ds[:n_ds]
    effective_fs = sample_rate_hz / decim
    prod_stream = np.conj(ch0_ds) * ch1_ds
    nperseg = int(effective_fs * Tw)
    noverlap = int(effective_fs * (Tw - step))
    win = windows.blackman(nperseg)
    f, t, Sxx = spectrogram(prod_stream, fs=effective_fs, window=win, nperseg=nperseg, noverlap=noverlap, return_onesided=False, mode="complex")
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
    fmin, fmax = freq_range
    mask = (f >= fmin) & (f <= fmax)
    f_axis = f[mask]
    Sxx_db = Sxx_db[mask, :]
    return f_axis, t, Sxx_db


def range_label_for_delay(delay_bin, first_bin_max_m, step_m):
    """Delay 对应距离 (m)：delay 0 -> 0~half_step, 1 -> half_step~half_step+step, ...
    first_bin_max_m/step_m 请用 range_step_for_sample_rate(sample_rate_hz) 算，不要手填常数。"""
    if delay_bin <= 0:
        return 0.0, first_bin_max_m
    r_min = first_bin_max_m + (delay_bin - 1) * step_m
    r_max = first_bin_max_m + delay_bin * step_m
    return r_min, r_max


def run_validation(experiment_folder, Tw=TW_SEC, step=STEP_SEC, freq_range=(-600, 600), skip_magnitude_check=False,
                   max_duration_sec=None, use_pdp=True, show_phase=True):
    if experiment_folder is None:
        experiment_folder = find_latest_experiment_folder()
        if experiment_folder is None:
            print("未找到任何 experiment_* 文件夹")
            return None
    if not os.path.isdir(experiment_folder):
        print(f"文件夹不存在: {experiment_folder}")
        return None
    data_file = os.path.join(experiment_folder, "2ch_iq_data.bin")
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return None
    params = load_acquisition_parameters(experiment_folder)
    if not check_data_format(params, data_file):
        return None
    sample_rate_hz = params["sample_rate_hz"] if params and "sample_rate_hz" in params else 20e6
    use_sec = max_duration_sec if max_duration_sec is not None else DEFAULT_MAX_DURATION_SEC
    print(f"验证: 打开 memmap，加载前 {use_sec}s 数据...", flush=True)
    ch0_memmap, ch1_memmap, total_per_ch = read_2ch_iq_memmap(data_file)
    if ch0_memmap is None:
        print("内存映射创建失败")
        return None
    n_total = min(total_per_ch, int(sample_rate_hz * use_sec))
    if n_total < 2:
        return None
    print(f"验证: {n_total} 样点/通道，按 decim={DECIM} 跳采样处理 (峰值内存约 "
          f"{n_total*8*2/DECIM/(1024*1024):.0f} MB，而不是整段 {n_total*8*2/(1024*1024):.0f} MB)...", flush=True)
    effective_fs = sample_rate_hz / DECIM
    nperseg = int(effective_fs * Tw)
    if n_total < nperseg * DECIM:
        print("数据过短，无法做 CAF")
        return None
    ok, n0, n1 = check_iq_magnitude_abort_memmap(ch0_memmap, ch1_memmap, n_total)
    if not skip_magnitude_check and not ok:
        print(f"警告: 幅度超出 |IQ|>{MAGNITUDE_ABORT_THRESHOLD} 的点数超过阈值 "
              f"(ch0={n0}, ch1={n1})，可能存在削波/增益过高/定标错位，结果仅供参考。"
              f"仍继续计算并出图，请自行核实。")
    f_axis, t_axis, Sxx_db, Sxx_complex, eff_fs, delay_max, pdp_amp, pdp_delay_axis = fast_caf_spectrogram(
        ch0_memmap, ch1_memmap, n_total, sample_rate_hz, Tw=Tw, step=step, freq_range=freq_range, use_pdp=use_pdp)
    if use_pdp:
        print(f"PDP delay = {delay_max} (范围 {PDP_DELAY_NEG}..+{PDP_DELAY_POS})")
    print(f"CAF 形状: {Sxx_db.shape}, 有效采样率 = {eff_fs/1e6:.2f} MHz")

    out_dir = "validate_caf_output"
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(experiment_folder)

    if use_pdp and pdp_amp is not None and pdp_delay_axis is not None and np.any(pdp_amp > 0):
        pdp_db = 10 * np.log10(pdp_amp + 1e-12)
        fig_pdp, ax_pdp = plt.subplots(figsize=(8, 5))
        ax_pdp.bar(pdp_delay_axis, pdp_db, color="steelblue", edgecolor="navy", alpha=0.8, width=0.7)
        ax_pdp.axvline(delay_max, color="red", linestyle="--", linewidth=1.5, label=f"delay={delay_max}")
        ax_pdp.set_xlabel("Delay (bin)")
        ax_pdp.set_ylabel("PDP amplitude (dB)")
        ax_pdp.set_title(f"PDP — {base_name}")
        ax_pdp.legend()
        ax_pdp.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pdp_{base_name}.png"), bbox_inches="tight")
        plt.show()
        plt.close(fig_pdp)

    vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 95)

    if show_phase:
        use_phase_derotation = True
        if use_phase_derotation:
            phase_derotation = np.exp(-1j * 2 * np.pi * f_axis.reshape(-1, 1) * t_axis.reshape(1, -1))
            Sxx_derot = Sxx_complex * phase_derotation
        else:
            Sxx_derot = Sxx_complex
        phase_plot_deg = np.rad2deg(np.angle(Sxx_derot))

        pos_mask = f_axis > 0
        neg_mask = f_axis < 0
        zero_mask = np.abs(f_axis) < 1e-6
        has_fused = False
        if np.any(pos_mask) and np.any(neg_mask):
            min_len = min(np.sum(pos_mask), np.sum(neg_mask))
            freqs_pos = f_axis[pos_mask][:min_len]
            freqs_neg = f_axis[neg_mask][:min_len]
            S_pos = Sxx_derot[pos_mask, :][:min_len, :]
            S_neg = Sxx_derot[neg_mask, :][:min_len, :]
            S_neg_flipped = np.flipud(S_neg)
            S_fused = S_pos * S_neg_flipped
            phase_fused_deg = np.rad2deg(np.angle(S_fused))
            if np.any(zero_mask):
                S_0 = Sxx_derot[zero_mask, :][:1, :]
                phase_fused_deg = np.vstack([np.rad2deg(np.angle(S_0)), phase_fused_deg])
                freq_axis_folded = np.concatenate([[0.0], freqs_pos])
            else:
                freq_axis_folded = freqs_pos
            has_fused = True

        n_rows = 3 if has_fused else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
        ax_amp, ax_phase = axes[0], axes[1]
        mesh_amp = ax_amp.pcolormesh(t_axis, f_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax_amp.set_ylabel("Doppler Frequency (Hz)")
        ax_amp.set_title(f"CAF amplitude — {base_name} (Tw={Tw}s, step={step}s, delay={delay_max})")
        fig.colorbar(mesh_amp, ax=ax_amp, label="Relative Power (dB)")
        mesh_phase = ax_phase.pcolormesh(t_axis, f_axis, phase_plot_deg, shading="auto", cmap="hsv", vmin=-180, vmax=180)
        ax_phase.set_ylabel("Doppler Frequency (Hz)")
        ax_phase.set_xlabel("Time (s)" if not has_fused else None)
        ax_phase.set_title("Doppler phase")
        fig.colorbar(mesh_phase, ax=ax_phase, label="Phase (deg)")
        if has_fused:
            ax_fused = axes[2]
            mesh_fused = ax_fused.pcolormesh(t_axis, freq_axis_folded, phase_fused_deg, shading="auto", cmap="hsv", vmin=-180, vmax=180)
            ax_fused.set_ylabel("Doppler Frequency (Hz)")
            ax_fused.set_xlabel("Time (s)")
            ax_fused.set_title("Doppler phase (folded: 0 alone, +f x -f)")
            fig.colorbar(mesh_fused, ax=ax_fused, label="Phase (deg)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"caf_amp_phase_{base_name}.png")
    else:
        fig, ax_amp = plt.subplots(figsize=(12, 6))
        mesh_amp = ax_amp.pcolormesh(t_axis, f_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax_amp.set_ylabel("Doppler Frequency (Hz)")
        ax_amp.set_xlabel("Time (s)")
        ax_amp.set_title(f"CAF amplitude — {base_name} (Tw={Tw}s, step={step}s, delay={delay_max})")
        fig.colorbar(mesh_amp, ax=ax_amp, label="Relative Power (dB)")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"caf_{base_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"谱图已保存: {out_path}")
    plt.show()
    plt.close(fig)
    return f_axis, t_axis, Sxx_db


def run_validation_gif(
    experiment_folder,
    delay_min=DELAY_MIN_GIF,
    delay_max=DELAY_MAX_GIF,
    Tw=TW_SEC,
    step=STEP_SEC,
    freq_range=(-600, 600),
    skip_magnitude_check=False,
    max_duration_sec=10.0,
    gif_frame_duration_sec=GIF_FRAME_DURATION_SEC,
    first_bin_max_m=None,
    step_m=None,
    out_dir="validate_caf_output",
    gif_name=None,
):
    """
    多 delay（delay_min 到 delay_max，默认 -3..+3）各出一张 CAF 幅度谱图，每张显眼处标 delay 对应距离，
    存成临时 PNG 后立刻关图释内存；最后再读这些图拼成 GIF，拼完删临时文件。
    默认只读前 10 秒数据，避免大文件加载过久卡死（可传 max_duration_sec=30 等）。
    """
    print("GIF: 开始 run_validation_gif...", flush=True)
    try:
        from PIL import Image
    except ImportError:
        print("需要 Pillow (PIL) 才能导出 GIF，请安装: pip install Pillow")
        return None
    if experiment_folder is None:
        experiment_folder = find_latest_experiment_folder()
    if experiment_folder is None:
        print("未找到任何 experiment_* 文件夹")
        return None
    if not os.path.isdir(experiment_folder):
        print(f"文件夹不存在: {experiment_folder}")
        return None
    data_file = os.path.join(experiment_folder, "2ch_iq_data.bin")
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return None
    params = load_acquisition_parameters(experiment_folder)
    if not check_data_format(params, data_file):
        return None
    sample_rate_hz = params["sample_rate_hz"] if params and "sample_rate_hz" in params else 20e6
    if step_m is None or first_bin_max_m is None:
        auto_first_bin_max_m, auto_step_m = range_step_for_sample_rate(sample_rate_hz)
        if step_m is None:
            step_m = auto_step_m
        if first_bin_max_m is None:
            first_bin_max_m = auto_first_bin_max_m
    print(f"GIF: 打开 memmap {data_file} ...", flush=True)
    ch0_memmap, ch1_memmap, total_per_ch = read_2ch_iq_memmap(data_file)
    if ch0_memmap is None:
        print("内存映射创建失败")
        return None
    n_total = min(total_per_ch, int(sample_rate_hz * max_duration_sec))
    if n_total < 2:
        return None
    n_mb_full = n_total * 8 * 2 / (1024 * 1024)  # 整段转 complex64 会占用的大小（两通道）——之前就是卡在这里
    n_mb_ds = n_mb_full / DECIM
    print(f"GIF: {n_total} 样点/通道，按 decim={DECIM} 跳采样逐 delay 处理 "
          f"(每次峰值内存约 {n_mb_ds:.0f} MB，而不是整段 {n_mb_full:.0f} MB)...", flush=True)
    effective_fs = sample_rate_hz / DECIM
    nperseg = int(effective_fs * Tw)
    if n_total < nperseg * DECIM:
        print("数据过短，无法做 CAF")
        return None
    ok, n0, n1 = check_iq_magnitude_abort_memmap(ch0_memmap, ch1_memmap, n_total)
    if not skip_magnitude_check and not ok:
        print(f"警告: 幅度超出 |IQ|>{MAGNITUDE_ABORT_THRESHOLD} 的点数超过阈值 "
              f"(ch0={n0}, ch1={n1})，可能存在削波/增益过高/定标错位，结果仅供参考。"
              f"仍继续计算并出图，请自行核实。")

    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(experiment_folder)
    vmin, vmax = None, None
    temp_pngs = []

    for d in range(delay_min, delay_max + 1):
        f_axis, t_axis, Sxx_db = fast_caf_spectrogram_at_delay(
            ch0_memmap, ch1_memmap, d, n_total, sample_rate_hz, Tw=Tw, step=step, decim=DECIM, freq_range=freq_range
        )
        if vmin is None:
            vmin, vmax = np.percentile(Sxx_db, 5), np.percentile(Sxx_db, 95)
        r_min, r_max = range_label_for_delay(d, first_bin_max_m=first_bin_max_m, step_m=step_m)
        fig, ax = plt.subplots(figsize=(12, 6))
        mesh = ax.pcolormesh(t_axis, f_axis, Sxx_db, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
        ax.set_ylabel("Doppler Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"CAF amplitude — {base_name}  |  delay={d}")
        text = f"Range: {r_min:.1f} - {r_max:.1f} m"
        ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=16, fontweight="bold",
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9))
        plt.colorbar(mesh, ax=ax, label="Relative Power (dB)")
        plt.tight_layout()
        tmp_path = os.path.join(out_dir, f"_gif_frame_d{d}_{base_name}.png")
        plt.savefig(tmp_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        temp_pngs.append(tmp_path)

    if not temp_pngs:
        print("没有可用的帧")
        return None
    frames_pil = [Image.open(p).convert("RGB") for p in temp_pngs]
    out_name = gif_name or f"caf_delay{delay_min}_to_{delay_max}_{base_name}.gif"
    out_path = os.path.join(out_dir, out_name)
    frames_pil[0].save(
        out_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=int(gif_frame_duration_sec * 1000),
        loop=0,
    )
    for p in temp_pngs:
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"GIF 已保存: {out_path}（{len(frames_pil)} 帧，每帧 {gif_frame_duration_sec}s）")
    return out_path


def main():
    print("=== validate_capture_and_caf_fc32test 启动（用旧 fc32 数据回归验证 CAF 逻辑）===", flush=True)
    PDP_ON = True
    PHASE_ON = True
    run_gif = True
    # 不采集、不找最新文件夹：写死用户手头那份旧 fc32 数据
    # (experiment_30MHz_static_20260227_150212, data_format=interleaved_complex64)。
    experiment_folder = "experiment_30MHz_static_20260227_150212"
    max_duration_sec = 10.0  # 该实验实际约 19s 数据，这里只取前 10s 够验证逻辑
    print(f"使用实验文件夹: {experiment_folder}", flush=True)
    if run_gif:
        run_validation_gif(
            experiment_folder,
            delay_min=DELAY_MIN_GIF,
            delay_max=DELAY_MAX_GIF,
            Tw=TW_SEC,
            step=STEP_SEC,
            freq_range=(-600, 600),
            skip_magnitude_check=False,
            max_duration_sec=max_duration_sec,
            gif_frame_duration_sec=GIF_FRAME_DURATION_SEC,
        )
    else:
        run_validation(experiment_folder, Tw=TW_SEC, step=STEP_SEC, freq_range=(-600, 600), skip_magnitude_check=False,
                       max_duration_sec=max_duration_sec, use_pdp=PDP_ON, show_phase=PHASE_ON)


if __name__ == "__main__":
    main()
