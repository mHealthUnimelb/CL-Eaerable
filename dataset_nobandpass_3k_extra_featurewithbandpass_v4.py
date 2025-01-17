import os
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import chirp, correlate, butter, lfilter
from scipy.signal.windows import hamming
import json
import logging
import soundfile as sf
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_chirp(start_freq, end_freq, spl_db, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    calibrated_spl_db = spl_db + 7.5
    amplitude = 10 ** ((calibrated_spl_db - 94) / 20)
    return amplitude * chirp(t, start_freq, duration, end_freq, method='linear', phi=0)

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def find_start_of_mark_tone(data, mark_tone, sample_rate, start_time=0):
    search_start = int(start_time * sample_rate)
    search_end = min(len(data), search_start + 2 * sample_rate)
    corr = correlate(data[search_start:search_end], mark_tone, mode='valid')
    offset = (np.argmax(corr) - len(mark_tone))
    return start_time + offset / sample_rate

def extract_tone_from_recording(data, start_time, end_time, sample_rate=44100, frequency=None, bandwidth=200):
    start_sample = round(start_time * sample_rate)
    end_sample = round(end_time * sample_rate)
    segment = data[start_sample:end_sample]
    
    # 只保留后 0.8 秒
    samples_to_keep = int(0.8 * sample_rate)
    segment = segment[-samples_to_keep:]
    
    # 如果指定了 frequency，就在这段上再加带通滤波
    if frequency is not None:
        lowcut = frequency - bandwidth/2
        highcut = frequency + bandwidth/2
        segment = apply_bandpass_filter(segment, lowcut, highcut, sample_rate)
    
    return segment

def get_source_tone(source_audio_filename, spl_db0=50):
    sample_rate, data = read(source_audio_filename)
    mark_tone_up = generate_chirp(100, 20000, spl_db0, 0.1, sample_rate)
    mark_tone_down = generate_chirp(100, 20000, spl_db0, 0.1, sample_rate)[::-1]
    mark_tone = np.concatenate([mark_tone_up, mark_tone_down])

    start_of_mark_tone = find_start_of_mark_tone(data, mark_tone, sample_rate)

    # 分别取 3kHz, 2kHz, 1kHz 三段
    tone_3k = extract_tone_from_recording(data, start_of_mark_tone + 0.6,  start_of_mark_tone + 1.4,  sample_rate)
    tone_2k = extract_tone_from_recording(data, start_of_mark_tone + 10.6, start_of_mark_tone + 11.4, sample_rate)
    tone_1k = extract_tone_from_recording(data, start_of_mark_tone + 20.6, start_of_mark_tone + 21.4, sample_rate)
    
    return tone_3k, tone_2k, tone_1k, start_of_mark_tone, mark_tone

def compute_response_for_tone(original, recorded, sr, f_tone):
    """
    计算目标频率处的响应幅值，并归一化处理。
    :param original: 原始信号
    :param recorded: 录制信号
    :param sr: 采样率
    :param f_tone: 目标频率
    :param baseline_energy: 基线能量值（用于归一化）
    :return: 相对能量特征或当前能量特征
    """
    if len(original) != len(recorded):
        return 0

    # 带通滤波参数
    bandwidth = 200  # Hz
    lowcut = f_tone - bandwidth/2
    highcut = f_tone + bandwidth/2

    # 对信号进行带通滤波
    filtered_original = apply_bandpass_filter(original, lowcut, highcut, sr)
    filtered_recorded = apply_bandpass_filter(recorded, lowcut, highcut, sr)

    segment_length = 0.2  # 秒
    segment_samples = int(segment_length * sr)
    num_segments = len(filtered_original) // segment_samples
    magnitudes = []

    for i in range(num_segments):
        # 提取每段信号
        segment_original = filtered_original[i * segment_samples:(i + 1) * segment_samples]
        segment_recorded = filtered_recorded[i * segment_samples:(i + 1) * segment_samples]
        
        # 对每段进行FFT
        window = hamming(len(segment_original))  # 添加窗函数减少频谱泄漏
        S = np.fft.fft(segment_original * window)
        R = np.fft.fft(segment_recorded * window)
        
        freqs = np.fft.fftfreq(len(S), 1/sr)
        positive_freq_mask = freqs >= 0
        freqs = freqs[positive_freq_mask]
        S = S[positive_freq_mask]
        R = R[positive_freq_mask]
        
        # 计算频率响应
        H = np.where(np.abs(S) > 1e-10, R / S, 0)
        
        # 提取目标频率处的幅值
        index = np.argmin(np.abs(freqs - f_tone))
        magnitude_at_tone = np.abs(H[index])
        magnitudes.append(magnitude_at_tone)
    
    magnitude = np.mean(magnitudes)
    return magnitude

def process_file(
    file_path,
    source_audio_filename,
    case_number,
    last_num,
    num_samples,
    baseline_dict,
    baseline_ready,
    output_dir
):
    """
    返回每个样本的 (freq, magnitude) 以便后续做 baseline 差值或直接存。
    同时，会把 .wav 音频文件也写到 output_dir。
    """
    sample_rate, audio_data = read(file_path)
    tone_3k, tone_2k, tone_1k, start_of_mark_tone, mark_tone = get_source_tone(source_audio_filename)
    
    # case1: [3000]*6 + [2000]*6 + [1000]*6 => 共18次
    # case2 / 3 / 4 => dict mapping
    cases = {
        1: [3000]*6 + [2000]*6 + [1000]*6,
        2: {1: 3000, 2: 3000, 3: 2000, 4: 2000, 5: 1000, 6: 1000},
        3: {1: 3000, 2: 2000, 3: 1000, 4: 3000, 5: 2000, 6: 1000,
            7: 3000, 8: 2000, 9: 1000, 10: 3000, 11: 2000, 12: 1000},
        4: {1: 3000, 2: 2000, 3: 1000, 4: 3000, 5: 2000, 6: 1000,
            7: 2000, 8: 2000, 9: 1000, 10: 3000, 11: 2000, 12: 1000}
    }

    freq_magnitude_list = []  # 用来收集当前 .wav 文件中计算到的 (freq, mag)

    if case_number == 1:
        freqs = cases[1]
        start_of_mt = 0
        for i, freq in enumerate(freqs, start=1):
            if i % 6 == 1:
                start_of_mt = find_start_of_mark_tone(audio_data, mark_tone, sample_rate, start_of_mt)

            base_offset = 0 if freq == 3000 else 10 if freq == 2000 else 20
            start_time = start_of_mt + base_offset + ((i-1)%6)*1.5 + 0.6
            end_time   = start_time + 1.0

            # 不带滤波的原段（写 .wav）
            segment_no_filter = extract_tone_from_recording(audio_data, start_time, end_time, frequency=None, sample_rate=sample_rate)
            segment_filter = extract_tone_from_recording(audio_data, start_time, end_time, frequency=freq, sample_rate=sample_rate)
            
            # 获取对应的原始音频
            original_tone = generate_chirp(freq, freq, spl_db=50, duration=0.8, sample_rate=44100)
            
            magnitude = compute_response_for_tone(original_tone, segment_filter, sample_rate, freq)

            wav_filename = f"sample_{i}_{1 if freq == 3000 else 2 if freq == 2000 else 3}.wav"
            sf.write(os.path.join(output_dir, wav_filename), segment_filter, sample_rate)

            freq_magnitude_list.append((freq, magnitude))

    else:
        freq = cases[case_number][last_num]
        start_of_mt = find_start_of_mark_tone(audio_data, mark_tone, sample_rate)
        for i in range(num_samples):
            start_time = start_of_mt + i*1.5 + 0.6
            end_time   = start_time + 1.0

            segment_no_filter = extract_tone_from_recording(audio_data, start_time, end_time, frequency=None, sample_rate=sample_rate)
            segment_filter = extract_tone_from_recording(audio_data, start_time, end_time, frequency=freq, sample_rate=sample_rate)
            
            # 获取对应的原始音频
            # original_tone = tone_3k if freq == 3000 else tone_2k if freq == 2000 else tone_1k
            original_tone = generate_chirp(freq, freq, spl_db=50, duration=0.8, sample_rate=44100)
            
            magnitude = compute_response_for_tone(original_tone, segment_filter, sample_rate, freq)

            wav_filename = f"sample_{(last_num-1)*6+i+1}_{1 if freq == 3000 else 2 if freq == 2000 else 3}.wav"
            sf.write(os.path.join(output_dir, wav_filename), segment_filter, sample_rate)

            freq_magnitude_list.append((freq, magnitude))

    return freq_magnitude_list

def process_subject_directory(subject_dir, output_dir, num_samples=6):
    """
    在这里我们做两件事：
    1) 先收集 case1 的所有 18 个样本，对应 3k/2k/1k 各 6 次 => 计算 baseline
    2) 对所有样本（包括case1），计算并与 baseline 做差值 -> 存 .npy
    """
    subject_name = os.path.basename(subject_dir)
    logging.info(f"处理受试者: {subject_name}")

    # 这个 dict 用来存 {freq: [所有 case1 下该freq的magnitudes]}
    case1_mags = {3000: [], 2000: [], 1000: []}
    baseline_dict = {}

    # 缓存所有文件的 (case_number, last_num, freq_mags)
    # freq_mags 再在最后统一写入 .npy（包括 baseline 差值）
    all_results = []

    # 先把 .wav 文件排序
    filenames = sorted(os.listdir(subject_dir))

    # 找一下 case1_1.wav 在同目录下，用于 get_source_tone
    source_audio_path = os.path.join(subject_dir, 'case1_1.wav')
    if not os.path.isfile(source_audio_path):
        logging.warning(f"目录 {subject_dir} 下找不到 case1_1.wav，后续可能报错！")

    # 首先保证输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    sample_count = 0

    # ======== 第一遍：先把所有文件都算出 magnitudes，存到 all_results ========
    for filename in filenames:
        if not filename.endswith('.wav'):
            continue
        
        # 解析文件名，如 "case1_1.wav" -> case_number=1, last_num=1
        try:
            case_number = int(filename.split('_')[0][4])  # e.g. "case1" -> '1'
            last_num = int(filename.split('_')[1].split('.')[0])
        except ValueError:
            logging.warning(f"跳过非法文件名: {filename}")
            continue

        if case_number not in [1, 2, 3, 4]:
            logging.info(f"跳过非 case1-4 的文件: {filename}")
            continue

        file_path = os.path.join(subject_dir, filename)
        file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        os.makedirs(file_output_dir, exist_ok=True)

        freq_mags = process_file(
            file_path=file_path,
            source_audio_filename=source_audio_path,
            case_number=case_number,
            last_num=last_num,
            num_samples=num_samples,
            baseline_dict=baseline_dict,
            baseline_ready=False,
            output_dir=file_output_dir
        )
        all_results.append((filename, case_number, last_num, freq_mags, file_output_dir))

        if case_number == 1:
            # case1 一共 18 个样本 => 6个3k, 6个2k, 6个1k
            for (freq, mag) in freq_mags:
                if freq in case1_mags:
                    case1_mags[freq].append(mag)
                sample_count += len(freq_mags)
        else:
            # case2,3,4
            sample_count += len(freq_mags)

    # ======== 第二步：计算 Baseline (case1) ========
    for f_ in [3000, 2000, 1000]:
        values = case1_mags[f_]
        if len(values) == 0:
            baseline_dict[f_] = 0
            logging.warning(f"受试者 {subject_name} 在 freq={f_} 没有 case1 样本，baseline=0!")
        else:
            baseline_dict[f_] = np.mean(values)
    logging.info(f"受试者 {subject_name} baseline => {baseline_dict}")

    # ======== 第三步：存储所有样本数据 ========
    for (filename, case_number, last_num, freq_mags, file_output_dir) in all_results:
        # 所有case都计算与baseline的差值
        for idx, (freq, mag) in enumerate(freq_mags, start=1):
            baseline_val = baseline_dict.get(freq, 0.0)
            diff = abs(mag - baseline_val)
            suffix = 1 if freq == 3000 else 2 if freq == 2000 else 3
            
            if case_number == 1:
                npy_filename = f"sample_{idx}_{suffix}.npy"
            else:
                npy_filename = f"sample_{(last_num-1)*6+idx}_{suffix}.npy"
                
            out_path = os.path.join(file_output_dir, npy_filename)
            np.save(out_path, diff)  # 所有case都存储差值
    logging.info(f"清理1kHz相关文件...")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            # 检查文件名是否以 "_3.wav" 或 "_3.npy" 结尾（对应1kHz的文件）
            if file.endswith("_3.wav") or file.endswith("_3.npy") or file.endswith("_2.wav") or file.endswith("_2.npy"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logging.debug(f"已删除: {file_path}")
                except Exception as e:
                    logging.error(f"删除文件失败 {file_path}: {str(e)}")

    return sample_count

def process_all_participants(root_dir, output_root_dir, num_samples=6):
    total_sample_count = 0
    for participant in os.listdir(root_dir):
        participant_dir = os.path.join(root_dir, participant)
        if os.path.isdir(participant_dir):
            logging.info(f"Processing participant: {participant}")
            participant_output_dir = os.path.join(output_root_dir, participant)
            os.makedirs(participant_output_dir, exist_ok=True)

            participant_sample_count = process_subject_directory(participant_dir, participant_output_dir, num_samples)
            total_sample_count += participant_sample_count

            logging.info(f"参与者 {participant} 的样本数: {participant_sample_count}")
        logging.info("")

    logging.info("所有处理完成，开始清理空文件夹...")
    # 清理可能产生的空文件夹
    for root, dirs, files in os.walk(output_root_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # 如果文件夹为空
                    os.rmdir(dir_path)
                    logging.debug(f"已删除空文件夹: {dir_path}")
            except Exception as e:
                logging.error(f"删除文件夹失败 {dir_path}: {str(e)}")

    return total_sample_count


if __name__ == "__main__":
    root_directory = './Recordings'
    output_root_directory = './dataset_3k_withbandpass_extrafeatures_v3'
    total_samples = process_all_participants(root_directory, output_root_directory, num_samples=6)
    
    logging.info(f"处理完成。总样本数: {total_samples}")
