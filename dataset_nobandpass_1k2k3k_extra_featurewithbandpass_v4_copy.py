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

def extract_tone_from_recording(data, start_time, end_time, sample_rate=44100):
    start_sample = round(start_time * sample_rate)
    end_sample = round(end_time * sample_rate)
    full_segment = data[start_sample:end_sample]
    
    # 计算后0.8秒的起始点
    start_of_last_08s = len(full_segment) - int(0.8 * sample_rate)
    return full_segment[start_of_last_08s:]

def find_start_of_mark_tone(data, mark_tone, sample_rate, start_time=0):
    search_start = int(start_time * sample_rate)
    search_end = min(len(data), search_start + 2*sample_rate)
    corr = correlate(data[search_start:search_end], mark_tone, mode='valid')
    return start_time + (np.argmax(corr) - len(mark_tone)) / sample_rate

def get_source_tone(source_audio_filename, spl_db0=50):
    sample_rate, data = read(source_audio_filename)

    mark_tone_up = generate_chirp(100, 20000, spl_db0, 0.1, sample_rate)
    mark_tone_down = generate_chirp(100, 20000, spl_db0, 0.1, sample_rate)[::-1]
    mark_tone = np.concatenate([mark_tone_up, mark_tone_down])

    start_of_mark_tone = find_start_of_mark_tone(data, mark_tone, sample_rate)

    tone_3k = extract_tone_from_recording(data, start_of_mark_tone + 0.6, start_of_mark_tone + 1.4, sample_rate)
    tone_2k = extract_tone_from_recording(data, start_of_mark_tone + 10.6, start_of_mark_tone + 11.4, sample_rate)
    tone_1k = extract_tone_from_recording(data, start_of_mark_tone + 20.6, start_of_mark_tone + 21.4, sample_rate)
    
    return tone_3k, tone_2k, tone_1k, start_of_mark_tone, mark_tone

def compute_response_for_tone(original, recorded, sr, f_tone):
    if len(original) != len(recorded):
        return 0, 0

    S = np.fft.fft(original)
    R = np.fft.fft(recorded)
    
    freqs = np.fft.fftfreq(len(S), 1/sr)
    
    positive_freq_mask = freqs >= 0
    freqs = freqs[positive_freq_mask]
    S = S[positive_freq_mask]
    R = R[positive_freq_mask]
    
    H = np.where(np.abs(S) > 1e-10, R / S, 0)
    
    index = int(f_tone * len(original) / sr)
    
    magnitude = np.abs(H[index])
    
    return magnitude


def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def compute_response_for_tone(recorded, sr, f_tone):
    """
    :param original: 原始音频（0.8s）
    :param recorded: 录制音频（0.8s）
    :param sr: 采样率
    :param f_tone: 目标频率 fc (3000Hz, 2000Hz, 或 1000Hz)
    :return: float类型的能量特征
    """
    # 带通滤波
    bandwidth = 200  # Hz
    lowcut = f_tone - bandwidth/2
    highcut = f_tone + bandwidth/2
    filtered_signal = apply_bandpass_filter(recorded, lowcut, highcut, sr)
    
    # 分段处理（20ms窗口）
    window_duration = 0.02  # 20ms
    window_samples = int(window_duration * sr)
    num_windows = len(filtered_signal) // window_samples
    
    # 存储每个窗口的FFT结果
    magnitudes = []
    
    for i in range(num_windows):
        # 提取窗口数据
        start = i * window_samples
        end = start + window_samples
        window_data = filtered_signal[start:end]
        
        # 应用汉明窗
        window = hamming(len(window_data))
        windowed_data = window_data * window
        
        # 计算FFT
        fft_result = np.fft.fft(windowed_data)
        freqs = np.fft.fftfreq(len(fft_result), 1/sr)
        
        # 找到目标频率对应的索引
        target_idx = np.argmin(np.abs(freqs - f_tone))
        
        # 计算该窗口的能量
        magnitude = float(np.abs(fft_result[target_idx]) ** 2)
        magnitudes.append(magnitude)
    
    # 计算平均能量 M²ᵢ,ⱼ
    M_squared = float(np.mean(magnitudes))
    
    return M_squared

def get_baseline(subject_dir):
    """
    获取指定目录下的baseline值
    :param subject_dir: 受试者目录路径（如 'A1' 或 'A2'）
    :return: 包含3个频率baseline值的字典，如果获取失败则相应频率值为0
    """
    # print(subject_dir)
    # 初始化 baseline 字典
    baseline_dict = {
        3000: 0.0,
        2000: 0.0,
        1000: 0.0
    }

    # 寻找并处理 case1_1.wav
    source_audio_path = os.path.join(subject_dir, 'case1_1.wav')
    if not os.path.isfile(source_audio_path):
        logging.warning(f"找不到baseline文件: {source_audio_path}")
        return baseline_dict

    try:
        sample_rate, audio_data = read(source_audio_path)
        tone_3k, tone_2k, tone_1k, start_of_mark_tone, mark_tone = get_source_tone(source_audio_path)
        
        # 找到 mark tone 位置
        start_of_mt = find_start_of_mark_tone(audio_data, mark_tone, sample_rate)
        
        # 提取并计算3个频率的baseline值
        # 3kHz - segment 1
        start_time = start_of_mt + 0.6
        segment_3k = extract_tone_from_recording(audio_data, start_time, start_time + 1, sample_rate)
        baseline_dict[3000] = compute_response_for_tone(segment_3k, sample_rate, 3000)
        
        # 2kHz - segment 7
        start_time = start_of_mt + 10.6
        segment_2k = extract_tone_from_recording(audio_data, start_time, start_time + 1, sample_rate)
        baseline_dict[2000] = compute_response_for_tone(segment_2k, sample_rate, 2000)
        
        # 1kHz - segment 13
        start_time = start_of_mt + 20.6
        segment_1k = extract_tone_from_recording(audio_data, start_time, start_time + 1, sample_rate)
        baseline_dict[1000] = compute_response_for_tone(segment_1k, sample_rate, 1000)
        
        # logging.info(f"成功获取baseline值: 3kHz={baseline_dict[3000]:.2f}, 2kHz={baseline_dict[2000]:.2f}, 1kHz={baseline_dict[1000]:.2f}")
        
    except Exception as e:
        logging.error(f"处理baseline文件时出错: {str(e)}")
    
    return baseline_dict

def process_file(file_path, output_dir, case_number, last_num, num_samples=6):
    sample_rate, audio_data = read(file_path)
    source_audio_filename = os.path.join(os.path.dirname(file_path), 'case1_1.wav')
    tone_3k, tone_2k, tone_1k, start_of_mark_tone, mark_tone = get_source_tone(source_audio_filename)

    # Get baseline
    # 从文件路径中获取subject目录
    subject_dir = os.path.dirname(file_path)
    # 获取baseline值
    baseline_dict = get_baseline(subject_dir)

    cases = {
        1: [3000, 2000, 1000],
        2: {1: 3000, 2: 3000, 3: 2000, 4: 2000, 5: 1000, 6: 1000},
        3: {1: 3000, 2: 2000, 3: 1000, 4: 3000, 5: 2000, 6: 1000, 7: 3000, 8: 2000, 9: 1000, 10: 3000, 11: 2000, 12: 1000},
        4: {1: 3000, 2: 2000, 3: 1000, 4: 3000, 5: 2000, 6: 1000, 7: 2000, 8: 2000, 9: 1000, 10: 3000, 11: 2000, 12: 1000}
    }

    if case_number == 1:
        freqs = cases[1] * 6
        start_of_mark_tone = 0
        for i, freq in enumerate(freqs, start=1):
            if i % 6 == 1:
                start_of_mark_tone = find_start_of_mark_tone(audio_data, mark_tone, sample_rate, start_of_mark_tone)
            
            tone = tone_3k if freq == 3000 else tone_2k if freq == 2000 else tone_1k
            start_time = start_of_mark_tone + ((i-1)%6)*1.5 + 0.6
            end_time = start_time + 1.0
            segment = extract_tone_from_recording(audio_data, start_time, end_time, sample_rate)
            magnitude = compute_response_for_tone(segment, sample_rate, freq)
            baseline_diff = abs(magnitude - baseline_dict[freq])

            # 在保存之前应用带通滤波
            bandwidth = 200  # Hz
            lowcut = freq - bandwidth/2
            highcut = freq + bandwidth/2
            filtered_segment = apply_bandpass_filter(segment, lowcut, highcut, sample_rate)

            wav_filename = f"sample_{i}_{1 if freq == 3000 else 2 if freq == 2000 else 3}.wav"
            npy_filename = f"sample_{i}_{1 if freq == 3000 else 2 if freq == 2000 else 3}.npy"
            
            sf.write(os.path.join(output_dir, wav_filename), segment, sample_rate)
            np.save(os.path.join(output_dir, npy_filename), np.array([baseline_diff]))
    else:
        freq = cases[case_number][last_num]
        tone = tone_3k if freq == 3000 else tone_2k if freq == 2000 else tone_1k
        start_of_mark_tone = find_start_of_mark_tone(audio_data, mark_tone, sample_rate)
        for i in range(num_samples):
            start_time = start_of_mark_tone + i*1.5 + 0.6
            end_time = start_time + 1.0
            segment = extract_tone_from_recording(audio_data, start_time, end_time, sample_rate)
            magnitude = compute_response_for_tone(segment, sample_rate, freq)
            baseline_diff = abs(magnitude - baseline_dict[freq])

            # 在保存之前应用带通滤波
            bandwidth = 200  # Hz
            lowcut = freq - bandwidth/2
            highcut = freq + bandwidth/2
            filtered_segment = apply_bandpass_filter(segment, lowcut, highcut, sample_rate)
            
            wav_filename = f"sample_{(last_num-1)*6+i+1}_{1 if freq == 3000 else 2 if freq == 2000 else 3}.wav"
            npy_filename = f"sample_{(last_num-1)*6+i+1}_{1 if freq == 3000 else 2 if freq == 2000 else 3}.npy"
            sf.write(os.path.join(output_dir, wav_filename), segment, sample_rate)
            np.save(os.path.join(output_dir, npy_filename), np.array([baseline_diff]))

def process_subject_directory(subject_dir, output_dir, num_samples=6):
    subject_name = os.path.basename(subject_dir)
    logging.info(f"处理受试者: {subject_name}")
    sample_count = 0

    for filename in sorted(os.listdir(subject_dir)):
        if filename.endswith('.wav'):
            try:
                case_number = int(filename.split('_')[0][4])
            except ValueError:
                logging.warning(f"跳过非法文件名: {filename}")
                continue
            
            if case_number not in [1, 2, 3, 4]:
                logging.info(f"跳过非 case1-4 的文件: {filename}")
                continue
            
            file_path = os.path.join(subject_dir, filename)
            last_num = int(filename.split('_')[1].split('.')[0])
            
            file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            
            process_file(file_path, file_output_dir, case_number, last_num, num_samples)
            
            if case_number == 1:
                sample_count += 18  # case1 有18个样本
            else:
                sample_count += num_samples

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
    
    return total_sample_count

if __name__ == "__main__":
    root_directory = './Recordings'
    output_root_directory = './dataset_1k2k3k_withbandpass_extrafeatures'
    total_samples = process_all_participants(root_directory, output_root_directory, num_samples=6)
    
    logging.info(f"处理完成。总样本数: {total_samples}")    