import os
import glob
import librosa
import numpy as np
import soundfile as sf
import sys
import shutil  # 新增用于文件复制

def analyze_audio_range(file_path):
    """分析音频文件的振幅范围"""
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    max_val = np.max(audio)
    min_val = np.min(audio)
    return min_val, max_val

def normalize_audio(input_path, output_path, target_peak=0.5):
    """音频归一化处理核心函数"""
    audio, sr = librosa.load(input_path, sr=None, mono=False)
    
    # 计算绝对峰值和缩放比例
    current_peak = np.max(np.abs(audio))
    scaling_factor = target_peak / current_peak
    
    # 仅缩放不裁剪（保持波形比例）
    normalized_audio = audio * scaling_factor
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if len(normalized_audio.shape) > 1:
        sf.write(output_path, normalized_audio.T, sr)
    else:
        sf.write(output_path, normalized_audio, sr)
    
    return normalized_audio

def process_audio_folder(input_folder, output_folder):
    """智能处理文件夹音频"""
    wav_files = glob.glob(os.path.join(input_folder, '**', '*.wav'), recursive=True)
    processed_count = 0
    skipped_count = 0
    target_peak = 0.5
    tolerance = 1e-4
    
    print(f"\n发现 {len(wav_files)} 个WAV文件")
    
    for i, file_path in enumerate(wav_files, 1):
        try:
            rel_path = os.path.relpath(file_path, input_folder)
            dest_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # 获取原始范围
            min_val, max_val = analyze_audio_range(file_path)
            original_peak = max(abs(min_val), abs(max_val))
            
            print(f"\n[{i}/{len(wav_files)}] 文件: {rel_path}")
            print(f"原始峰值: {original_peak:.4f} | 目标峰值: {target_peak:.4f}")
            print(f"原始范围: [{min_val:.4f}, {max_val:.4f}]")  # 新增原始范围显示
            
            if abs(original_peak - target_peak) < tolerance:
                shutil.copy2(file_path, dest_path)
                print("已跳过（已达目标峰值）")
                skipped_count += 1
            else:
                normalized_audio = normalize_audio(file_path, dest_path)
                new_min = np.min(normalized_audio)
                new_max = np.max(normalized_audio)
                new_peak = max(abs(new_min), abs(new_max))
                print(f"已处理 | 新范围: [{new_min:.4f}, {new_max:.4f}] | 新峰值: {new_peak:.4f}")
                processed_count += 1
                
        except Exception as e:
            print(f"处理失败: {str(e)}")
    
    print(f"\n处理完成！已处理 {processed_count} 个文件，跳过 {skipped_count} 个文件")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        process_audio_folder(sys.argv[1], sys.argv[2])
    else:
        input_folder = r"D:\NBU\副线\语音对抗\zq\data\Target command audio\kokoro_16"
        output_folder = r"D:\NBU\副线\语音对抗\zq\data\Target command audio\kokoro_16_normalized"
        print("使用默认路径进行批处理...")
        process_audio_folder(input_folder, output_folder)