import os
import librosa
import soundfile as sf

def resample_wavs(input_dir, output_dir, target_sr=16000):
    """
    重采样目录中的所有WAV文件到指定采样率
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param target_sr: 目标采样率（默认16000Hz）
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的WAV文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 读取音频文件（自动获取原始采样率）
                y, orig_sr = librosa.load(input_path, sr=None)
                
                # 执行重采样
                if orig_sr != target_sr:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
                
                # 保持原始位深度（16-bit）保存文件
                sf.write(output_path, y, target_sr, subtype='PCM_16')
                print(f"成功处理: {filename}")
                
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_folder = r"D:\NBU\副线\语音对抗\zq\data\Target command audio\kokoro"
    output_folder = r"D:\NBU\副线\语音对抗\zq\data\Target command audio\kokoro_16"
    resample_wavs(input_folder, output_folder)