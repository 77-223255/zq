'''
1.pip uninstall torch torchaudio torchvision
2.pip install torch torchaudio torchvision（防止torch.nn.attention丢失）
3.pip install 'nemo_toolkit[all]'
4.https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr（模型）
5.python 3.11
'''
import nemo
import nemo.collections.asr as nemo_asr
import os
from pathlib import Path
import torchaudio  # 确保torchaudio已安装

# 初始化语音识别模型
Conformer_CTC = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_small")
Conformer_Transducer = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_conformer_transducer_small")
ContextNet = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_contextnet_256")
Citrinet = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_citrinet_256")

def transcribe_audio(audio_path):
    if os.path.isdir(audio_path):
        # 遍历文件夹中的所有wav文件
        for wav_file in Path(audio_path).glob('*.wav'):
            print(f"\n{'='*30}\n处理文件：{wav_file.name}\n{'='*30}")
            audio_file = [str(wav_file)]
            
            print("[Conformer CTC] 识别结果：", Conformer_CTC.transcribe(audio_file)[0].text)
            print("[Conformer Transducer] 识别结果：", Conformer_Transducer.transcribe(audio_file)[0].text)
            print("[ContextNet] 识别结果：", ContextNet.transcribe(audio_file)[0].text)
            print("[Citrinet] 识别结果：", Citrinet.transcribe(audio_file)[0].text)
    else:
        # 处理单个文件
        audio_file = [audio_path]
        print(f"\n{'='*30}\n处理文件：{os.path.basename(audio_path)}\n{'='*30}")
        print("[Conformer CTC] 识别结果：", Conformer_CTC.transcribe(audio_file)[0].text)
        print("[Conformer Transducer] 识别结果：", Conformer_Transducer.transcribe(audio_file)[0].text)
        print("[ContextNet] 识别结果：", ContextNet.transcribe(audio_file)[0].text)
        print("[Citrinet] 识别结果：", Citrinet.transcribe(audio_file)[0].text)

if __name__ == "__main__":
    
    # 确保下载路径存在
    librispeech_path = Path("/home/featurize/temperory_data/LibriSpeech")
    librispeech_path.mkdir(parents=True, exist_ok=True)  # 新增目录创建
    
    # 下载并加载LibriSpeech测试集
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(librispeech_path),
        url="test-clean",
        download=True
    )
    
    # 创建临时目录保存测试音频
    temp_dir = Path("/home/featurize/temperory_data/librispeech_samples")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 选5个样本进行测试
    sample_count = 5
    for i in range(sample_count):
        # 获取音频数据和正确答案
        waveform, sample_rate, utterance, _, _, _ = dataset[i]
        
        # 保存为临时wav文件
        file_path = temp_dir / f"sample_{i}.wav"
        torchaudio.save(file_path, waveform, sample_rate)
        
        # 进行语音识别
        print(f"\n{'='*30}\n样本 {i+1}/{sample_count}\n{'='*30}")
        print(f"正确答案：{utterance}")
        transcribe_audio(str(file_path))  # 复用原有识别逻辑
       
    # 保留原有功能（可选）
    '''
    original_path = r'/home/featurize/data/data/Target command audio/kokoro_16'
    transcribe_audio(original_path)
    '''

