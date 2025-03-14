'''
1.pip uninstall torch torchaudio torchvision
2.pip install torch torchaudio torchvision（防止torch.nn.attention丢失）
3.pip install 'nemo_toolkit[all]'
4.https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr（模型）
'''
import nemo
import nemo.collections.asr as nemo_asr

Conformer_CTC = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_conformer_ctc_small")
Conformer_Transducer = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_conformer_transducer_small")
ContextNet = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_contextnet_256_mls")
Citrinet = nemo_asr.models.ASRModel.from_pretrained(model_name="stt_en_citrinet_256")


def transcribe_audio(audio_path):
    # 指定要使用的音频文件路径（请替换为实际路径）
    audio_file = [audio_path]
    
    print("\nConformer CTC 识别结果：")
    print(Conformer_CTC.transcribe(audio_file))
    
    print("\nConformer Transducer 识别结果：")
    print(Conformer_Transducer.transcribe(audio_file))
    
    print("\nContextNet 识别结果：")
    print(ContextNet.transcribe(audio_file))
    
    print("\nCitrinet 识别结果：")
    print(Citrinet.transcribe(audio_file))

if __name__ == "__main__":
    # 替换为您的音频文件实际路径，例如：D:/test.wav
    audio_path = "your_audio_file.wav"
    transcribe_audio(audio_path)

