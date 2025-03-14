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

