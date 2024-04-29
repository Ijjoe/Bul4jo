#!pip install nemo_toolkit['all']



import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained("eesungkim/stt_kr_conformer_transducer_large")


#재우님이 작업해주실거
#컴퓨터상 녹음 -> wav 파일 -> 16000변환 -> asr_model.transcribe 작동하여 텍스트로 나오는것 까지만 요작업만
#아마 STT 구동시 변환 시간이 오래걸릴것으로 예상 


asr_model.transcribe(['/content/coro.wav'])
