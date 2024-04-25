import streamlit as st 
from streamlit_mic_recorder import mic_recorder,speech_to_text
import os
from audio_recorder_streamlit import audio_recorder
import sys
import datetime
import nemo.collections.asr as nemo_asr
import copy
import librosa
from st_audiorec import st_audiorec
import soundfile as sf

asr_model = nemo_asr.models.ASRModel.from_pretrained("eesungkim/stt_kr_conformer_transducer_large")


def convert_sample_rate(input_file, output_file, target_sr=16000):
    # 오디오 파일을 원본 샘플 레이트로 로드합니다.
    audio, sr = librosa.load(input_file, sr=None)
    # 샘플 레이트를 16000Hz로 변경합니다.
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # 변경된 샘플 레이트의 오디오 파일을 저장합니다.
    sf.write(output_file, audio_resampled, target_sr)

# 사용 예
convert_sample_rate('/content/fisu.wav', '/content/output_16.wav')


working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')



# streamlit run main.py
