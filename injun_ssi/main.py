import streamlit as st 
import os
import sys
import datetime
import copy
from st_audiorec import st_audiorec
#import nemo.collections.asr as nemo_asr
st.set_page_config(
    page_title="Bul4jo",
    page_icon=":white_check_mark:",
    layout="wide"    
    )
tab1,tab2 = st.tabs(['Tab A','Tab B'])
#asr_model = nemo_asr.models.ASRModel.from_pretrained("eesungkim/stt_kr_conformer_transducer_large")

#convert_sample_rate('/content/fisu.wav', '/content/output_16.wav')

with tab1 :
  # tab1  에 담을 내용
  working_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(working_dir)
  st.title(working_dir)
with tab2 :
  # tab2 에 담을 내용
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        st.title('here is column2')
        st.checkbox('this is checkbox1 in col2 ')





# streamlit run main.py
