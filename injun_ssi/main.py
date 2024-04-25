import streamlit as st 
from langchain.llms import CTransformers
from streamlit_mic_recorder import mic_recorder,speech_to_text
import os
from audio_recorder_streamlit import audio_recorder
import sys
import datetime


working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)


audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes,format="audio/wav")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #with open(f"audio_{timestamp}.wav", "wb") as f:
        #f.write(audio_bytes)


# audio = mic_recorder(
#     start_prompt="Start recording",
#     stop_prompt="Stop recording",
#     just_once=False,
#     use_container_width=False,
#     format="wav",
#     callback=None,
#     args=(),
#     kwargs={},
#     key=None
# )     
            
# llm =CTransformers(
#    model = "llama-2-7b-chat.ggmlv3.q2_K.bin",
#    model_type ="llama"
# )

# st.title('인공지는 시인')

# con = st.text_input('시의 주제를 제시해주세요')

# if st.button('시 작성 요청하기'):
#   with st.spinner('시 작성 중 ...'):
#     res = llm.predict("write a poem about " + con + ":")
#     st.write(res)
    
# streamlit run main.py
