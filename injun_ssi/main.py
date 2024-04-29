import streamlit as st 
import os
import sys
import datetime
import copy
from streamlit.components.v1 import html
from streamlit_extras.buy_me_a_coffee import button
from st_audiorec import st_audiorec
#import nemo.collections.asr as nemo_asr

st.set_page_config(
    page_title="Bul4jo",
    page_icon=":white_check_mark:",
    layout="wide"    
    )
tab1,tab2 = st.tabs(['Tab A','Tab B'])

#convert_sample_rate('/content/fisu.wav', '/content/output_16.wav')
def example():
    button(username="bul4jo", floating=True, width=220 ,text='커피 한 잔' )

example()
st.markdown(
    """
    <style>
  
 .iframeContainer {
  position: relative;
  width: 100%;
}
.iframeContainer iframe {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
}

    </style>
    """,
    unsafe_allow_html=True,
)

st.image('C:\\CIJ\\cijGit\\Bul4jo\\injun_ssi\\bmc_qr.png', caption='거! 커피한잔은 괜찮잖아~~',width=180)

with tab1 :
  # tab1 에 담을 내용
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        st.title('here is column2')
        st.checkbox('this is checkbox1 in col2 ')

with tab2 :
  # tab2  에 담을 내용
  working_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(working_dir)
  st.title(working_dir)
  
  




# streamlit run main.py
