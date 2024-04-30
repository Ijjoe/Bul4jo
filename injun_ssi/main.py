import streamlit as st 
import os
import sys
import librosa
from IPython.display import Audio
import numpy as np
import random
from librosa import resample
from streamlit.components.v1 import html
from streamlit_extras.buy_me_a_coffee import button
from st_audiorec import st_audiorec
from dotenv import load_dotenv
from transformers import pipeline
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor
#import nemo.collections.asr as nemo_asr
# .env 파일에서 환경 변수 로드
load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



st.set_page_config(
    page_title="Bul4jo",
    page_icon=":white_check_mark:",
    layout="wide"    
    )

# 환경 변수에서 Hugging Face API 키 가져오기
api_key = os.getenv("HUGGINGFACE_API_KEY")
#st.title(api_key)
model_name="Ljrabbit/wav2vec2-large-xls-r-300m-korean-test0"
#pipe=pipeline(model_name="Ljrabbit/wav2vec2-large-xls-r-300m-korean-test0", token=api_key)
pro=Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)


#speech,sample_rate=sf.read('C:\AI_gitRep\Bul4jo\injun_ssi\샘플_1.wav')


# 오디오 파일 읽기 및 리샘플링
speech,sample_rate = librosa.load('C:\AI_gitRep\Bul4jo\injun_ssi\샘플_1.wav', sr=16000)  # y는 오디오 시그널, sr은 샘플링 레이트




#speech,_=sf.read('C:\CIJ\cijGit\Bul4jo\injun_ssi\coro16.wav')
#ipd.Audio(data=np.asarry(speech), autoplay=True,rate=16000)

##---------------------------------------------------------------------------------
inputs = pro(speech,sampling_rate=sample_rate, return_tensors='pt' , padding='longest').to(device)



    # retrieve logits & take argmax
with torch.no_grad():    
    outputs = model(inputs.input_values)
    predicted_ids = outputs.logits.argmax(-1)
    #transcription = pro.tokenizer.batch_decode(predicted_ids[0])
    transcription = pro.decode(predicted_ids[0], skip_special_tokens=True)




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

st.image('C:\AI_gitRep\Bul4jo\injun_ssi\cbmc_qr.png', caption='거! 커피한잔은 괜찮잖아~~',width=180)
#st.image('C:\\CIJ\\cijGit\\Bul4jo\\injun_ssi\\bmc_qr.png', caption='거! 커피한잔은 괜찮잖아~~',width=180)

with tab1 :
  # tab1 에 담을 내용
    
    st.write(transcription)
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        st.title('here is column2')
        st.checkbox('this is checkbox1 in col2 ')
      

with tab2 :
  # tab2  에 담을 내용
  st.write(sample_rate)
  working_dir = os.path.dirname(os.path.abspath(__file__))
  sys.path.append(working_dir)
  st.title(working_dir)
  
  




# streamlit run main.py
