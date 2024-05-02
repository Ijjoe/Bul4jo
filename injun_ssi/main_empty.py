import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Streamlit 앱 타이틀 설정
st.title('음성 파일 파형')

# 파일 업로더를 통해 사용자로부터 음성 파일 입력 받기
uploaded_file = st.file_uploader("C:\CIJ\cijGit\Bul4jo\injun_ssi\샘플_1.wav", type=['wav'])

if uploaded_file is not None:
    # WAV 파일 읽기
    rate, data = wavfile.read(uploaded_file)
    # 스테레오 음성 파일 처리
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # 시간 축 계산
    time = np.linspace(0., len(data) / rate, len(data))

    # 파형 그리기
    fig, ax = plt.subplots()
    ax.plot(time, data)
    ax.set(xlabel='Time (s)', ylabel='Amplitude')

    # Streamlit에 그래프 표시
    st.pyplot(fig)
