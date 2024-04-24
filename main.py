import re
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def example(ttx):
    add_n_lines = st.slider("Add n vertical lines below this", 1, 20, 5)
    add_vertical_space(add_n_lines)
    st.write(ttx)



example("Test bear")    
print("Test")
st.write("불사조 2차 미니 프로젝트")
st.image('./img/qrcode.jpg', caption='주소')
st.image('./img/techstack.jpg', caption='기술스택')



