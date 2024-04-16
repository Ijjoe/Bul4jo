import re
import streamlit as st
import streamlit_shadcn_ui as ui
from streamlit_extras.add_vertical_space import add_vertical_space

def example():
    add_n_lines = st.slider("Add n vertical lines below this", 1, 20, 5)
    add_vertical_space(add_n_lines)
    st.write("Here is text after the nth line!")



example    

st.write("Hello world")
st.image('./img/qrcode.jpg', caption='주소')
st.image('./img/techstack.jpg', caption='기술스택')



