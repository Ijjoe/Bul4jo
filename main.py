import re
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def example(ttx):
    add_n_lines = st.slider("Add n vertical lines below this", 1, 20, 5)
    add_vertical_space(add_n_lines)
    st.write(ttx)



example("Test bear")    
print("<script data-name="BMC-Widget" data-cfasync="false" src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" data-id="bul4jo" data-description="Support me on Buy me a coffee!" data-message="지원 감사합니다. 들숨에 재력을 날숨에 건강을" data-color="#40DCA5" data-position="Right" data-x_margin="18" data-y_margin="18"></script>")
st.write("불사조 2차 미니 프로젝트")
st.image('./img/qrcode.jpg', caption='주소')
st.image('./img/techstack.jpg', caption='기술스택')



