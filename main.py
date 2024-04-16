import re
import streamlit as st
import streamlit_shadcn_ui as ui


trigger_btn = ui.button(text="Trigger Button", key="trigger_btn")

ui.alert_dialog(show=trigger_btn, title="Alert Dialog", description="This is an alert dialog", confirm_label="OK", cancel_label="Cancel", key="alert_dialog1")


st.write("Hello world")
st.image('./img/qrcode.jpg', caption='주소')
st.image('./img/techstack.jpg', caption='기술스택')



