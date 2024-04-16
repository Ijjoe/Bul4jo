import re
import streamlit as st
import streamlit_shadcn_ui as ui

st.write("Hello world")
st.image('./img/qrcode.jpg', caption='주소')
st.image('./img/techstack.jpg', caption='기술스택')




with ui.card(key="card1"):
    ui.element("span", children=["Email"], className="text-gray-400 text-sm font-medium m-1", key="label1")
    ui.element("input", key="email_input", placeholder="Your email")

    ui.element("span", children=["User Name"], className="text-gray-400 text-sm font-medium m-1", key="label2")
    ui.element("input", key="username_input", placeholder="Create a User Name")
    ui.element("button", text="Submit", key="button", className="m-1")