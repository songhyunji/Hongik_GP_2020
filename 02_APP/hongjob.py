import main
import draft_v1
import draft_v2
import more
import streamlit as st
from PIL import Image

PAGES = {
    "MAIN": main,
    "Draft ver.1": draft_v1,
    "Draft ver.2": draft_v2,
    "MORE": more
}
img = Image.open("APP/Image/icon.png")
st.set_page_config(page_title="HONGJOB", page_icon=img, layout="centered")

# Title
st.title('HONGJOB')

# Header/Subheader
st.header('Self-Introdection Draft Service')

st.sidebar.title('MENU')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()