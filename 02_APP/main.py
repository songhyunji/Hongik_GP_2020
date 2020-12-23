import streamlit as st
from PIL import Image

def app():
    st.subheader('Main')

    st.markdown("- 2020 홍익대학교 컴퓨터공학과 졸업프로젝트입니다.")

    st.markdown("- 서비스 이용 방법")
    img = Image.open("APP/Image/Guide.png")
    st.image(img, width=800)

    st.markdown("- 개발자\n"
                "   - B611101 송현지\n"
                "       - 홍익대학교 컴퓨터공학과\n"
                "       - songhj97@kakao.com\n"
                "       - https://github.com/songhyunji\n"
                "   - B635150 박서이\n"
                "       - 홍익대학교 컴퓨터공학과\n"
                "       - tjdl8345@naver.com\n")

    st.markdown("- 프로젝트 Git\n"
                "   - https://github.com/songhyunji/Hongik_GP_2020")