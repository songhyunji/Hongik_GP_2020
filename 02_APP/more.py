import streamlit as st

def app():
    st.subheader('More')

    st.markdown("1. 본 서비스는 자기소개서 초안 작성 기법 중, SCAR/MVPC 기법을 기초로하여 기획하였습니다.\n"
                "   1. SCAR 기법\n"
                "       - 경험과 역량 관련 질문의 답변 초안에 사용하였습니다. \n"
                "       1. 상황 Situation : 주어진 상횡/배경\n"
                "       2. 위기 Crisis : 발생된 문제/위기\n"
                "       3. 행동 Action : 자신의 행동/태도\n"
                "       4. 결과 Result : 성과/결과/교훈\n"
                "   2. MVPC 기법\n"
                "       - 지원 동기 및 포부 관련 질문의 답변 초안에 사용하였습니다. \n"
                "       1. 동기 Motivation : 기업/직무를 선택한 동기/계기\n"
                "       2. 비전 Vision : 기업과 자신의 비전/목표/계획\n"
                "       3. 열정 Passion : 기업과 직무에 대한 의지/열정\n"
                "       4. 역량 Conmpetence : 자신의 역량/강점\n"
                "   - 출처 : https://getjob.co.kr/scar-%EA%B8%B0%EB%B2%95%EC%9C%BC%EB%A1%9C-%EC%8A%A4%ED%86%A0%EB%A6%AC-%EB%A7%8C%EB%93%A4%EA%B8%B0/\n"
                "2. 위 기법에 따라 입력된 사용자의 경험을 요약할 때,\n"
                "   1. SCAR 기법\n"
                "       - 상황과 결과 문장을 합쳐 요약을 진행했습니다.\n"
                "   2. MVPC 기법\n"
                "       - 열정과 비전 문장을 합쳐 요약을 진행했습니다.\n"
                "3. 다음의 출처를 참고하여 개발하였습니다.\n"
                "   - 문장 분류\n"
                "       - 데이터셋 : 잡코리아 합격자소서 http://www.jobkorea.co.kr/starter/passassay/\n"
                "       - 알고리즘 : 딥 러닝을 이용한 자연어 처리 입문 https://wikidocs.net/44249\n"
                "   - 문장 요약\n"
                "       - 데이터셋 : Amazon Fine Food Reviews https://www.kaggle.com/snap/amazon-fine-food-reviews\n"
                "       - 데이터셋 : NEWS SUMMARY https://www.kaggle.com/sunnysai12345/news-summary\n"
                "       - 알고리즘 : 딥 러닝을 이용한 자연어 처리 입문 https://wikidocs.net/72820\n")