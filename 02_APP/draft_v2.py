import streamlit as st

import pandas as pd
import classify
import summary_v2

def app():
    st.subheader('Select the Question')

    data = pd.read_csv('APP/Data/question_draft.CSV')
    question = st.selectbox("질문을 선택해 주십시오.", data['question'])
    st.write("선택된 질문은 <" + question + "> 입니다.")

    c_btn = st.button("Classify", key='1')
    if c_btn:
        result = question.title()
        clssfy = classify.Classify()
        with st.spinner('분류 중 입니다...'):
            category = clssfy.classification(result)

        selected_data = data[data['question'].str.contains(question)]
        draft1 = selected_data['draft1'].values[0]
        draft2 = selected_data['draft2'].values[0]
        draft3 = selected_data['draft3'].values[0]
        draft4 = selected_data['draft4'].values[0]

        if category == 0:
            # 지원동기 및 포부
            st.success("지원 동기 및 포부 관련 질문입니다.")
            st.subheader('Enter the Experience Automatically')
            st.markdown("- 동기 Motivation\n"
                        "   - " + draft1 + "\n"
                        "- 비전 Vision\n"
                        "   - " + draft2 + "\n"
                        "- 열정 Passion\n"
                        "   - " + draft3 + "\n"
                        "- 역량 Competence\n"
                        "   - " + draft4)
            # 요약 : 역량 + 비전
            draft = draft3 + draft2
        elif category == 1:
            # 경험 역량
            st.success("경험, 역량 관련 질문입니다.")
            st.subheader('Enter the Experience Automatically')
            st.markdown("- 상황 Situation\n"
                        "   - " + draft1 + "\n"
                        "- 위기 Crisis\n"
                        "   - " + draft2 + "\n"
                        "- 행동 Action\n"
                        "   - " + draft3 + "\n"
                        "- 결과 Result\n"
                        "   - " + draft4)
            # 요약 : 상황 + 결과
            draft = draft1 + draft4
        elif category == 2:
            # for test
            st.success("요약 테스트입니다.")
            st.subheader('Test Summarization Automatically')
            st.markdown("- --\n"
                        "   - " + draft1)
            draft2 = ""
            draft3 = ""
            draft4 = ""
            draft = draft1

        smmry = summary_v2.Summary()
        with st.spinner('요약 중 입니다...'):
            drft_smmry = smmry.summarization(draft)

        st.text("\"NEWS SUMMARY\" Dataset을 이용하여 Summary 하였습니다.")
        st.success("완성된 요약문은 \"" + drft_smmry + "\" 입니다.")

        st.subheader('Draft')
        st.info(drft_smmry + "\n\n "
                + draft1 + "\n\n "
                + draft2 + "\n\n "
                + draft3 + "\n\n "
                + draft4 + "\n\n ")