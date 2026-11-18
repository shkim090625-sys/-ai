
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="해조류 수질 정화 AI",
    page_icon="🌿",
    layout="wide"
)

# 제목
st.title("🌿 해조류 기반 수질 정화 AI")
st.write(
    "오염수의 특성과 정화 목적을 바탕으로 "
    "적합한 해조류와 혼합 조건을 추천하는 시스템입니다."
)

st.divider()

# 정화 목적 선택
st.header("1. 정화 목적 선택")

target = st.selectbox(
    "어떤 정화 기능을 우선적으로 고려할까요?",
    ["흡광도 관련 정화", "탁도 개선", "항균성"]
)

# 오염수 정보 입력
st.header("2. 오염수 정보 입력")

pollutant = st.text_input(
    "오염물질 이름",
    placeholder="예: 메틸렌 블루"
)

description = st.text_area(
    "오염수의 특성 및 추가 정보",
    placeholder="예: 염료가 포함된 수용액이며 색이 진함"
)

concentration = st.text_input(
    "오염물질 농도 (선택)",
    placeholder="예: 10 mg/L"
)

# 추천 버튼
st.divider()
st.header("3. 해조류 추천")

if st.button("해조류 추천하기", type="primary"):

    if not pollutant and not description:
        st.warning("오염수 정보를 하나 이상 입력해 주세요.")

    else:
        st.success("입력 정보가 정상적으로 접수되었습니다.")

        st.write("### 입력한 정보")
        st.write("정화 목적:", target)
        st.write("오염물질:", pollutant if pollutant else "미입력")
        st.write("농도:", concentration if concentration else "미입력")
        st.write("추가 정보:", description if description else "없음")

        st.info(
            "추천 알고리즘은 다음 개발 단계에서 연결할 예정입니다."
        )
