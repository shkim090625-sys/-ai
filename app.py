
import streamlit as st
import pandas as pd
from pathlib import Path

# -----------------------------
# 1. 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="해조류 수질 정화 AI",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 해조류 기반 수질 정화 AI")
st.write(
    "오염수의 특성과 정화 목적을 바탕으로 "
    "실험 데이터에 근거한 해조류와 혼합 조건을 추천합니다."
)

# -----------------------------
# 2. CSV 데이터 불러오기
# -----------------------------
FILES = {
    "START_HERE": "seaweed_ai_database_corrected(1).csv",
    "Absorbance_Data": "seaweed_ai_database_corrected(1-2).csv",
    "Turbidity_Data": "seaweed_ai_database_corrected(1-3).csv",
    "Antimicrobial_Data": "seaweed_ai_database_corrected(1-4).csv",
    "Seaweed_Info": "seaweed_ai_database_corrected(1-5).csv",
    "Literature": "seaweed_ai_database_corrected(1-6).csv",
    "Recommendation_Rules": "seaweed_ai_database_corrected(1-7).csv",
    "Data_Dictionary": "seaweed_ai_database_corrected(1-8).csv",
}


@st.cache_data
def load_data():
    data = {}

    for sheet_name, filename in FILES.items():
        path = Path(__file__).parent / filename

        if not path.exists():
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {filename}"
            )

        try:
            data[sheet_name] = pd.read_csv(
                path,
                encoding="utf-8-sig"
            )
        except pd.errors.EmptyDataError:
            data[sheet_name] = pd.DataFrame()

    return data


try:
    data = load_data()
    st.success("실험 데이터베이스 8개 파일을 불러왔습니다.")

except Exception as e:
    st.error("데이터베이스를 불러오지 못했습니다.")
    st.error(str(e))
    st.stop()


# -----------------------------
# 3. 공통 함수
# -----------------------------
def find_column(df, candidates):
    """후보 이름 중 실제 데이터에 존재하는 열을 찾는다."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def show_data(df):
    """데이터가 존재할 때 표로 표시한다."""
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
    else:
        st.dataframe(df, use_container_width=True)


def clean_numeric(series):
    """측정값을 숫자로 변환한다."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )


# -----------------------------
# 4. 사용자 입력
# -----------------------------
st.divider()
st.header("1. 정화 목적 선택")

target = st.selectbox(
    "어떤 정화 기능을 우선적으로 고려할까요?",
    [
        "흡광도 관련 정화",
        "탁도 개선",
        "항균성"
    ]
)

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

# -----------------------------
# 5. 추천 결과
# -----------------------------
st.divider()
st.header("3. 해조류 추천")

if st.button("해조류 추천하기", type="primary"):

    if not pollutant.strip() and not description.strip():
        st.warning("오염수 정보를 하나 이상 입력해 주세요.")

    else:
        st.subheader("입력 정보")

        st.write("**정화 목적:**", target)
        st.write("**오염물질:**", pollutant or "미입력")
        st.write("**농도:**", concentration or "미입력")
        st.write("**추가 정보:**", description or "없음")

        st.divider()

        # -------------------------
        # 흡광도 기반 추천
        # -------------------------
        if target == "흡광도 관련 정화":

            df = data["Absorbance_Data"].copy()

            value_col = find_column(df, ["흡광도"])
            name_col = find_column(df, ["해조류/조합", "해조류", "조합"])
            type_col = find_column(df, ["실험유형"])
            concentration_col = find_column(df, ["농도"])
            ratio_col = find_column(df, ["혼합비"])

            if value_col is None or name_col is None:
                st.error("흡광도 데이터의 필수 열을 찾지 못했습니다.")
            else:
                df[value_col] = clean_numeric(df[value_col])
                valid = df.dropna(subset=[value_col]).copy()

                if valid.empty:
                    st.warning("사용할 수 있는 흡광도 측정값이 없습니다.")
                else:
                    # 흡광도 원자료가 낮은 조건을 우선 표시
                    best = valid.sort_values(
                        by=value_col,
                        ascending=True
                    )

                    st.subheader("추천 결과")
                    st.write(
                        "측정된 흡광도 원자료가 낮은 조건부터 표시했어. "
                        "이는 해당 실험 조건에서 관찰된 값의 비교이며, "
                        "오염물질 제거율을 의미하지는 않아."
                    )

                    st.markdown("**흡광도 측정값이 낮은 조건**")
                    show_data(best.head(5))

                    st.markdown("**전체 측정 결과**")
                    show_data(valid)

                    st.info(
                        "증류수 blank 값은 0으로 기록되어 있습니다. "
                        "해조류 추출물 자체의 색이나 탁도를 보정한 값인지는 "
                        "별도로 확인해야 합니다."
                    )

        # -------------------------
        # 탁도 기반 추천
        # -------------------------
        elif target == "탁도 개선":

            df = data["Turbidity_Data"].copy()

            value_col = find_column(df, ["점수"])
            name_col = find_column(df, ["해조류/조합", "해조류", "조합"])

            if value_col is None or name_col is None:
                st.error("탁도 데이터의 필수 열을 찾지 못했습니다.")
            else:
                df[value_col] = clean_numeric(df[value_col])
                valid = df.dropna(subset=[value_col]).copy()

                if valid.empty:
                    st.warning("사용할 수 있는 탁도 측정값이 없습니다.")
                else:
                    best = valid.sort_values(
                        by=value_col,
                        ascending=False
                    )

                    st.subheader("추천 결과")
                    st.write(
                        "줄무늬 배경의 가시성을 기준으로 점수가 높은 "
                        "조건부터 표시했어. 점수가 높을수록 줄무늬가 "
                        "잘 보였다는 뜻이야."
                    )

                    st.markdown("**탁도 관찰 점수가 높은 조건**")
                    show_data(best.head(5))

                    st.markdown("**전체 측정 결과**")
                    show_data(valid)

                    st.info(
                        "이 점수는 육안 관찰 점수(0~8)이며, "
                        "NTU나 탁도 제거율이 아닙니다. "
                        "자연 침전의 영향도 고려해야 합니다."
                    )

        # -------------------------
        # 항균성 기반 추천
        # -------------------------
        elif target == "항균성":

            df = data["Antimicrobial_Data"].copy()

            result_col = find_column(
                df,
                ["억제환 유무", "억제환", "결과"]
            )

            if result_col is None:
                st.error("항균성 데이터의 결과 열을 찾지 못했습니다.")
            else:
                def is_positive(value):
                    value = str(value).strip().lower()

                    positive_words = [
                        "있음", "유", "o", "yes",
                        "positive", "양성", "나타남"
                    ]

                    return value in positive_words

                positive = df[
                    df[result_col].apply(is_positive)
                ].copy()

                st.subheader("추천 결과")

                if positive.empty:
                    st.warning(
                        "현재 데이터에서 억제환이 확인된 조건이 없습니다."
                    )
                else:
                    st.write(
                        "억제환이 관찰된 조건이야. "
                        "억제환의 크기를 측정하지 않았으므로 "
                        "항균 효과의 강도를 순위화하지 않았어."
                    )
                    show_data(positive)

                st.markdown("**전체 항균성 실험 결과**")
                show_data(df)

                st.info(
                    "실험 날짜와 대조군 조건이 다른 결과는 "
                    "직접적인 시간 변화나 효과 크기 비교에 "
                    "사용하지 않았습니다."
                )

        # -------------------------
        # 결과 해석 안내
        # -------------------------
        st.divider()
        st.subheader("결과 해석 시 주의사항")

        st.write(
            "현재 추천은 입력한 오염물질의 특성을 자동으로 분석해 "
            "새로운 정화 효과를 예측하는 머신러닝 모델이 아닙니다."
        )

        st.write(
            "실제 추천은 데이터베이스에 기록된 실험 결과를 "
            "정화 목적에 따라 비교하는 방식입니다."
        )

        st.write(
            "실험하지 않은 오염물질이나 해조류 조합의 효과는 "
            "확인된 사실처럼 제시하지 않습니다."
        )
