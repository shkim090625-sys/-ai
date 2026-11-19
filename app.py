
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================
# 1. 페이지 설정
# =====================================
st.set_page_config(
    page_title="해조류 수질 정화 AI",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 해조류 기반 수질 정화 AI")
st.write(
    "오염수의 특성과 사용자의 정화 우선순위를 바탕으로 "
    "실험 데이터에 근거한 해조류 및 혼합 조건을 추천합니다."
)

# =====================================
# 학습된 머신러닝 모델 불러오기
# =====================================

MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource
def load_ml_models():
    single_model = joblib.load(
        MODEL_DIR / "single_seaweed_model.pkl"
    )

    mixture_model = joblib.load(
        MODEL_DIR / "mixture_seaweed_model.pkl"
    )

    model_info = joblib.load(
        MODEL_DIR / "model_info.pkl"
    )

    return single_model, mixture_model, model_info


try:
    single_model, mixture_model, model_info = load_ml_models()
    st.success("머신러닝 모델 연결 완료")

except Exception as e:
    st.error("머신러닝 모델을 불러오지 못했습니다.")
    st.code(str(e))

# =====================================
# 2. CSV 데이터베이스
# =====================================
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
    result = {}

    for sheet, filename in FILES.items():
        path = Path(__file__).parent / filename

        if not path.exists():
            raise FileNotFoundError(f"파일 없음: {filename}")

        try:
            result[sheet] = pd.read_csv(
                path,
                encoding="utf-8-sig"
            )
        except pd.errors.EmptyDataError:
            result[sheet] = pd.DataFrame()

    return result


try:
    data = load_data()
    st.success("데이터베이스 연결 완료")

except Exception as e:
    st.error("데이터베이스 연결에 실패했습니다.")
    st.code(str(e))
    st.stop()


# =====================================
# 3. 공통 데이터 처리 함수
# =====================================
def find_col(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_number(value):
    return pd.to_numeric(
        pd.Series([value]).astype(str).str.replace(",", ""),
        errors="coerce"
    ).iloc[0]


def is_positive(value):
    value = clean_text(value).lower()

    if any(word in value for word in ["미측정", "기록 없음", "불명"]):
        return None

    if value in ["있음", "유", "o", "yes", "positive", "양성", "나타남"]:
        return True

    if value in ["없음", "무", "x", "no", "negative", "음성", "안 나타남"]:
        return False

    return None

# =====================================
# 머신러닝 흡광도 예측 함수
# =====================================

SEAWEEDS = ["곰피", "바다포도", "다시마", "청각"]


def predict_single(seaweed, concentration):
    features = pd.DataFrame(
        [[
            1 if name == seaweed else 0
            for name in SEAWEEDS
        ] + [
            1 if concentration == level else 0
            for level in ["저농도", "중농도", "고농도"]
        ]],
        columns=model_info["single_features"]
    )

    return float(single_model.predict(features)[0])


def predict_mixture(seaweed1, seaweed2, ratio):
    proportions = {
        seaweed1: 0.75 if ratio == "3:1" else 0.5,
        seaweed2: 0.25 if ratio == "3:1" else 0.5
    }

    features = pd.DataFrame(
        [[
            proportions.get(name, 0)
            for name in SEAWEEDS
        ] + [
            1 if ratio == "1:1" else 0,
            1 if ratio == "3:1" else 0
        ]],
        columns=model_info["mixture_features"]
    )

    return float(mixture_model.predict(features)[0])

# =====================================
# 4. 실험 조건 정리
# =====================================
def prepare_conditions(df, metric):

    if df.empty:
        return {}

    name_col = find_col(
        df,
        ["해조류/조합", "해조류", "조합"]
    )

    if name_col is None:
        return {}

    concentration_col = find_col(df, ["농도"])
    ratio_col = find_col(df, ["혼합비"])

    if metric == "absorbance":
        value_col = find_col(df, ["흡광도"])

    elif metric == "turbidity":
        value_col = find_col(df, ["점수"])

    else:
        value_col = find_col(
            df,
            ["억제환 유무", "억제환", "결과"]
        )

    if value_col is None:
        return {}

    groups = {}

    for _, row in df.iterrows():

        name = clean_text(row.get(name_col, ""))

        if not name:
            continue

        # 대조군은 해조류 추천 대상에서 제외
        if name in ["증류수", "메틸렌 블루"]:
            continue
        
        concentration = (
            clean_text(row.get(concentration_col, ""))
            if concentration_col else ""
        )

        ratio = (
            clean_text(row.get(ratio_col, ""))
            if ratio_col else ""
        )

        display = name

        if concentration:
            display += " / 농도 " + concentration

        if ratio:
            display += " / 혼합비 " + ratio

        key = "|".join([name, concentration, ratio])

        if key not in groups:
            groups[key] = {
                "display": display,
                "values": []
            }

        value = row.get(value_col)

        if metric in ["absorbance", "turbidity"]:

            number = clean_number(value)

            if pd.notna(number):
                groups[key]["values"].append(float(number))

        else:

            result = is_positive(value)

            if result is not None:
                groups[key]["values"].append(result)

    for item in groups.values():

        values = item["values"]

        if metric in ["absorbance", "turbidity"]:

            item["value"] = (
                sum(values) / len(values)
                if values else None
            )

        else:

            if True in values:
                item["value"] = "억제환 관찰"

            elif values and all(v is False for v in values):
                item["value"] = "억제환 미관찰"

            else:
                item["value"] = "미측정 또는 불명확"

    return groups


# =====================================
# 5. 사용자 입력
# =====================================
st.divider()
st.header("1. 오염수 정보")

pollutant = st.text_input(
    "오염물질 이름",
    placeholder="예: 메틸렌 블루"
)

description = st.text_area(
    "오염수의 특성",
    placeholder="예: 염료가 포함된 수용액이며 색이 진함"
)

concentration = st.text_input(
    "오염물질 농도 (선택)",
    placeholder="예: 10 mg/L"
)


# =====================================
# 6. 정화 우선순위 설정
# =====================================
st.divider()
st.header("2. 정화 기능 우선순위")

st.write(
    "가장 중요하게 생각하는 정화 기능부터 순서대로 선택해 줘."
)

metric_names = ["흡광도", "탁도", "항균성"]

priority1 = st.selectbox(
    "1순위 정화 기능",
    metric_names,
    index=0
)

remaining1 = [x for x in metric_names if x != priority1]

priority2 = st.selectbox(
    "2순위 정화 기능",
    remaining1,
    index=0
)

remaining2 = [x for x in remaining1 if x != priority2]

priority3 = st.selectbox(
    "3순위 정화 기능",
    remaining2,
    index=0
)

priorities = [priority1, priority2, priority3]


# =====================================
# 7. 통합 추천 알고리즘
# =====================================
def make_recommendations(data, priorities):

    metric_data = {
        "흡광도": prepare_conditions(
            data["Absorbance_Data"], "absorbance"
        ),
        "탁도": prepare_conditions(
            data["Turbidity_Data"], "turbidity"
        ),
        "항균성": prepare_conditions(
            data["Antimicrobial_Data"], "antimicrobial"
        )
    }

    all_conditions = set()

    for metric in metric_data.values():
        all_conditions.update(metric.keys())

    if not all_conditions:
        return [], metric_data, {}

    ranks = {
        "흡광도": {},
        "탁도": {},
        "항균성": {}
    }

    # 흡광도: 낮을수록 우선
    valid_abs = [
        (key, item["value"])
        for key, item in metric_data["흡광도"].items()
        if item["value"] is not None
    ]

    valid_abs.sort(key=lambda x: x[1])

    for rank, (key, value) in enumerate(valid_abs, start=1):
        ranks["흡광도"][key] = rank

    # 탁도: 관찰 점수가 높을수록 우선
    valid_turb = [
        (key, item["value"])
        for key, item in metric_data["탁도"].items()
        if item["value"] is not None
    ]

    valid_turb.sort(key=lambda x: x[1], reverse=True)

    for rank, (key, value) in enumerate(valid_turb, start=1):
        ranks["탁도"][key] = rank

    # 항균성: 관찰 여부만 구분
    for key, item in metric_data["항균성"].items():

        value = item["value"]

        if value == "억제환 관찰":
            ranks["항균성"][key] = 1

        elif value == "억제환 미관찰":
            ranks["항균성"][key] = 2

        else:
            ranks["항균성"][key] = 3

    # 사용자 우선순위에 따라 순차 비교
    def sort_key(key):

        result = []

        for priority in priorities:

            rank = ranks[priority].get(key)

            result.append(
                rank if rank is not None else 999999
            )

        return tuple(result)

    ordered = sorted(
        all_conditions,
        key=sort_key
    )

    return ordered, metric_data, ranks


# =====================================
# 8. 순위 확인 함수
# =====================================
def ranks_for_display(metric_data, priority, key):

    items = metric_data[priority]

    if priority == "흡광도":

        valid = sorted(
            [
                (k, v["value"])
                for k, v in items.items()
                if v["value"] is not None
            ],
            key=lambda x: x[1]
        )

    elif priority == "탁도":

        valid = sorted(
            [
                (k, v["value"])
                for k, v in items.items()
                if v["value"] is not None
            ],
            key=lambda x: x[1],
            reverse=True
        )

    else:

        return (
            1 if items[key]["value"] == "억제환 관찰"
            else 2
        )

    for rank, (condition, value) in enumerate(valid, start=1):

        if condition == key:
            return rank

    return "순위 없음"


# =====================================
# 9. 추천 결과 출력
# =====================================
st.divider()
st.header("3. 통합 분석 및 추천")

if st.button("통합 분석 및 추천하기", type="primary"):

    if not pollutant.strip() and not description.strip():

        st.warning("오염수 정보를 하나 이상 입력해 줘.")

    else:

        ordered, metric_data, ranks = make_recommendations(
            data,
            priorities
        )

        st.subheader("입력 정보")

        st.write("**오염물질:**", pollutant or "미입력")
        st.write("**농도:**", concentration or "미입력")
        st.write("**특성:**", description or "미입력")

        st.write("**정화 우선순위:**")
        st.write(" → ".join(priorities))

        # 메틸렌 블루 직접 실험 여부 확인
        if pollutant.strip().lower() in [
            "메틸렌 블루",
            "methylene blue"
        ]:

            st.success(
                "입력한 물질은 흡광도 실험에 사용한 "
                "메틸렌 블루와 일치합니다."
            )

        else:

            st.warning(
                "현재 흡광도 실험은 메틸렌 블루를 대상으로 했습니다. "
                "다른 오염물질에 동일한 결과가 나타난다고 "
                "확정할 수 없습니다."
            )

        st.divider()

        if not ordered:

            st.error("추천에 사용할 수 있는 데이터가 없습니다.")

        else:

            st.subheader("추천 결과")

            st.write(
                "사용자가 설정한 우선순위에 따라 "
                "실험 조건을 순차적으로 비교한 결과입니다."
            )

            for index, key in enumerate(ordered[:5]):

                display = None

                for metric in metric_data.values():

                    if key in metric:
                        display = metric[key]["display"]
                        break

                st.markdown("---")

                if index == 0:
                    st.markdown("### 🏆 1순위 추천")
                else:
                    st.markdown(f"### {index + 1}순위 추천")

                st.write("**추천 조건:**", display)


                # 머신러닝 예측 결과 표시
                st.markdown("**🤖 머신러닝 예측 결과**")

                try:
                    parts = key.split("|")
                    seaweed_name = parts[0]

                    if "+" not in seaweed_name:
                        concentration_value = parts[1] if len(parts) > 1 else ""

                        if concentration_value in [
                            "저농도", "중농도", "고농도"
                        ]:
                            predicted_value = predict_single(
                                seaweed_name,
                                concentration_value
                            )

                            st.write(
                                f"예측 흡광도: {predicted_value:.3f}"
                            )

                        else:
                            st.info(
                                "농도 정보가 없어 예측하지 않았습니다."
                            )

                    else:
                        st.info(
                            "혼합 조건의 예측은 다음 단계에서 연결합니다."
                        )

                except Exception as e:
                    st.warning(f"예측 결과 표시 실패: {e}")

                
                st.markdown("**정화 기능별 실험 근거**")

                for priority in priorities:

                    item = metric_data[priority].get(key)

                    if item is None:

                        st.write(f"- {priority}: 측정 자료 없음")

                    elif priority == "흡광도":

                        value = item["value"]

                        if value is None:
                            st.write("- 흡광도: 측정값 없음")

                        else:

                            rank = ranks_for_display(
                                metric_data, priority, key
                            )

                            st.write(
                                f"- 흡광도: {value:.3f} "
                                f"(측정값 순위 {rank}위)"
                            )

                    elif priority == "탁도":

                        value = item["value"]

                        if value is None:
                            st.write("- 탁도: 측정값 없음")

                        else:

                            rank = ranks_for_display(
                                metric_data, priority, key
                            )

                            st.write(
                                f"- 탁도 관찰 점수: {value:.2f}/8 "
                                f"(관찰 점수 순위 {rank}위)"
                            )


            # 전체 추천 순위 구성
            recommendation_items = []

            for key in ordered[:5]:
                display = None

                for metric in metric_data.values():
                    if key in metric:
                        display = metric[key]["display"]
                        break

                recommendation_items.append((key, display))

            # 전체 추천 순위 표시
            st.subheader("전체 추천 순위")

            for index, (key, display) in enumerate(
                recommendation_items
            ):
                st.write(
                    f"**{index + 1}순위:** {display}"
                )

            st.divider()

            # 순위별 상세 분석
            st.subheader("추천 조건 상세 분석")


for index, (key, display) in enumerate(
    recommendation_items
):

    with st.expander(
        f"{index + 1}순위 상세 분석",
        expanded=(index == 0)
    ):

        st.markdown(f"**추천 조건:** {display}")

        st.markdown("### 실험 데이터")

        for priority in priorities:

            item = metric_data[priority].get(key)

            if item is None:
                st.write(
                    f"- {priority}: 측정 자료 없음"
                )

            elif priority == "흡광도":

                value = item["value"]

                if value is None:
                    st.write("- 흡광도: 측정값 없음")

                else:
                    rank = ranks_for_display(
                        metric_data, priority, key
                    )

                    st.write(
                        f"- 실제 흡광도: {value:.3f} "
                        f"(측정값 순위 {rank}위)"
                    )

            elif priority == "탁도":

                value = item["value"]

                if value is None:
                    st.write("- 탁도: 측정값 없음")

                else:
                    rank = ranks_for_display(
                        metric_data, priority, key
                    )

                    st.write(
                        f"- 탁도 관찰 점수: {value:.2f}/8 "
                        f"(관찰 점수 순위 {rank}위)"
                    )

            else:
                st.write(f"- 항균성: {item['value']}")

                st.divider()

                    # 머신러닝 예측
                st.markdown("### 🤖 머신러닝 분석")

                    try:
                        parts = key.split("|")
                        seaweed_name = parts[0]
                        concentration_value = (
                            parts[1] if len(parts) > 1 else ""
                        )

                        if (
                            "+" not in seaweed_name
                            and concentration_value in [
                                "저농도", "중농도", "고농도"
                            ]
                        ):

                            predicted_value = predict_single(
                                seaweed_name,
                                concentration_value
                            )

                            st.metric(
                                "예측 흡광도",
                                f"{predicted_value:.3f}"
                            )

                            absorbance_item = metric_data.get(
                                "흡광도", {}
                            ).get(key)

                            if (
                                absorbance_item is not None
                                and absorbance_item["value"] is not None
                            ):

                                actual_value = absorbance_item["value"]

                                st.write(
                                    f"실제 측정값: {actual_value:.3f}"
                                )

                                st.write(
                                    f"예측 절대 오차: "
                                    f"{abs(absorbance_item['value'] - predicted_value):.3f}"
                                )

                        else:
                            st.info(
                                "현재 조건은 단일 해조류 모델의 "
                                "입력 형식과 일치하지 않아 "
                                "예측하지 않았습니다."
                            )

                    except Exception as e:
                        st.warning(
                            f"예측 결과 표시 실패: {e}"
                        )

                    st.caption(
                        "머신러닝 예측값은 실제 측정값이 아닌 "
                        "학습 모델의 추정값입니다."
                    )

            st.divider()

            st.subheader("추천 판단 기준")

            for i, priority in enumerate(priorities, start=1):
                st.write(
                    f"{i}순위 기준: {priority}"
                )

            st.info(
                "흡광도와 탁도는 각각의 측정 방향에 따라 "
                "비교했습니다. 항균성은 억제환 관찰 여부만 "
                "구분했습니다. 서로 다른 지표를 하나의 "
                "수치로 합산하지 않았습니다."
            )

            st.warning(
                "추천 결과는 현재 실험 데이터에 기반한 "
                "조건 비교입니다. 실험하지 않은 오염물질이나 "
                "조건의 효과를 보장하지 않습니다."
            )

        
        # =================================
        # 10. 문헌 자료
        # =================================
        st.divider()
        st.subheader("문헌 근거")

        literature = data["Literature"]

        if literature.empty:

            st.info(
                "현재 등록된 문헌 자료가 없습니다. "
                "따라서 이번 결과에는 문헌 기반 추천이 포함되지 않았습니다."
            )

        else:

            st.write(
                "등록된 문헌 자료입니다. "
                "실험 결과와 문헌 결과는 구분해서 해석해야 합니다."
            )

            st.dataframe(
                literature,
                use_container_width=True
            )

import json
import urllib.parse
import urllib.request
import urllib.error

st.divider()
st.header("📚 실시간 해외 논문 검색")

st.write("오염물질과 해조류 관련 연구 논문을 OpenAlex에서 검색합니다.")

search_keyword = st.text_input(
    "검색할 연구 주제",
    placeholder="예: methylene blue seaweed"
)

if st.button("논문 검색하기"):
    if not search_keyword.strip():
        st.warning("검색어를 입력해 주세요.")
    else:
        try:
            api_key = st.secrets["OPENALEX_API_KEY"]

            params = urllib.parse.urlencode({
                "search": search_keyword,
                "per-page": 5,
                "api_key": api_key
            })

            url = "https://api.openalex.org/works?" + params

            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SeaweedWaterPurificationAI"}
            )

            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.loads(response.read().decode("utf-8"))

            papers = result.get("results", [])

            if not papers:
                st.info("검색 결과가 없습니다.")

            for paper in papers:
                st.subheader(paper.get("title", "제목 없음"))

                authors = paper.get("authorships", [])
                author_names = [
                    a.get("author", {}).get("display_name", "")
                    for a in authors[:5]
                ]

                st.write("저자:", ", ".join(author_names) or "정보 없음")
                st.write("출판 연도:", paper.get("publication_year", "정보 없음"))
                st.write("DOI:", paper.get("doi") or "정보 없음")

                abstract_index = paper.get("abstract_inverted_index")
                if abstract_index:
                    words = [
                        (position, word)
                        for word, positions in abstract_index.items()
                        for position in positions
                    ]
                    abstract = " ".join(
                        word for position, word in sorted(words)
                    )
                    st.write("초록:", abstract)
                else:
                    st.write("초록: 제공되지 않음")

                st.divider()

        except KeyError:
            st.error("Streamlit Secrets에서 API 키를 찾을 수 없습니다.")
        except urllib.error.HTTPError as e:
            st.error(f"OpenAlex 검색 오류: HTTP {e.code}")
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")


# =====================================
# 11. 데이터베이스 정보
# =====================================
with st.expander("사용 중인 데이터베이스 확인"):

    for name, df in data.items():

        st.write(f"**{name}:** {len(df)}개 행")
        

# =====================================
# 12. 머신러닝 흡광도 예측
# =====================================

st.divider()
st.header("🤖 머신러닝 흡광도 예측")

st.write(
    "학습된 Random Forest 모델을 이용해 "
    "단일 해조류 또는 혼합 해조류의 흡광도를 예측합니다."
)

prediction_type = st.radio(
    "예측할 실험 유형",
    ["단일 해조류", "혼합 해조류"],
    horizontal=True
)

if prediction_type == "단일 해조류":

    selected_seaweed = st.selectbox(
        "해조류 선택",
        SEAWEEDS
    )

    selected_concentration = st.selectbox(
        "농도 선택",
        ["저농도", "중농도", "고농도"]
    )

    if st.button("단일 해조류 흡광도 예측"):

        prediction = predict_single(
            selected_seaweed,
            selected_concentration
        )

        st.metric(
            "예측 흡광도",
            f"{prediction:.3f}"
        )

else:

    seaweed1 = st.selectbox(
        "첫 번째 해조류",
        SEAWEEDS,
        key="mixture_seaweed1"
    )

    seaweed2_options = [
        name for name in SEAWEEDS
        if name != seaweed1
    ]

    seaweed2 = st.selectbox(
        "두 번째 해조류",
        seaweed2_options
    )

    ratio = st.selectbox(
        "혼합비 (첫 번째 해조류 기준)",
        ["1:1", "3:1"]
    )

    if st.button("혼합 해조류 흡광도 예측"):

        prediction = predict_mixture(
            seaweed1,
            seaweed2,
            ratio
        )

        st.metric(
            "예측 흡광도",
            f"{prediction:.3f}"
        )

st.caption(
    "주의: 예측값은 현재 보유한 실험 데이터로 학습한 모델의 추정값입니다. "
    "실제 측정값이 아니며, 학습 데이터 범위를 벗어난 조건에서는 "
    "정확도가 보장되지 않습니다."
)
