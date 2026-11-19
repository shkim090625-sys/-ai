
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
# 9. 통합 분석 및 추천 결과
# =====================================

st.divider()
st.header("3. 통합 분석 및 추천")

if st.button("통합 분석 및 추천하기", type="primary"):

    if not pollutant.strip() and not description.strip():

        st.warning("오염수 정보를 하나 이상 입력해 줘.")
        st.session_state.pop("recommendation_result", None)

    else:

        try:
            ordered, metric_data, ranks = make_recommendations(
                data,
                priorities
            )

            st.session_state["recommendation_result"] = {
                "ordered": ordered,
                "metric_data": metric_data,
                "ranks": ranks,
                "pollutant": pollutant,
                "concentration": concentration,
                "description": description,
                "priorities": priorities.copy()
            }

        except Exception as e:
            st.error("추천 분석 중 오류가 발생했습니다.")
            st.code(str(e))


# 추천 결과 출력
if "recommendation_result" in st.session_state:

    result = st.session_state["recommendation_result"]

    ordered = result["ordered"]
    metric_data = result["metric_data"]
    ranks = result["ranks"]
    saved_priorities = result["priorities"]

    st.divider()
    st.subheader("입력 정보")

    st.write("**오염물질:**", result["pollutant"] or "미입력")
    st.write("**농도:**", result["concentration"] or "미입력")
    st.write("**특성:**", result["description"] or "미입력")

    st.write("**정화 우선순위:**")
    st.write(" → ".join(saved_priorities))

    if result["pollutant"].strip().lower() in [
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
            "다른 오염물질에도 동일한 결과가 나타난다고 "
            "확정할 수 없습니다."
        )

    st.divider()

    if not ordered:

        st.error("추천에 사용할 수 있는 데이터가 없습니다.")

    else:

        # 전체 추천 순위
        st.subheader("추천 순위")

        recommendation_items = []

        for index, key in enumerate(ordered[:5]):

            display = key

            for metric in metric_data.values():
                if key in metric:
                    display = metric[key]["display"]
                    break

            recommendation_items.append((key, display))

            st.write(f"**{index + 1}순위:** {display}")

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

                # 실험 데이터
                st.markdown("### 실험 데이터")

                for priority in saved_priorities:

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
                                metric_data,
                                priority,
                                key
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
                                metric_data,
                                priority,
                                key
                            )

                            st.write(
                                f"- 탁도 관찰 점수: {value:.2f}/8 "
                                f"(관찰 점수 순위 {rank}위)"
                            )

                    else:
                        st.write(
                            f"- 항균성: {item['value']}"
                        )

                st.divider()

                # 머신러닝 예측
                st.markdown("### 🤖 머신러닝 분석")

                try:

                    parts = key.split("|")

                    seaweed_name = parts[0].strip()
                    concentration_value = (
                        parts[1].strip()
                        if len(parts) > 1 else ""
                    )
                    ratio_value = (
                        parts[2].strip()
                        if len(parts) > 2 else ""
                    )

                    predicted_value = None

                    if "+" not in seaweed_name:

                        if concentration_value in [
                            "저농도",
                            "중농도",
                            "고농도"
                        ]:

                            predicted_value = predict_single(
                                seaweed_name,
                                concentration_value
                            )

                        else:
                            st.info(
                                "농도 정보가 없어 예측하지 않았습니다."
                            )

                    else:

                        mixture_names = [
                            name.strip()
                            for name in seaweed_name.split("+")
                        ]

                        if (
                            len(mixture_names) == 2
                            and ratio_value in ["1:1", "3:1"]
                        ):

                            predicted_value = predict_mixture(
                                mixture_names[0],
                                mixture_names[1],
                                ratio_value
                            )

                        else:
                            st.info(
                                "혼합 해조류의 이름 또는 혼합비를 "
                                "확인할 수 없어 예측하지 않았습니다."
                            )

                    if predicted_value is not None:

                        st.metric(
                            "예측 흡광도",
                            f"{predicted_value:.3f}"
                        )

                        actual_item = metric_data[
                            "흡광도"
                        ].get(key)

                        if (
                            actual_item is not None
                            and actual_item["value"] is not None
                        ):

                            actual_value = actual_item["value"]

                            error = abs(
                                predicted_value - actual_value
                            )

                            st.write(
                                f"실제 측정값: {actual_value:.3f}"
                            )

                            st.write(
                                f"예측값과 실제값의 절대 차이: "
                                f"{error:.3f}"
                            )

                except Exception as e:

                    st.warning(
                        f"머신러닝 예측 결과를 표시하지 못했습니다: {e}"
                    )

                st.caption(
                    "머신러닝 예측값은 학습된 모델의 추정값이며 "
                    "실제 실험 측정값과 구분해야 합니다."
                )

        st.divider()

        # 추천 판단 기준
        st.subheader("추천 판단 기준")

        for i, priority in enumerate(
            saved_priorities,
            start=1
        ):
            st.write(f"{i}순위 기준: {priority}")

        st.info(
            "흡광도는 낮을수록, 탁도 관찰 점수는 높을수록 "
            "우선적으로 비교했습니다. "
            "항균성은 억제환 관찰 여부로 구분했습니다. "
            "서로 다른 지표를 하나의 수치로 합산하지 않았습니다."
        )

        st.warning(
            "추천 결과는 현재 확보한 실험 데이터에 기반한 "
            "조건 비교입니다. 실험하지 않은 오염물질이나 "
            "조건의 정화 효과를 보장하지 않습니다."
        )


# =====================================
# 10. 논문 검색 (OpenAlex)
# =====================================

st.divider()
st.header("4. 추천 결과 기반 관련 논문 검색")

st.write(
    "추천된 해조류와 입력한 오염물질을 바탕으로 "
    "관련 학술 자료를 자동 검색합니다."
)

if "recommendation_result" not in st.session_state:

    st.info("먼저 통합 분석 및 추천을 실행해 주세요.")

else:

    result = st.session_state["recommendation_result"]

    pollutant = str(result.get("pollutant") or "").strip()

    ordered = result.get("ordered", [])
    metric_data = result.get("metric_data", {})

    # 해조류별 학명 및 영문명
    seaweed_names = {
        "곰피": ["Ecklonia stolonifera"],
        "바다포도": ["Caulerpa lentillifera"],
        "다시마": ["Saccharina japonica", "Laminaria japonica"],
        "청각": ["Codium fragile"]
    }

    # 추천 상위 5개에서 해조류 추출
    recommended_seaweeds = []

    for key in ordered[:5]:

        display = str(key)

        for metric in metric_data.values():

            if key in metric:
                display = str(metric[key]["display"])
                break

        for seaweed in seaweed_names:

            if seaweed in display and seaweed not in recommended_seaweeds:
                recommended_seaweeds.append(seaweed)

    st.write("**검색 대상 오염물질:**", pollutant or "미입력")

    st.write(
        "**추천 해조류:**",
        ", ".join(recommended_seaweeds) or "확인되지 않음"
    )

    st.caption(
        "논문 검색은 추천 결과에 포함된 해조류를 기준으로 진행합니다."
    )

    if st.button("추천 결과 관련 논문 검색"):

        if not pollutant:

            st.warning("오염물질 정보가 없어 검색할 수 없습니다.")

        elif not recommended_seaweeds:

            st.warning("추천 결과에서 해조류를 확인할 수 없습니다.")

        else:

            import json
            import urllib.parse
            import urllib.request
            import urllib.error
            import re

            all_papers = {}
            search_errors = []

            # 해조류와 오염물질을 조합한 검색
            search_queries = []

            for seaweed in recommended_seaweeds:

                scientific_names = seaweed_names[seaweed]

                for scientific_name in scientific_names:

                    search_queries.append(
                        f'"{scientific_name}" "{pollutant}"'
                    )

                    search_queries.append(
                        f'"{scientific_name}" water treatment'
                    )

                search_queries.append(
                    f'{seaweed} {pollutant} 수질 정화'
                )

            # 중복 검색어 제거
            search_queries = list(dict.fromkeys(search_queries))

            with st.spinner("관련 논문을 검색하고 있습니다..."):

                for query in search_queries:

                    params = urllib.parse.urlencode({
                        "search": query,
                        "per-page": 20
                    })

                    url = (
                        "https://api.openalex.org/works?"
                        + params
                    )

                    request = urllib.request.Request(
                        url,
                        headers={
                            "User-Agent":
                            "SeaweedWaterPurificationAI/1.0"
                        }
                    )

                    try:

                        with urllib.request.urlopen(
                            request,
                            timeout=20
                        ) as response:

                            search_data = json.loads(
                                response.read().decode("utf-8")
                            )

                        for paper in search_data.get("results", []):

                            paper_id = paper.get("id")

                            if paper_id:
                                all_papers[paper_id] = paper

                    except Exception as e:

                        search_errors.append(str(e))

            # 초록 복원
            def get_abstract(paper):

                abstract_index = paper.get(
                    "abstract_inverted_index"
                )

                if not abstract_index:
                    return ""

                words = [
                    (position, word)
                    for word, positions in abstract_index.items()
                    for position in positions
                ]

                return " ".join(
                    word for position, word in sorted(words)
                )

            # 검색 결과 분류
            direct_papers = []
            similar_papers = []

            pollutant_lower = pollutant.lower()

            for paper in all_papers.values():

                title = paper.get("title") or ""
                abstract = get_abstract(paper)

                text_content = (
                    title + " " + abstract
                ).lower()

                matched_seaweeds = []

                for seaweed in recommended_seaweeds:

                    for name in seaweed_names[seaweed]:

                        if name.lower() in text_content:

                            matched_seaweeds.append(seaweed)
                            break

                pollutant_match = (
                    pollutant_lower in text_content
                )

                water_keywords = [
                    "adsorption",
                    "absorption",
                    "removal",
                    "purification",
                    "water treatment",
                    "antimicrobial",
                    "pollutant",
                    "dye"
                ]

                function_match = any(
                    word in text_content
                    for word in water_keywords
                )

                paper["reconstructed_abstract"] = abstract
                paper["matched_seaweeds"] = list(
                    dict.fromkeys(matched_seaweeds)
                )

                # 동일 해조류와 오염물질이 제목 또는 초록에
                # 모두 확인된 경우에만 직접 근거 후보로 분류
                if matched_seaweeds and pollutant_match:

                    direct_papers.append(paper)

                elif matched_seaweeds or (
                    pollutant_match and function_match
                ):

                    similar_papers.append(paper)

            # 검색 결과 표시
            st.success(
                f"총 {len(all_papers)}개의 중복 제거된 자료를 검색했습니다."
            )

            st.subheader("① 직접 근거 논문")

            st.caption(
                "추천 해조류의 학명과 입력 오염물질이 "
                "제목 또는 초록에서 함께 확인된 자료입니다. "
                "실제 실험 조건까지 동일하다는 뜻은 아닙니다."
            )

            if not direct_papers:

                st.info(
                    "검색된 자료 중 직접 근거 기준에 "
                    "해당하는 논문이 없습니다."
                )

            else:

                for index, paper in enumerate(direct_papers[:5]):

                    with st.expander(
                        f"{index + 1}. {paper.get('title') or '제목 없음'}"
                    ):

                        st.write(
                            "**발행 연도:**",
                            paper.get("publication_year") or "정보 없음"
                        )

                        st.write(
                            "**확인된 해조류:**",
                            ", ".join(paper["matched_seaweeds"])
                        )

                        st.write(
                            "**DOI:**",
                            paper.get("doi") or "정보 없음"
                        )

                        if paper.get("doi"):

                            st.markdown(
                                f"[논문 원문 링크]({paper['doi']})"
                            )

                        if paper["reconstructed_abstract"]:

                            st.write(
                                "**초록:**",
                                paper["reconstructed_abstract"]
                            )

                        else:

                            st.write("초록: 제공되지 않음")

                        st.caption(
                            "주의: 검색어 일치에 따른 후보 분류이며 "
                            "논문의 실험 결과를 직접 검증한 것은 아닙니다."
                        )

            st.divider()

            st.subheader("② 유사 근거 논문")

            st.caption(
                "추천 해조류 또는 오염물질이나 정화 기작 중 "
                "일부가 관련된 자료입니다."
            )

            if not similar_papers:

                st.info("검색된 유사 근거 자료가 없습니다.")

            else:

                for index, paper in enumerate(similar_papers[:5]):

                    with st.expander(
                        f"{index + 1}. {paper.get('title') or '제목 없음'}"
                    ):

                        st.write(
                            "**발행 연도:**",
                            paper.get("publication_year") or "정보 없음"
                        )

                        st.write(
                            "**관련 해조류:**",
                            ", ".join(paper["matched_seaweeds"])
                            or "검색된 해조류 없음"
                        )

                        st.write(
                            "**DOI:**",
                            paper.get("doi") or "정보 없음"
                        )

                        if paper.get("doi"):

                            st.markdown(
                                f"[논문 원문 링크]({paper['doi']})"
                            )

                        if paper["reconstructed_abstract"]:

                            st.write(
                                "**초록:**",
                                paper["reconstructed_abstract"]
                            )

                        else:

                            st.write("초록: 제공되지 않음")

                        st.caption(
                            "유사 근거는 추천 조건에 대한 간접적인 "
                            "참고 자료이며 동일한 정화 효과를 보장하지 않습니다."
                        )

            if search_errors:

                st.warning(
                    "일부 검색 요청에 실패했습니다. "
                    "검색 결과가 제한되었을 수 있습니다."
                )


# 데이터베이스에 저장된 문헌 정보
with st.expander("데이터베이스에 저장된 문헌 정보"):

    literature_df = data["Literature"]

    if literature_df.empty:

        st.info("저장된 문헌 정보가 없습니다.")

    else:

        st.dataframe(
            literature_df,
            use_container_width=True
        )


# =====================================
# 11. 데이터베이스 정보
# =====================================

st.divider()
st.header("5. 데이터베이스 정보")

with st.expander("사용 중인 데이터베이스 확인"):

    for name, df in data.items():

        st.write(f"**{name}:** {len(df)}개 행")

        st.dataframe(
            df,
            use_container_width=True
        )


# =====================================
# 12. 별도 머신러닝 흡광도 예측
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

        try:

            prediction = predict_single(
                selected_seaweed,
                selected_concentration
            )

            st.metric(
                "예측 흡광도",
                f"{prediction:.3f}"
            )

        except Exception as e:

            st.error(
                f"예측에 실패했습니다: {e}"
            )


else:

    seaweed1 = st.selectbox(
        "첫 번째 해조류",
        SEAWEEDS,
        key="mixture_seaweed1"
    )

    seaweed2_options = [
        name
        for name in SEAWEEDS
        if name != seaweed1
    ]

    seaweed2 = st.selectbox(
        "두 번째 해조류",
        seaweed2_options,
        key="mixture_seaweed2"
    )

    ratio = st.selectbox(
        "혼합비 (첫 번째 해조류 기준)",
        ["1:1", "3:1"]
    )

    if st.button("혼합 해조류 흡광도 예측"):

        try:

            prediction = predict_mixture(
                seaweed1,
                seaweed2,
                ratio
            )

            st.metric(
                "예측 흡광도",
                f"{prediction:.3f}"
            )

        except Exception as e:

            st.error(
                f"예측에 실패했습니다: {e}"
            )


st.info(
    "예상 흡광도 해석 안내\n\n"
    "- 예상 흡광도가 낮을수록 메틸렌 블루 용액의 흡광도가 낮을 것으로 예측한 것입니다.\n"
    "- 예상값은 머신러닝 모델의 추정치이며 실제 측정값과 차이가 발생할 수 있습니다.\n"
    "- 현재 모델은 흡광도만 예측하며, 탁도와 항균성은 예측하지 않습니다.\n"
    "- 메틸렌 블루 이외의 오염물질에 대한 예측 성능은 검증되지 않았습니다."
)
