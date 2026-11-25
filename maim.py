
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="KBO 구단 투수력 분석",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ KBO 구단 투수력 분석")
st.subheader("팀 평균자책점(ERA)과 승수(W)의 상관관계 분석")

st.markdown("""
2024년부터 2026년까지 KBO 10개 구단의 기록을 활용하여
팀 평균자책점과 승수 사이의 관계를 분석합니다.

산점도, 상관계수, 선형 회귀분석을 통해
평균자책점으로 구단의 승수를 예측할 수 있는지 알아봅니다.
""")

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "data" / "kbo_team.csv"

uploaded_file = st.file_uploader(
    "KBO 데이터 CSV 파일",
    type=["csv"]
)


@st.cache_data
def read_csv_file(file_bytes, encoding):
    from io import BytesIO
    return pd.read_csv(
        BytesIO(file_bytes),
        encoding=encoding
    )


def load_data():
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
    elif CSV_PATH.exists():
        file_bytes = CSV_PATH.read_bytes()
    else:
        return None

    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return read_csv_file(file_bytes, encoding)
        except Exception:
            continue

    raise ValueError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인해 주세요.")


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터 읽기 오류: {e}")
    st.stop()


if df is None:
    st.error("CSV 파일을 찾을 수 없습니다.")
    st.info(
        "GitHub에 data 폴더를 만들고 "
        "그 안에 kbo_team.csv 파일을 올려 주세요."
    )
    st.stop()


# ==========================================
# 3. 데이터 전처리
# ==========================================

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

# 열 이름 통일
column_aliases = {
    "승수": "승수",
    "승": "승수",
    "W": "승수",
    "무승부": "무승부",
    "무": "무승부",
    "D": "무승부",
    "패수": "패수",
    "패": "패수",
    "L": "패수",
    "팀ERA": "팀ERA",
    "ERA": "팀ERA",
    "평균자책점": "팀ERA",
    "팀 평균자책점": "팀ERA",
    "팀 평균자책점(ERA)": "팀ERA",
    "팀명": "팀명",
    "팀": "팀명",
    "구단명": "팀명",
    "연도": "연도",
    "년도": "연도",
    "시즌": "연도"
}

df = df.rename(columns=column_aliases)

required_columns = [
    "승수", "무승부", "패수", "팀ERA", "팀명", "연도"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(f"CSV에 필요한 열이 없습니다: {missing}")
    st.write("현재 CSV 열:", list(df.columns))
    st.stop()

# 필요한 6개 열만 사용
df = df[required_columns].copy()

# 숫자형 변환
numeric_columns = [
    "승수", "무승부", "패수", "팀ERA", "연도"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

df["팀명"] = df["팀명"].astype(str).str.strip()

# 결측값 제거
before_count = len(df)

df = df.dropna(
    subset=required_columns
).copy()

removed_count = before_count - len(df)

# 숫자 오류 제거
df = df[
    (df["승수"] >= 0) &
    (df["무승부"] >= 0) &
    (df["패수"] >= 0) &
    (df["팀ERA"] >= 0)
].copy()

# 연도 정수형 변환
df["연도"] = df["연도"].astype(int)

# 분석 기간 제한
df = df[
    df["연도"].between(2024, 2026)
].copy()

# 합계 행 제거
df = df[
    ~df["팀명"].str.contains(
        "합계|전체|평균|TOTAL",
        case=False,
        regex=True
    )
].copy()

# 중복 확인
if df.duplicated(["연도", "팀명"]).any():
    st.error("같은 연도에 동일한 구단이 중복되어 있습니다.")
    st.stop()

if len(df) < 5:
    st.error("분석할 데이터가 부족합니다. CSV 내용을 확인해 주세요.")
    st.stop()

df = df.sort_values(
    ["연도", "팀명"]
).reset_index(drop=True)

st.success("데이터를 정상적으로 불러왔습니다.")

if removed_count > 0:
    st.warning(f"결측값 {removed_count}행을 제거했습니다.")

if len(df) != 30:
    st.warning(
        f"현재 분석 데이터는 {len(df)}행입니다. "
        "보고서 기준인 30행인지 확인해 주세요."
    )

# 구단별 10개 데이터 확인
team_counts = df.groupby("연도")["팀명"].nunique()

for year, count in team_counts.items():
    if count != 10:
        st.warning(
            f"{year}년 구단 수가 {count}개입니다. "
            "10개 구단 데이터인지 확인해 주세요."
        )


# ==========================================
# 4. 데이터 요약
# ==========================================

st.divider()
st.header("1. 데이터 개요")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 데이터", f"{len(df)}행")
c2.metric("분석 연도", f"{df['연도'].nunique()}개")
c3.metric("구단 수", f"{df['팀명'].nunique()}개")
c4.metric("평균 승수", f"{df['승수'].mean():.1f}승")

st.caption(
    "출처: KBO 공식 기록실 | 분석 단위: 구단별 시즌 기록"
)

with st.expander("전체 데이터 확인하기"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

st.divider()


# ==========================================
# 5. 분석 연도 선택
# ==========================================

st.header("2. 분석 조건 선택")

year_options = ["전체"] + sorted(
    df["연도"].unique().tolist()
)

selected_year = st.selectbox(
    "분석할 시즌",
    year_options
)

if selected_year == "전체":
    data = df.copy()
else:
    data = df[
        df["연도"] == selected_year
    ].copy()

st.write(f"현재 분석 대상: **{len(data)}개 구단 기록**")

if len(data) < 3:
    st.warning("분석할 데이터가 부족합니다.")
    st.stop()


# ==========================================
# 6. 기초 통계
# ==========================================

st.divider()
st.header("3. 기초 통계 분석")

stats = data[["팀ERA", "승수", "무승부", "패수"]].describe()

st.dataframe(
    stats.round(2),
    use_container_width=True
)

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "평균 팀 ERA",
        f"{data['팀ERA'].mean():.2f}"
    )

with c2:
    st.metric(
        "평균 승수",
        f"{data['승수'].mean():.1f}승"
    )


# ==========================================
# 7. 상관관계 분석
# ==========================================

st.divider()
st.header("4. 상관관계 분석")

st.markdown("""
팀 평균자책점(ERA)과 승수(W) 사이의
피어슨 상관계수를 계산합니다.
""")

if data["팀ERA"].nunique() > 1 and data["승수"].nunique() > 1:

    correlation = data["팀ERA"].corr(data["승수"])

    if correlation < -0.7:
        relation = "강한 음의 상관관계"
    elif correlation < -0.3:
        relation = "중간 정도의 음의 상관관계"
    elif correlation < 0:
        relation = "약한 음의 상관관계"
    elif correlation < 0.3:
        relation = "약한 양의 상관관계"
    elif correlation < 0.7:
        relation = "중간 정도의 양의 상관관계"
    else:
        relation = "강한 양의 상관관계"

    c1, c2 = st.columns(2)

    c1.metric(
        "피어슨 상관계수",
        f"{correlation:.3f}"
    )

    c2.metric(
        "상관관계 유형",
        relation
    )

    st.info(
        f"현재 데이터에서는 ERA와 승수 사이에 "
        f"**{relation}**가 나타났습니다."
    )

else:
    correlation = None
    st.warning("상관계수를 계산할 수 있는 데이터가 부족합니다.")

st.caption(
    "상관관계는 두 변수의 관련성을 나타내며, "
    "평균자책점이 승수의 직접적인 원인이라는 뜻은 아닙니다."
)


# ==========================================
# 8. 핵심 산점도
# ==========================================

st.divider()
st.header("5. 산점도 분석")

st.markdown("""
**그래프 1) 산점도 - 팀 평균자책점과 승수의 관계**

- X축: 팀 평균자책점(ERA)
- Y축: 팀 승수(W)
- 점 하나: 한 시즌의 한 구단
- 빨간 선: 선형 회귀선
""")

scatter = px.scatter(
    data,
    x="팀ERA",
    y="승수",
    color="연도",
    hover_name="팀명",
    hover_data={
        "무승부": True,
        "패수": True,
        "팀ERA": ":.2f",
        "승수": True
    },
    labels={
        "팀ERA": "팀 평균자책점 (ERA)",
        "승수": "팀 승수 (W)",
        "연도": "시즌"
    },
    title="KBO 구단 평균자책점과 승수의 관계"
)

# 회귀선 추가
if data["팀ERA"].nunique() > 1:

    slope, intercept = np.polyfit(
        data["팀ERA"],
        data["승수"],
        1
    )

    x_line = np.linspace(
        data["팀ERA"].min(),
        data["팀ERA"].max(),
        100
    )

    y_line = slope * x_line + intercept

    scatter.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="회귀선",
            line=dict(
                color="red",
                width=3
            )
        )
    )

scatter.update_layout(
    height=600,
    template="plotly_white"
)

st.plotly_chart(
    scatter,
    use_container_width=True
)

st.markdown("""
**산점도 해석 기준**

- 왼쪽 위에서 오른쪽 아래 방향: 음의 상관관계
- 왼쪽 아래에서 오른쪽 위 방향: 양의 상관관계
- 점들이 일정한 방향 없이 흩어짐: 약한 상관관계
""")

st.divider()


# ==========================================
# 9. 구단별 비교 그래프
# ==========================================

st.header("6. 구단별 기록 비교")

tab1, tab2 = st.tabs([
    "팀 평균자책점 비교",
    "팀 승수 비교"
])

with tab1:

    era_fig = px.bar(
        data.sort_values("팀ERA"),
        x="팀명",
        y="팀ERA",
        color="연도",
        barmode="group",
        title="구단별 팀 평균자책점 비교",
        labels={
            "팀명": "구단",
            "팀ERA": "평균자책점"
        },
        text_auto=".2f"
    )

    era_fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(
        era_fig,
        use_container_width=True
    )

    st.caption("평균자책점은 낮을수록 실점 억제 측면에서 유리합니다.")

with tab2:

    win_fig = px.bar(
        data.sort_values("승수", ascending=False),
        x="팀명",
        y="승수",
        color="연도",
        barmode="group",
        title="구단별 승수 비교",
        labels={
            "팀명": "구단",
            "승수": "승수"
        },
        text_auto=True
    )

    win_fig.update_layout(
        template="plotly_white",
        height=450
    )

    st.plotly_chart(
        win_fig,
        use_container_width=True
    )

st.divider()


# ==========================================
# 10. 선형 회귀분석
# ==========================================

st.header("7. 선형 회귀분석")

st.markdown("""
팀 평균자책점(ERA)을 독립변수(X),
팀 승수(W)를 종속변수(Y)로 설정합니다.

데이터를 학습용 80%, 테스트용 20%로 나누고
기준선 모델과 선형 회귀모델의 예측 성능을 비교합니다.
""")

if len(data) >= 10 and data["팀ERA"].nunique() > 1:

    X = data[["팀ERA"]]
    y = data["승수"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 기준선 모델
    baseline = np.full(
        len(y_test),
        y_train.mean()
    )

    # 선형 회귀모델
    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(X_test)

    # 평가 지표
    baseline_mae = mean_absolute_error(
        y_test,
        baseline
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline
        )
    )

    model_mae = mean_absolute_error(
        y_test,
        prediction
    )

    model_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    model_r2 = r2_score(
        y_test,
        prediction
    ) if len(y_test) >= 2 else np.nan

    # 회귀식
    st.subheader("회귀식")

    st.latex(
        rf"W = {model.coef_[0]:.3f} \times ERA + ({model.intercept_:.3f})"
    )

    st.write(
        f"학습 데이터: {len(X_train)}개 / "
        f"테스트 데이터: {len(X_test)}개"
    )

    # 성능 비교표
    comparison = pd.DataFrame({
        "평가 지표": ["MAE", "RMSE"],
        "기준선 모델": [
            baseline_mae,
            baseline_rmse
        ],
        "선형 회귀모델": [
            model_mae,
            model_rmse
        ]
    })

    st.subheader("기준선 모델과 회귀모델 비교")

    st.dataframe(
        comparison.round(3),
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{model_mae:.2f}승"
    )

    c2.metric(
        "RMSE",
        f"{model_rmse:.2f}승"
    )

    c3.metric(
        "R²",
        f"{model_r2:.3f}"
    )

    if model_mae < baseline_mae:
        st.success(
            "선형 회귀모델의 MAE가 기준선 모델보다 작습니다."
        )
    else:
        st.warning(
            "선형 회귀모델이 기준선 모델보다 "
            "더 좋은 예측 성능을 보이지 못했습니다."
        )

    # 테스트 데이터 예측 비교
    result = pd.DataFrame({
        "실제 승수": y_test.values,
        "예측 승수": prediction.round(2),
        "예측 오차": (
            y_test.values - prediction
        ).round(2)
    })

    st.subheader("실제 승수와 예측 승수 비교")

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "회귀모델 평가를 위해서는 최소 10개 이상의 데이터가 필요합니다."
    )


# ==========================================
# 11. 분석 결과 정리
# ==========================================

st.divider()
st.header("8. 분석 결과 정리")

if correlation is not None:

    st.markdown("### 우리 데이터의 분석 결과")

    st.write(
        f"- 분석에 사용한 데이터: **{len(data)}개**"
    )

    st.write(
        f"- 평균 팀 ERA: **{data['팀ERA'].mean():.2f}**"
    )

    st.write(
        f"- 평균 승수: **{data['승수'].mean():.2f}승**"
    )

    st.write(
        f"- ERA와 승수의 상관계수: **{correlation:.3f}**"
    )

    if correlation < 0:
        st.write(
            "평균자책점이 높아질수록 승수가 감소하는 "
            "방향의 관계가 나타났습니다."
        )
    elif correlation > 0:
        st.write(
            "평균자책점이 높아질수록 승수가 증가하는 "
            "방향의 관계가 나타났습니다."
        )
    else:
        st.write(
            "두 변수 사이에 선형적인 상관관계가 거의 나타나지 않았습니다."
        )

st.markdown("""
### 분석할 때 주의할 점

1. 평균자책점은 투수진의 실점 억제 능력을 나타내는 지표입니다.
2. 승수는 투수진뿐 아니라 타선의 득점 지원과 수비력에도 영향을 받습니다.
3. 상관관계가 있다고 해서 한 변수가 다른 변수의 원인이라고 단정할 수 없습니다.
4. 30개 데이터만으로 모든 시즌의 구단 성적을 설명하기는 어렵습니다.
5. 2026년 자료가 정규시즌 최종 기록인지 확인하고 다른 연도와 동일한 기준으로 비교해야 합니다.

""")

st.divider()

st.caption(
    "데이터 출처: KBO 공식 기록실 "
    "| 분석 방법: 상관관계 분석 및 선형 회귀분석 "
    "| 제작: 21023 이준서 · 21013 김규민 · 21011 최서현"
)
