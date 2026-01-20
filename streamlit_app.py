# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Interactive Data Analysis MVP", layout="wide")

st.title("🎯 데이터 분석 루트 선택 MVP")

# --- 데이터 업로드 ---
st.sidebar.header("1. 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기:", df.head())
else:
    st.info("CSV 파일을 업로드해주세요.")
    st.stop()

# --- 루트 선택 ---
st.sidebar.header("2. 분석 루트 선택")
eda_option = st.sidebar.checkbox("EDA(기초 통계 + 시각화)", value=True)
regression_option = st.sidebar.checkbox("회귀 분석", value=False)
clustering_option = st.sidebar.checkbox("군집 분석", value=False)

# --- 분석 AI ---
st.header("📊 분석 결과")

if eda_option:
    st.subheader("1️⃣ EDA 결과")
    st.write(df.describe())
    st.write("상관관계 히트맵")
    plt.figure(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

if regression_option:
    st.subheader("2️⃣ 회귀 분석 결과")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        X_col = st.selectbox("설명 변수 선택", numeric_cols, index=0)
        y_col = st.selectbox("타깃 변수 선택", numeric_cols, index=1)
        X = df[[X_col]].values
        y = df[y_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"회귀 계수: {model.coef_[0]:.3f}, 절편: {model.intercept_:.3f}")
        st.write(f"R^2 점수: {r2_score(y_test, y_pred):.3f}")
    else:
        st.warning("수치형 변수가 최소 2개 필요합니다.")

if clustering_option:
    st.subheader("3️⃣ 군집 분석 결과")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        n_clusters = st.slider("클러스터 개수 선택", 2, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df[numeric_cols])
        st.write(df.head())
        st.write("각 클러스터 중심점:")
        st.write(kmeans.cluster_centers_)
    else:
        st.warning("수치형 변수가 필요합니다.")

# --- 보고서 AI (간단 한국어 요약) ---
st.header("📝 보고서 요약")

report = ""
if eda_option:
    report += "EDA: 데이터 기본 통계 및 상관관계 시각화 완료.\n"
if regression_option:
    report += f"회귀 분석: {X_col} → {y_col} 모델링 완료, R^2={r2_score(y_test, y_pred):.3f}.\n"
if clustering_option:
    report += f"군집 분석: {n_clusters}개의 클러스터 생성, 데이터에 Cluster 컬럼 추가.\n"

st.text_area("보고서", value=report, height=200)
