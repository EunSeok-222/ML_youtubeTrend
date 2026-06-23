import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="유튜브 조회수 구간 예측", layout="wide")

FILEPATH = 'trending_yt_videos_113_countries.csv'
FEATURES = [
    'title_length', 'tag_count', 'like_ratio', 'comment_ratio',
    'days_to_trend', 'trending_days', 'publish_weekday', 'publish_hour', 'is_korean',
    'best_rank', 'log_channel_avg',
]
FEATURE_LABELS = {
    'title_length':    '제목 길이',
    'tag_count':       '태그 수',
    'like_ratio':      '좋아요 비율',
    'comment_ratio':   '댓글 비율',
    'days_to_trend':   '트렌딩 소요 일수',
    'trending_days':   '트렌딩 유지 일수',
    'publish_weekday': '업로드 요일',
    'publish_hour':    '업로드 시간대',
    'is_korean':       '한국어 여부',
    'best_rank':       '최고 순위',
    'log_channel_avg': '채널 평균 조회수(로그)',
}


@st.cache_data(show_spinner="데이터 로딩 중... (약 1~2분 소요)")
def load_data():
    kr_rows = []
    with open(FILEPATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['country'] == 'KR':
                kr_rows.append(row)

    df = pd.DataFrame(kr_rows)

    for col in ['view_count', 'like_count', 'comment_count',
                'daily_rank', 'daily_movement', 'weekly_movement']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['publish_date']  = pd.to_datetime(df['publish_date'], utc=True).dt.tz_localize(None)

    agg = df.groupby('video_id').agg(
        title         =('title', 'first'),
        channel_name  =('channel_name', 'first'),
        view_count    =('view_count', 'max'),
        like_count    =('like_count', 'max'),
        comment_count =('comment_count', 'max'),
        best_rank     =('daily_rank', 'min'),
        trending_days =('snapshot_date', 'nunique'),
        first_trend   =('snapshot_date', 'min'),
        publish_date  =('publish_date', 'first'),
        video_tags    =('video_tags', 'first'),
        language      =('langauge', 'first'),
    ).reset_index()

    agg['title_length']    = agg['title'].str.len()
    agg['tag_count']       = agg['video_tags'].apply(
        lambda x: 0 if (pd.isna(x) or str(x).strip() == '') else len(str(x).split('|'))
    )
    agg['like_ratio']      = agg['like_count'] / (agg['view_count'] + 1)
    agg['comment_ratio']   = agg['comment_count'] / (agg['view_count'] + 1)
    agg['days_to_trend']   = (agg['first_trend'] - agg['publish_date']).dt.days.clip(lower=0)
    agg['publish_weekday'] = agg['publish_date'].dt.weekday
    agg['publish_hour']    = agg['publish_date'].dt.hour
    agg['is_korean']       = agg['language'].apply(lambda x: 1 if str(x).startswith('ko') else 0)

    channel_avg            = agg.groupby('channel_name')['view_count'].mean()
    agg['log_channel_avg'] = np.log1p(agg['channel_name'].map(channel_avg))

    df_ml = agg.dropna(subset=['view_count', 'like_count', 'comment_count',
                                'days_to_trend', 'publish_weekday', 'publish_hour',
                                'best_rank']).copy()

    df_ml['view_class'], bins = pd.qcut(
        df_ml['view_count'], q=3,
        labels=['하(저조회수)', '중(보통)', '상(고조회수)'],
        retbins=True
    )

    return df_ml, bins


@st.cache_resource(show_spinner="모델 학습 중...")
def train_model(_df_ml):
    le = LabelEncoder()
    df = _df_ml.copy()
    df['label'] = le.fit_transform(df['view_class'])

    X = df[FEATURES].fillna(0)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')

    return model, le, X_test, y_test, acc, f1


df_ml, bins = load_data()
model, le, X_test, y_test, acc, f1 = train_model(df_ml)

st.title("🎬 유튜브 한국 트렌딩 — 조회수 구간 예측")
st.write("KR 트렌딩 데이터를 기반으로 영상 정보를 분석해 조회수 구간(상/중/하)을 예측합니다.")

st.subheader("데이터 개요")
col1, col2, col3 = st.columns(3)
col1.metric("고유 영상 수", f"{len(df_ml):,}개")
col2.metric("모델 정확도 (Accuracy)", f"{acc:.1%}")
col3.metric("F1-score (macro)", f"{f1:.3f}")

st.write(
    f"조회수 구간 기준 | "
    f"하: ~{bins[1]:,.0f}회 | "
    f"중: ~{bins[2]:,.0f}회 | "
    f"상: {bins[2]:,.0f}회 초과"
)

with st.expander("데이터 샘플 보기"):
    st.dataframe(
        df_ml[['title', 'channel_name', 'view_count', 'like_count',
               'trending_days', 'view_class']].head(20),
        use_container_width=True
    )

st.divider()

st.subheader("📈 탐색적 데이터 분석 (EDA)")
tab1, tab2, tab3, tab4 = st.tabs(["조회수 분포", "요일별 조회수", "트렌딩 유지 일수", "상관관계"])

with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    df_ml['view_count'].apply(np.log1p).hist(bins=50, ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('조회수 분포 (로그 변환)', fontsize=9)
    axes[0].set_xlabel('log(조회수)', fontsize=8)
    axes[0].tick_params(labelsize=7)
    df_ml['view_class'].value_counts().plot(
        kind='bar', ax=axes[1], color=['#e74c3c', '#f39c12', '#2ecc71']
    )
    axes[1].set_title('조회수 구간(종속변수) 분포', fontsize=9)
    axes[1].tick_params(axis='x', rotation=0, labelsize=7)
    axes[1].tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close()

with tab2:
    day_labels = ['월', '화', '수', '목', '금', '토', '일']
    day_avg = df_ml.groupby('publish_weekday')['view_count'].mean() / 1e6
    fig, ax = plt.subplots(figsize=(8, 3))
    day_avg.plot(kind='bar', color='steelblue', ax=ax)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels, rotation=0, fontsize=8)
    ax.set_title('업로드 요일별 평균 조회수 (단위: 백만)', fontsize=9)
    ax.set_ylabel('평균 조회수 (백만)', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close()

with tab3:
    trend_avg = df_ml.groupby('trending_days')['view_count'].mean() / 1e6
    fig, ax = plt.subplots(figsize=(8, 3))
    trend_avg[:30].plot(kind='bar', color='coral', ax=ax)
    ax.set_title('트렌딩 유지 일수별 평균 조회수 (단위: 백만)', fontsize=9)
    ax.set_xlabel('트렌딩 유지 일수', fontsize=8)
    ax.set_ylabel('평균 조회수 (백만)', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close()

with tab4:
    num_cols = ['view_count', 'like_count', 'comment_count', 'title_length', 'tag_count',
                'like_ratio', 'comment_ratio', 'days_to_trend', 'trending_days',
                'publish_weekday', 'publish_hour', 'is_korean', 'log_channel_avg']
    corr = df_ml[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 6})
    ax.set_title('변수 간 상관관계 히트맵', fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close()

st.divider()

st.subheader("모델 성능 분석 (GradientBoosting — 11피처)")
col_cm, col_fi = st.columns(2)

with col_cm:
    st.write("**혼동 행렬 (Confusion Matrix)**")
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['하', '중', '상'],
        colorbar=False, ax=ax
    )
    ax.set_title('혼동 행렬')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_fi:
    st.write("**변수 중요도**")
    feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    feat_imp.index = [FEATURE_LABELS.get(i, i) for i in feat_imp.index]
    fig, ax = plt.subplots(figsize=(4, 3))
    feat_imp.plot(kind='barh', color='mediumseagreen', ax=ax)
    ax.set_title('변수 중요도', fontsize=9)
    ax.set_xlabel('중요도', fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.write("**ROC 곡선**")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score    = model.predict_proba(X_test)
fig, ax    = plt.subplots(figsize=(5, 3.5))
for i, (name, color) in enumerate(zip(['하(0)', '중(1)', '상(2)'],
                                       ['#e74c3c', '#f39c12', '#2ecc71'])):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('클래스별 ROC 곡선')
ax.legend()
plt.tight_layout()
roc_col, _ = st.columns(2)
with roc_col:
    st.pyplot(fig, use_container_width=True)
plt.close()

with st.sidebar:
    st.title("🔮 조회수 구간 예측")
    st.caption("영상 정보를 입력하고 예측하기를 눌러보세요.")
    st.divider()

    st.subheader("📝 영상 제목")
    title_input = st.text_input("제목을 직접 입력하세요", placeholder="예: 우리들의 재밌는 영상")
    title_len   = len(title_input)
    if title_len > 0:
        st.caption(f"제목 길이: {title_len}글자")

    st.subheader("📋 영상 기본 정보")
    day_map      = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    tag_count    = st.slider("태그 수", 0, 50, 10)
    weekday_sel  = st.selectbox("업로드 요일", list(day_map.keys()))
    publish_hour = st.slider("업로드 시간대 (시)", 0, 23, 18)
    is_korean    = st.radio("한국어 영상 여부", ["예", "아니오"], horizontal=True)

    st.subheader("📊 참여 지표")
    like_ratio    = st.slider("좋아요 비율 (좋아요/조회수)", 0.0, 0.2, 0.05, step=0.001, format="%.3f")
    comment_ratio = st.slider("댓글 비율 (댓글/조회수)", 0.0, 0.05, 0.005, step=0.001, format="%.3f")

    st.subheader("📅 트렌딩 정보")
    days_to_trend = st.number_input("트렌딩까지 걸린 일수", min_value=0, max_value=365, value=1)
    trending_days = st.number_input("트렌딩 유지 일수", min_value=1, max_value=60, value=2)
    best_rank     = st.number_input("최고 순위 (1위=최고)", min_value=1, max_value=100, value=10)

    st.subheader("📺 채널 인기도")
    channel_size = st.selectbox(
        "채널 규모",
        ["신규/소규모 (평균 10만 이하)", "중소형 (평균 10만~100만)",
         "대형 (평균 100만~500만)", "메가 (평균 500만 이상)"]
    )
    channel_avg_map = {
        "신규/소규모 (평균 10만 이하)":   50_000,
        "중소형 (평균 10만~100만)":       500_000,
        "대형 (평균 100만~500만)":      2_500_000,
        "메가 (평균 500만 이상)":       10_000_000,
    }
    channel_avg_val = channel_avg_map[channel_size]

    st.divider()
    predict_btn = st.button("예측하기", type="primary", use_container_width=True)

    if predict_btn:
        user_input = np.array([[
            title_len,
            tag_count,
            like_ratio,
            comment_ratio,
            days_to_trend,
            trending_days,
            day_map[weekday_sel],
            publish_hour,
            1 if is_korean == "예" else 0,
            best_rank,
            np.log1p(channel_avg_val),
        ]])

        pred  = model.predict(user_input)[0]
        proba = model.predict_proba(user_input)[0]
        label = le.inverse_transform([pred])[0]

        score   = proba[0] * 83.5 + proba[1] * 50.0 + proba[2] * 16.5
        top_pct = max(1, min(99, int(round(100 - score))))

        color_map = {'상(고조회수)': 'success', '중(보통)': 'warning', '하(저조회수)': 'error'}
        icon_map  = {'상(고조회수)': '📈', '중(보통)': '📊', '하(저조회수)': '📉'}

        if top_pct <= 10:
            detail = "조회수가 매우 잘 나올 영상입니다! 🔥"
        elif top_pct <= 33:
            detail = "조회수가 잘 나올 가능성이 높습니다."
        elif top_pct <= 67:
            detail = "평균 수준의 조회수가 예상됩니다."
        else:
            detail = "조회수가 낮을 가능성이 있습니다."

        getattr(st, color_map[label])(
            f"{icon_map[label]} **{label}** — 상위 {top_pct}%\n\n{detail}"
        )

        prob_df = pd.DataFrame({'확률': proba}, index=le.classes_)
        st.bar_chart(prob_df, height=200)
