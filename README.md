# 유튜브 트렌딩 영상 조회수 구간 예측

Kaggle의 글로벌 유튜브 트렌딩 데이터(113개국)에서 한국(KR) 데이터를 필터링하여, 영상 메타데이터만으로 조회수 구간(상/중/하)을 예측하는 머신러닝 프로젝트입니다.

## 프로젝트 구조

```
ML_youtubeTrend/
├── notebooks/
│   └── youtube_trending_kr_analysis.ipynb   # 전체 분석 노트북 (STEP 1~14)
├── reports/
│   └── figures/                             # 분석 결과 이미지 10종
├── youtube_app.py                           # Streamlit 예측 웹앱
├── requirements.txt
└── .gitignore
```

## 사용 데이터

- **출처**: Kaggle — Global YouTube Trending Video Statistics
- **파일**: `trending_yt_videos_113_countries.csv` (113개국, 약 50만 행)
- **분석 대상**: `country == 'KR'` 필터링 후 약 19,000개 고유 영상

> CSV 파일은 용량 문제로 `.gitignore`에 포함되어 있습니다. Kaggle에서 직접 다운로드하세요.

## 사용 피처 (11개)

| 피처 | 설명 |
|---|---|
| title_length | 제목 글자 수 |
| tag_count | 태그 개수 |
| like_ratio | 좋아요 / 조회수 비율 |
| comment_ratio | 댓글 / 조회수 비율 |
| days_to_trend | 업로드 후 트렌딩까지 소요 일수 |
| trending_days | 트렌딩 유지 일수 |
| publish_weekday | 업로드 요일 (0=월 ~ 6=일) |
| publish_hour | 업로드 시간대 |
| is_korean | 한국어 영상 여부 (1/0) |
| best_rank | 트렌딩 기간 중 최고 순위 |
| log_channel_avg | 채널 평균 조회수 (로그 변환) |

## 모델 비교

| 모델 | Accuracy | F1-score |
|---|---|---|
| 로지스틱 회귀 | 0.557 | 0.540 |
| 결정 트리 | 0.592 | 0.554 |
| 랜덤 포레스트 | 0.624 | 0.616 |
| GradientBoosting | 0.611 | 0.589 |
| XGBoost | - | - |
| **랜덤 포레스트 (튜닝)** | **0.636** | **0.626** |
| **GradientBoosting (11피처)** | **개선** | **개선** |

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run youtube_app.py
```

## 종속변수 정의

`pd.qcut`으로 조회수를 3분위 분할:
- **하** (하위 33%)
- **중** (중위 33~66%)
- **상** (상위 33%)
