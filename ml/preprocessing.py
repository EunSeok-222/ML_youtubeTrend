import csv
import numpy as np
import pandas as pd


FILEPATH = 'trending_yt_videos_113_countries.csv'


def load_kr_data(filepath=FILEPATH):
    kr_rows = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['country'] == 'KR':
                kr_rows.append(row)
    return pd.DataFrame(kr_rows)


def convert_types(df):
    for col in ['view_count', 'like_count', 'comment_count', 'daily_rank']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['publish_date'] = pd.to_datetime(df['publish_date'], utc=True).dt.tz_localize(None)
    return df


def aggregate_by_video(df):
    return df.groupby('video_id').agg(
        title=('title', 'first'),
        channel_name=('channel_name', 'first'),
        view_count=('view_count', 'max'),
        like_count=('like_count', 'max'),
        comment_count=('comment_count', 'max'),
        best_rank=('daily_rank', 'min'),
        trending_days=('snapshot_date', 'nunique'),
        first_trend=('snapshot_date', 'min'),
        publish_date=('publish_date', 'first'),
        video_tags=('video_tags', 'first'),
        language=('langauge', 'first'),
    ).reset_index()


def add_features(agg):
    agg['title_length'] = agg['title'].str.len()
    agg['tag_count'] = agg['video_tags'].apply(
        lambda x: 0 if (pd.isna(x) or str(x).strip() == '') else len(str(x).split('|'))
    )
    agg['like_ratio'] = agg['like_count'] / (agg['view_count'] + 1)
    agg['comment_ratio'] = agg['comment_count'] / (agg['view_count'] + 1)
    agg['days_to_trend'] = (agg['first_trend'] - agg['publish_date']).dt.days.clip(lower=0)
    agg['publish_weekday'] = agg['publish_date'].dt.weekday
    agg['publish_hour'] = agg['publish_date'].dt.hour
    agg['is_korean'] = agg['language'].apply(lambda x: 1 if str(x).startswith('ko') else 0)
    channel_avg = agg.groupby('channel_name')['view_count'].mean()
    agg['log_channel_avg'] = np.log1p(agg['channel_name'].map(channel_avg))
    return agg


def make_target(agg):
    df_ml = agg.dropna(subset=['view_count', 'like_count', 'comment_count',
                                'days_to_trend', 'publish_weekday', 'publish_hour',
                                'best_rank']).copy()
    df_ml['view_class'], bins = pd.qcut(
        df_ml['view_count'], q=3,
        labels=['하(저조회수)', '중(보통)', '상(고조회수)'],
        retbins=True
    )
    return df_ml, bins


def build_dataset(filepath=FILEPATH):
    df = load_kr_data(filepath)
    df = convert_types(df)
    agg = aggregate_by_video(df)
    agg = add_features(agg)
    df_ml, bins = make_target(agg)
    return df_ml, bins
