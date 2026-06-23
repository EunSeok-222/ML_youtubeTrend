from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

FEATURES = [
    'title_length', 'tag_count', 'like_ratio', 'comment_ratio',
    'days_to_trend', 'trending_days', 'publish_weekday', 'publish_hour', 'is_korean',
    'best_rank', 'log_channel_avg',
]


def prepare_xy(df_ml):
    le = LabelEncoder()
    df = df_ml.copy()
    df['label'] = le.fit_transform(df['view_class'])
    X = df[FEATURES].fillna(0)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, le


def train_all_models(X_train, y_train):
    models = {
        '로지스틱 회귀': LogisticRegression(max_iter=500, random_state=42),
        '결정 트리': DecisionTreeClassifier(max_depth=5, random_state=42),
        '랜덤 포레스트': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='mlogloss',
                                  random_state=42, n_jobs=-1, verbosity=0),
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
    return models


def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=5, scoring='f1_macro', n_jobs=-1
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, eval_metric='mlogloss',
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)
    return model
