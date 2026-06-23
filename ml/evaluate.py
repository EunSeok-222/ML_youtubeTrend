import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import platform
matplotlib.rcParams['font.family'] = 'AppleGothic' if platform.system() == 'Darwin' else 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro'),
    }


def plot_confusion_matrix(model, X_test, y_test, title='혼동 행렬'):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['하', '중', '상'],
        colorbar=False, ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, label_map=None, title='변수 중요도'):
    import pandas as pd
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    if label_map:
        imp.index = [label_map.get(i, i) for i in imp.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot(kind='barh', color='mediumseagreen', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('중요도')
    plt.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test, title='ROC 곡선'):
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (name, color) in enumerate(zip(['하(0)', '중(1)', '상(2)'],
                                           ['#e74c3c', '#f39c12', '#2ecc71'])):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def print_report(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
