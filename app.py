import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

FEATURES_INFO = {
    'fixed_acidity': ('Фиксированная кислотность, г/л', 0.1, '%.1f'),
    'volatile_acidity': ('Летучая кислотность, г/л', 0.01, '%.3f'),
    'citric_acid': ('Лимонная кислота, г/л', 0.01, '%.2f'),
    'residual_sugar': ('Остаточный сахар, г/л', 0.1, '%.1f'),
    'chlorides': ('Хлориды, г/л', 0.001, '%.3f'),
    'free_sulfur_dioxide': ('Свободный диоксид серы, мг/л', 1.0, '%.1f'),
    'total_sulfur_dioxide': ('Общий диоксид серы, мг/л', 1.0, '%.1f'),
    'density': ('Плотность, г/см³', 0.0001, '%.5f'),
    'pH': ('pH', 0.01, '%.2f'),
    'sulphates': ('Сульфаты, г/л', 0.01, '%.2f'),
    'alcohol': ('Алкоголь, % об.', 0.1, '%.2f'),
}
FEATURE_LABELS = {feature: info[0] for feature, info in FEATURES_INFO.items()}
FEATURE_LABELS['is_red'] = 'Красное вино'


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def test_curves(y, proba):
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    precision, recall, _ = metrics.precision_recall_curve(y, proba)
    return {
        'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall,
        'roc_auc': metrics.roc_auc_score(y, proba), 'pr_auc': metrics.average_precision_score(y, proba),
    }


@st.cache_data
def feature_stats(X):
    return X.agg(['min', 'median', 'max'])


def show_single_wine(model, features, stats, threshold):
    st.subheader('Характеристики вина')
    wine_type = st.radio('Тип вина', ['Белое', 'Красное'], horizontal=True)
    values = {}
    columns = st.columns(3)
    for i, (feature, (label, step, fmt)) in enumerate(FEATURES_INFO.items()):
        values[feature] = columns[i % 3].number_input(
            label,
            min_value=float(stats.loc['min', feature]),
            max_value=float(stats.loc['max', feature]),
            value=float(stats.loc['median', feature]),
            step=step,
            format=fmt,
            key=f'input_{feature}',
        )
    values['is_red'] = int(wine_type == 'Красное')

    proba = model.predict_proba(pd.DataFrame([values])[features])[0, 1]
    st.divider()
    left, right = st.columns(2)
    left.metric('Вероятность, что вино хорошее', f'{proba:.1%}')
    right.metric('Текущий порог', f'{threshold:.3f}')
    if proba >= threshold:
        st.success('Решение модели: хорошее вино, рекомендуем в подборку.')
    else:
        st.warning('Решение модели: не хорошее вино, в подборку не попадает.')


def show_metrics(y_test, proba_test, threshold):
    pred = (proba_test >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    st.subheader(f'Качество на тестовой выборке при пороге {threshold:.3f}')
    cols = st.columns(4)
    cols[0].metric('Precision', f'{precision:.3f}', help='Доля действительно хороших вин среди попавших в подборку')
    cols[1].metric('Recall', f'{recall:.3f}', help='Доля хороших вин, которые попали в подборку')
    cols[2].metric('F1', f'{f1:.3f}')
    cols[3].metric('Вин в подборке', f'{tp + fp} из {len(pred)}')

    st.info(
        'Как порог меняет подборку. Если порог увеличить, модель берёт только вина, в которых уверена: '
        'подборка становится меньше, посредственных вин в ней меньше (precision растёт), '
        'но больше хороших вин остаётся за бортом (recall падает). '
        'Если порог уменьшить, подборка становится шире: хорошие вина пропускаются реже, '
        'зато в неё чаще попадают посредственные.'
    )

    curves = test_curves(y_test.to_numpy(), proba_test)
    col_cm, col_roc, col_pr = st.columns(3)

    with col_cm:
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        metrics.ConfusionMatrixDisplay(cm, display_labels=['не хорошее', 'хорошее']).plot(
            ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set_xlabel('Решение модели')
        ax.set_ylabel('Истинный класс')
        ax.set_title('Матрица ошибок')
        st.pyplot(fig)
        plt.close(fig)

    with col_roc:
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        ax.plot(curves['fpr'], curves['tpr'], label=f'ROC-AUC = {curves["roc_auc"]:.3f}')
        ax.plot([0, 1], [0, 1], ':', color='gray')
        ax.scatter(fp / (fp + tn), recall, s=90, color='red', edgecolor='k', zorder=3, label='текущий порог')
        ax.set(xlabel='FPR', ylabel='TPR (recall)', title='ROC-кривая')
        ax.legend(loc='lower right')
        st.pyplot(fig)
        plt.close(fig)

    with col_pr:
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        ax.step(curves['recall'], curves['precision'], where='post', label=f'PR-AUC = {curves["pr_auc"]:.3f}')
        ax.axhline(y_test.mean(), linestyle=':', color='gray', label='случайная модель')
        if tp + fp > 0:
            ax.scatter(recall, precision, s=90, color='red', edgecolor='k', zorder=3, label='текущий порог')
        ax.set(xlabel='recall', ylabel='precision', title='PR-кривая', ylim=(0, 1.05))
        ax.legend(loc='upper right')
        st.pyplot(fig)
        plt.close(fig)


def show_batch(model, features, example, threshold):
    st.subheader('Пакетное предсказание')
    st.write(
        f'Загрузите CSV-файл с признаками вин (столбцы: {", ".join(features)}). '
        f'Решение принимается при пороге из боковой панели, сейчас {threshold:.3f}.'
    )
    st.download_button('Скачать пример файла', example.to_csv(index=False).encode('utf-8'),
                       file_name='wines_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV-файл с признаками вин', type='csv')
    if uploaded is None:
        return

    try:
        data = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f'Не удалось прочитать файл как CSV: {error}')
        return
    if data.empty:
        st.error('В файле нет ни одной строки с данными.')
        return

    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        st.error('В файле не хватает столбцов: ' + ', '.join(missing))
        return

    X = data[features].apply(pd.to_numeric, errors='coerce')
    bad_rows = np.flatnonzero(X.isna().any(axis=1).to_numpy()) + 1
    if len(bad_rows) > 0:
        st.error(
            'В строках ' + ', '.join(map(str, bad_rows[:10])) + (' и других' if len(bad_rows) > 10 else '')
            + ' есть пропуски или нечисловые значения. Исправьте файл и загрузите его снова.'
        )
        return

    result = data.copy()
    proba = model.predict_proba(X)[:, 1]
    result.insert(0, 'probability', proba)
    result.insert(1, 'decision', np.where(proba >= threshold, 'хорошее', 'не хорошее'))
    st.write(f'Вин в файле: {len(result)}, в подборку попадает: {(result["decision"] == "хорошее").sum()}.')
    st.dataframe(result, column_config={'probability': st.column_config.NumberColumn('probability', format='%.3f')})
    st.download_button('Скачать таблицу с предсказаниями', result.to_csv(index=False).encode('utf-8'),
                       file_name='wine_predictions.csv', mime='text/csv')


def short_name(label):
    name = label.split(',')[0]
    return name if name == 'pH' else name[0].lower() + name[1:]


def show_coefficients(model, features):
    st.subheader('Коэффициенты логистической регрессии')
    coefs = pd.Series(model[-1].coef_[0], index=[FEATURE_LABELS.get(f, f) for f in features]).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = np.where(coefs.to_numpy() > 0, '#2e7d32', '#c62828')
    ax.barh(coefs.index, coefs.to_numpy(), color=colors)
    ax.axvline(0, color='k', linewidth=0.8)
    for y_pos, value in enumerate(coefs.to_numpy()):
        ax.text(value + (0.02 if value > 0 else -0.02), y_pos, f'{value:+.2f}',
                va='center', ha='left' if value > 0 else 'right', fontsize=9)
    ax.set_xlim(coefs.min() - 0.3, coefs.max() + 0.3)
    ax.set_xlabel('коэффициент при стандартизированном признаке')
    ax.set_title('Зелёные столбцы голосуют за хорошее вино, красные против')
    st.pyplot(fig)
    plt.close(fig)

    top_for = coefs[coefs > 0].sort_values(ascending=False).head(3)
    top_against = coefs[coefs < 0].head(3)
    st.write('Сильнее всего за хорошее вино голосуют: ' + ', '.join(map(short_name, top_for.index)) + '.')
    st.write('Сильнее всего против: ' + ', '.join(map(short_name, top_against.index)) + '.')
    st.caption(
        'Признаки стандартизированы, поэтому коэффициент показывает, насколько меняется логит вероятности '
        'при росте признака на одно стандартное отклонение, если остальные признаки не меняются. '
        'Плотность, сахар и алкоголь сильно связаны между собой, поэтому их коэффициенты лучше читать вместе.'
    )


st.set_page_config(page_title='Подбор хороших вин', page_icon='🍷', layout='wide')

model = load_model()
config = load_config()
test = load_test()
features = list(model.feature_names_in_)
X_test, y_test = test[features], test['target']
proba_test = test_proba(model, X_test)

st.title('🍷 Подбор хороших вин')
st.caption('Логистическая регрессия по результатам лабораторного анализа. '
           'Хорошее вино: оценка экспертов не ниже 7 из 10.')

st.sidebar.header('Порог решения')
st.sidebar.slider('Вино попадает в подборку, если вероятность не ниже порога', 0.0, 1.0,
                  float(config['threshold']), step=0.001, format='%.3f', key='threshold')
st.sidebar.caption(f'Значение по умолчанию из config.json: {config["threshold"]:.3f}')
threshold = st.session_state['threshold']

tab_single, tab_metrics, tab_batch, tab_coef = st.tabs(
    ['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab_single:
    show_single_wine(model, features, feature_stats(X_test), threshold)
with tab_metrics:
    show_metrics(y_test, proba_test, threshold)
with tab_batch:
    show_batch(model, features, X_test.head(20), st.session_state['threshold'])
with tab_coef:
    show_coefficients(model, features)
