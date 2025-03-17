import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import joblib  # для сохранения модели

# Загрузка данных из файла с развёрнутыми событиями
input_file = "../unified_json/events.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Преобразуем список событий в DataFrame
df = pd.DataFrame(data)

# Функция для объединения даты и времени начала в одну временную метку
def combine_datetime(row):
    try:
        dt_str = row['date'] + " " + row['begin']
        return datetime.strptime(dt_str, "%Y %m %d %H%M")
    except Exception:
        return pd.NaT

df['timestamp'] = df.apply(combine_datetime, axis=1)
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# Извлекаем временные признаки
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday
df['month'] = df['timestamp'].dt.month

# Функция для преобразования строки времени в количество минут от полуночи
def time_str_to_minutes(time_str):
    if pd.isna(time_str):
        return np.nan
    clean = re.sub(r'\D', '', time_str)
    if clean == "":
        return np.nan
    clean = clean.zfill(4)
    try:
        hours = int(clean[:2])
        minutes = int(clean[2:4])
        return hours * 60 + minutes
    except Exception:
        return np.nan

df['begin_mins'] = df['begin'].apply(time_str_to_minutes)
df['end_mins'] = df['end'].apply(time_str_to_minutes)

# Вычисляем длительность события (с учетом перехода через полночь)
def compute_duration(row):
    b = row['begin_mins']
    e = row['end_mins']
    if pd.notna(b) and pd.notna(e):
        diff = e - b
        if diff < 0:
            diff += 1440
        return diff
    return np.nan

df['duration'] = df.apply(compute_duration, axis=1)

# Приводим категориальные признаки к типу "category"
categorical_cols = ['type', 'particulars', 'loc_freq', 'region']
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Удаляем строки с пропущенными значениями в ключевых признаках (begin, end, particulars, type)
df = df.dropna(subset=['begin', 'end', 'particulars', 'type'])

# Функция для создания целевой метки:
# Возвращает 1, если в заданный горизонт (24 часа) после текущей временной метки происходит вспышка типа "XRA"
# и в поле particulars начинается с "M" или "X"
def label_future_strong_flare(ts, horizon_hours, df):
    t_end = ts + timedelta(hours=horizon_hours)
    mask = (df['timestamp'] > ts) & (df['timestamp'] <= t_end) & (df['type'] == "XRA")
    strong = df.loc[mask, 'particulars'].dropna().astype(str)
    return int(any(val.startswith(('M', 'X')) for val in strong))

# Сортируем по timestamp и создаем целевые метки для горизонтов 12, 24 и 48 часов
df = df.sort_values("timestamp").reset_index(drop=True)
df['target_12'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 12, df))
df['target_24'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 24, df))
df['target_48'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 48, df))

# Функция для агрегирования истории событий до текущего момента в заданном окне (в часах)
def add_aggregated_features(df, window_hours):
    agg_count = []
    agg_duration_sum = []
    agg_duration_mean = []
    agg_xra_count = []
    start_idx = 0
    n = len(df)
    for i in range(n):
        current_time = df.loc[i, 'timestamp']
        window_start = current_time - timedelta(hours=window_hours)
        while start_idx < i and df.loc[start_idx, 'timestamp'] < window_start:
            start_idx += 1
        window_df = df.iloc[start_idx:i]
        count = len(window_df)
        agg_count.append(count)
        if count > 0:
            dur_sum = window_df['duration'].sum()
            dur_mean = window_df['duration'].mean()
            xra_count = (window_df['type'] == 'XRA').sum()
        else:
            dur_sum = 0
            dur_mean = 0
            xra_count = 0
        agg_duration_sum.append(dur_sum)
        agg_duration_mean.append(dur_mean)
        agg_xra_count.append(xra_count)
    df[f'agg_count_{window_hours}h'] = agg_count
    df[f'agg_duration_sum_{window_hours}h'] = agg_duration_sum
    df[f'agg_duration_mean_{window_hours}h'] = agg_duration_mean
    df[f'agg_xra_count_{window_hours}h'] = agg_xra_count
    return df

# Добавляем агрегированные признаки за окна 6, 10, 24, 30 и 48 часов
for window in [6, 10, 24, 30, 48]:
    df = add_aggregated_features(df, window)

# Определяем список признаков
agg_features = []
for window in [6, 10, 24, 30, 48]:
    agg_features += [f'agg_count_{window}h', f'agg_duration_sum_{window}h',
                     f'agg_duration_mean_{window}h', f'agg_xra_count_{window}h']
base_features = ['hour', 'weekday', 'month', 'duration'] + categorical_cols
features = base_features + agg_features
cat_features = categorical_cols  # для LightGBM

# Перед обучением удаляем все строки с пропущенными значениями среди признаков и целевой метки target_24
df_clean = df.dropna(subset=features + ['target_24']).reset_index(drop=True)

# Для моделей scikit-learn (например, RandomForest) нужно, чтобы все признаки были числовыми.
# Преобразуем категориальные признаки в числовые коды.
for col in categorical_cols:
    df_clean[col] = df_clean[col].cat.codes

# Хронологически разделяем данные: первые 80% - обучение, последние 20% - тест
split_index = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_index]
test_df = df_clean.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["target_24"]
X_test = test_df[features]
y_test = test_df["target_24"]

# --- Обучение модели LightGBM ---
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42)
lgb_model.fit(X_train, y_train, categorical_feature=cat_features)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_pred_prob_lgb >= 0.5).astype(int)

# --- Обучение модели RandomForest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

# --- Энамблирование: усредняем вероятности обеих моделей ---
y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf) / 2
y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)

# Функция для вычисления и вывода метрик
def print_metrics(y_true, y_pred, y_prob, model_name="Model"):
    print(f"\nМетрики предсказания для {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f} ({accuracy_score(y_true, y_pred)*100:.1f}%)")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

# Выводим метрики для каждой модели и для ансамбля
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, "LightGBM (24 часов)")
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, "Random Forest (24 часов)")
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, "Ensemble (LightGBM + RF)")

# Вывод важности признаков для обеих моделей
print("\nВажность признаков (LightGBM):")
importance_lgb = pd.Series(lgb_model.feature_importances_, index=features)
print(importance_lgb.sort_values(ascending=False))

print("\nВажность признаков (Random Forest):")
importance_rf = pd.Series(rf_model.feature_importances_, index=features)
print(importance_rf.sort_values(ascending=False))

# Сохраняем модели и обучающие данные
lgb_model_filename = "../unified_json/lightgbm_model_target_24.pkl"
rf_model_filename = "../unified_json/random_forest_model_target_24.pkl"
joblib.dump(lgb_model, lgb_model_filename)
joblib.dump(rf_model, rf_model_filename)
print(f"\nМодель LightGBM сохранена в {lgb_model_filename}")
print(f"Модель Random Forest сохранена в {rf_model_filename}")

train_data_filename = "../unified_json/train_data_target_24.csv"
df_clean.to_csv(train_data_filename, index=False)
print(f"Обучающие данные сохранены в {train_data_filename}")

full_data_filename = "../unified_json/full_events_with_targets.csv"
df.to_csv(full_data_filename, index=False)
print(f"Полный DataFrame с прогнозами сохранен в {full_data_filename}")
