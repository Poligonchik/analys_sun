import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
import joblib  # для сохранения модели

# Загрузка данных из файла с развёрнутыми событиями
input_file = "../unified_json/events.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Данные уже представляют список событий
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

# Извлекаем признаки из временной метки
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday
df['month'] = df['timestamp'].dt.month


# Функция для преобразования строк времени в количество минут от полуночи
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


# Функция для создания целевой метки:
# Возвращает 1, если в заданный горизонт (например, 12 часов) после текущей временной метки происходит вспышка типа "XRA"
# при этом в поле particulars должна быть сильная вспышка (начинается с "M" или "X")
def label_future_strong_flare(ts, horizon_hours, df):
    t_end = ts + timedelta(hours=horizon_hours)
    mask = (df['timestamp'] > ts) & (df['timestamp'] <= t_end) & (df['type'] == "XRA")
    strong = df.loc[mask, 'particulars'].dropna().astype(str)
    return int(any(val.startswith(('M', 'X')) for val in strong))


# Создаем метки для горизонтов 12, 24 и 48 часов
df = df.sort_values("timestamp").reset_index(drop=True)
df['target_12'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 12, df))
df['target_24'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 24, df))
df['target_48'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 48, df))


# Функция для агрегирования истории событий до текущего момента в заданном окне (в часах)
def add_aggregated_features(df, window_hours):
    # Будем рассчитывать агрегаты для событий, произошедших строго до текущего
    agg_count = []
    agg_duration_sum = []
    agg_duration_mean = []
    agg_xra_count = []

    start_idx = 0
    n = len(df)
    for i in range(n):
        current_time = df.loc[i, 'timestamp']
        window_start = current_time - timedelta(hours=window_hours)
        # Перемещаем начало окна: поскольку данные отсортированы, увеличиваем start_idx пока событие не попадёт в окно
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


# Добавляем агрегированные признаки для двух окон: 6 часов и 24 часов
df = add_aggregated_features(df, 6)
df = add_aggregated_features(df, 24)

# Теперь расширим список признаков: базовые признаки и агрегированные
agg_features = [f'agg_count_6h', f'agg_duration_sum_6h', f'agg_duration_mean_6h', f'agg_xra_count_6h',
                f'agg_count_24h', f'agg_duration_sum_24h', f'agg_duration_mean_24h', f'agg_xra_count_24h']

features = ['hour', 'weekday', 'month', 'duration'] + categorical_cols + agg_features
cat_features = categorical_cols  # для LightGBM

# Хронологически разделяем данные: первые 80% - обучение, последние 20% - тест
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["target_12"]
X_test = test_df[features]
y_test = test_df["target_12"]

# Обучаем модель LightGBM
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42)
lgb_model.fit(X_train, y_train, categorical_feature=cat_features)

# Оцениваем модель
y_pred_prob = lgb_model.predict_proba(X_test)[:, 1]
y_pred = lgb_model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)

print(f"ROC-AUC (12 часов): {auc:.3f}")
print(f"F1-Score (12 часов): {f1:.3f}")

# Вывод важности признаков
importance = pd.Series(lgb_model.feature_importances_, index=features)
print("Важность признаков:")
print(importance.sort_values(ascending=False))

# Сохраняем модель и данные для дальнейшего использования

# Сохранение модели
model_filename = "../unified_json/lightgbm_model_target_12.pkl"
joblib.dump(lgb_model, model_filename)
print(f"Модель сохранена в {model_filename}")

# Сохранение обучающих данных (например, в CSV)
train_data_filename = "../unified_json/train_data_target_12.csv"
train_df.to_csv(train_data_filename, index=False)
print(f"Обучающие данные сохранены в {train_data_filename}")

# Также можно сохранить весь DataFrame с прогнозами (если нужно)
full_data_filename = "../unified_json/full_events_with_targets.csv"
