import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import joblib


#############################################
# Функция для оценки многоклассовых прогнозов
#############################################
def evaluate_multiclass(y_true, y_pred, intensity_map):
    # Точное совпадение
    exact_accuracy = np.mean(y_true == y_pred)

    # Допустимое совпадение (±1 по интенсивности)
    tol_correct = []
    for true_val, pred_val in zip(y_true, y_pred):
        diff = abs(intensity_map.get(true_val, 0) - intensity_map.get(pred_val, 0))
        tol_correct.append(1 if diff <= 1 else 0)
    tolerance_accuracy = np.mean(tol_correct)

    print(f"Exact accuracy: {exact_accuracy:.3f}")
    print(f"Tolerance (±1) accuracy: {tolerance_accuracy:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


#############################################
# Функция извлечения класса вспышки из particulars
#############################################
def extract_flare_class(particulars):
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"


#############################################
# Функция для определения будущей вспышки (на ближайшие 24 часов)
# Среди событий типа XRA выбирается событие с наивысшей интенсивностью
#############################################
def label_future_flare(ts, horizon_hours, df):
    t_end = ts + timedelta(hours=horizon_hours)
    mask = (df['timestamp'] > ts) & (df['timestamp'] <= t_end) & (df['type'] == "XRA")
    candidate_events = df.loc[mask].copy()
    if candidate_events.empty:
        return "None"
    candidate_events['flare_class'] = candidate_events['particulars'].apply(extract_flare_class)
    intensity = {"None": 0, "A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
    candidate_events['intensity'] = candidate_events['flare_class'].apply(lambda x: intensity.get(x, 0))
    best_idx = candidate_events['intensity'].idxmax()
    best = candidate_events.loc[best_idx]
    return best['flare_class']


#############################################
# Загрузка и подготовка данных из events.json
#############################################
input_file = "../../result_json/events.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)


# Объединяем дату и время начала события в один timestamp
def combine_datetime(row):
    try:
        dt_str = row['date'] + " " + row['begin']
        return datetime.strptime(dt_str, "%Y %m %d %H%M")
    except Exception:
        return pd.NaT


df['timestamp'] = df.apply(combine_datetime, axis=1)
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# Временные признаки
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday
df['month'] = df['timestamp'].dt.month


# Преобразование времени (begin, end) в минуты от полуночи
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


# Вычисление длительности события (в минутах)
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

# Приводим текстовые признаки к типу "category"
for col in ['type', 'particulars', 'loc_freq', 'region']:
    df[col] = df[col].astype("category")

# Удаляем строки с пропусками в ключевых признаках
df = df.dropna(subset=['begin', 'end', 'particulars', 'type'])

#############################################
# Формирование целевого признака: прогноз класса вспышки в ближайшие 24 часов
#############################################
df = df.sort_values("timestamp").reset_index(drop=True)
df['target_24_multi'] = df['timestamp'].apply(lambda x: label_future_flare(x, 24, df))
df['target_24_multi'] = pd.Categorical(df['target_24_multi'],
                                       categories=["None", "A", "B", "C", "M", "X"],
                                       ordered=True)

# Добавляем колонку flare_class для формирования признаков по истории
df['flare_class'] = df['particulars'].apply(extract_flare_class)


#############################################
# Функция для добавления признаков по истории вспышек за последние 24 часов
#############################################
def add_flare_type_features(df, window_hours):
    last_flare = []
    count_A = []
    count_B = []
    count_C = []
    count_M = []
    count_X = []
    total_count = []
    for current_time in df['timestamp']:
        window_start = current_time - timedelta(hours=window_hours)
        window_df = df[(df['timestamp'] >= window_start) & (df['timestamp'] < current_time)]
        # Если в окне нет событий, запишем "None"
        if not window_df.empty:
            last_flare.append(window_df.iloc[-1]['flare_class'])
        else:
            last_flare.append("None")
        count_A.append((window_df['flare_class'] == "A").sum())
        count_B.append((window_df['flare_class'] == "B").sum())
        count_C.append((window_df['flare_class'] == "C").sum())
        count_M.append((window_df['flare_class'] == "M").sum())
        count_X.append((window_df['flare_class'] == "X").sum())
        total_count.append(len(window_df))
    df[f'last_flare_{window_hours}h'] = last_flare
    df[f'count_A_{window_hours}h'] = count_A
    df[f'count_B_{window_hours}h'] = count_B
    df[f'count_C_{window_hours}h'] = count_C
    df[f'count_M_{window_hours}h'] = count_M
    df[f'count_X_{window_hours}h'] = count_X
    df[f'total_flare_count_{window_hours}h'] = total_count
    df[f'ratio_MX_{window_hours}h'] = (np.array(count_M) + np.array(count_X)) / (np.array(total_count) + 1e-5)
    return df


# Добавляем признаки за последние 24 часов
df = add_flare_type_features(df, 24)
# Преобразуем категориальный признак last_flare_24h в числовой код
df['last_flare_24h_code'] = df['last_flare_24h'].astype('category').cat.codes

#############################################
# Формирование финального датасета для моделирования
#############################################
features_to_use = ['hour', 'weekday', 'month', 'duration',
                   'last_flare_24h_code', 'count_A_24h', 'count_B_24h',
                   'count_C_24h', 'count_M_24h', 'count_X_24h',
                   'total_flare_count_24h', 'ratio_MX_24h']
target = 'target_24_multi'

df_model = df[features_to_use + [target]].dropna().reset_index(drop=True)

# Хронологическое разделение: 80% обучение, 20% тест
split_index = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_index]
test_df = df_model.iloc[split_index:]

X_train = train_df[features_to_use]
y_train = train_df[target]
X_test = test_df[features_to_use]
y_test = test_df[target]

#############################################
# Обучение моделей (многоклассовое прогнозирование)
#############################################
# LightGBM: objective='multiclass'
lgb_model = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train.cat.codes)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)

# RandomForest (многоклассовая)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train.cat.codes)
y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)

# XGBoost: objective='multi:softmax'
num_classes = len(y_train.cat.categories)
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes,
                              use_label_encoder=False, eval_metric='mlogloss',
                              random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train.cat.codes)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)

#############################################
# Оценка моделей с учётом двух критериев:
# 1) Точное совпадение
# 2) Допустимое совпадение (±1 по интенсивности)
#############################################
intensity_map = {"None": 0, "A": 1, "B": 2, "C": 3, "M": 4, "X": 5}
categories = y_train.cat.categories.tolist()

# Используем .cat.codes для получения числовых кодов целевой переменной
y_true_labels = pd.Categorical.from_codes(y_test.cat.codes, categories=categories, ordered=True)
y_pred_labels_lgb = pd.Categorical.from_codes(y_pred_lgb, categories=categories, ordered=True)
y_pred_labels_rf = pd.Categorical.from_codes(y_pred_rf, categories=categories, ordered=True)
y_pred_labels_xgb = pd.Categorical.from_codes(y_pred_xgb, categories=categories, ordered=True)

print("\nLightGBM:")
evaluate_multiclass(y_true_labels, y_pred_labels_lgb, intensity_map)
print("\nRandomForest:")
evaluate_multiclass(y_true_labels, y_pred_labels_rf, intensity_map)
print("\nXGBoost:")
evaluate_multiclass(y_true_labels, y_pred_labels_xgb, intensity_map)

#############################################
# Сохранение моделей и данных
#############################################
lgb_model_filename = "../models/lightgbm_model_multi_target_24.pkl"
rf_model_filename = "../models/random_forest_model_multi_target_24.pkl"
xgb_model_filename = "../models/xgboost_model_multi_target_24.pkl"

joblib.dump(lgb_model, lgb_model_filename)
joblib.dump(rf_model, rf_model_filename)
joblib.dump(xgb_model, xgb_model_filename)

print(f"\nМодель LightGBM сохранена в {lgb_model_filename}")
print(f"Модель RandomForest сохранена в {rf_model_filename}")
print(f"Модель XGBoost сохранена в {xgb_model_filename}")

train_data_filename = "../models/train_data_multi_target_24.csv"
df_model.to_csv(train_data_filename, index=False)
print(f"Обучающие данные сохранены в {train_data_filename}")

full_data_filename = "../result_json/full_events_with_multi_target.csv"
df.to_csv(full_data_filename, index=False)
print(f"Полный DataFrame с прогнозами сохранен в {full_data_filename}")
