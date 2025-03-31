import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, roc_auc_score)
import xgboost as xgb
import joblib

#############################################
# Функция для вывода метрик (бинарная классификация)
#############################################
def print_metrics(y_true, y_pred, y_prob, model_name="Model"):
    print(f"\nМетрики предсказания для {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f} ({accuracy_score(y_true, y_pred)*100:.1f}%)")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.3f}")
    try:
        print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.3f}")
    except Exception:
        pass
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
# Функция для определения наличия сильной вспышки (M или X) строго в интервале (24,48] часов
#############################################
def label_strong_flare_day2(ts, df):
    t_start = ts + timedelta(hours=24)
    t_end = ts + timedelta(hours=48)
    mask = (df['timestamp'] > t_start) & (df['timestamp'] <= t_end) & (df['type'] == "XRA")
    strong = df.loc[mask, 'particulars'].dropna().astype(str)
    return int(any(val.startswith(('M', 'X')) for val in strong))

#############################################
# Функция для добавления истории вспышек в произвольном временном окне
# Параметры lower_hours и upper_hours задают нижнюю и верхнюю границу окна относительно текущего момента.
# Например, для окна 24–48 часов: lower_hours=24, upper_hours=48.
# Функция считает последний класс вспышки, количество вспышек каждого класса, общее число и отношение (M+X)/total.
#############################################
def add_flare_history_features(df, lower_hours, upper_hours, prefix):
    last_flare = []
    count_A = []
    count_B = []
    count_C = []
    count_M = []
    count_X = []
    total_count = []
    for current_time in df['timestamp']:
        window_start = current_time - timedelta(hours=upper_hours)
        window_end = current_time - timedelta(hours=lower_hours)
        window_df = df[(df['timestamp'] >= window_start) & (df['timestamp'] < window_end)]
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
    df[f'{prefix}_last_flare'] = last_flare
    df[f'{prefix}_count_A'] = count_A
    df[f'{prefix}_count_B'] = count_B
    df[f'{prefix}_count_C'] = count_C
    df[f'{prefix}_count_M'] = count_M
    df[f'{prefix}_count_X'] = count_X
    df[f'{prefix}_total_flare_count'] = total_count
    df[f'{prefix}_ratio_MX'] = (np.array(count_M) + np.array(count_X)) / (np.array(total_count) + 1e-5)
    return df

#############################################
# Загрузка и подготовка данных из events.json
#############################################
input_file = "../result_json/events.json"
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

df = df.dropna(subset=['begin', 'end', 'particulars', 'type'])

#############################################
# Формирование целевого признака: прогноз того, что сильная вспышка (M или X) произойдёт строго на второй день
# (то есть в интервале (24,48] часов)
#############################################
df = df.sort_values("timestamp").reset_index(drop=True)
df['target_day2'] = df['timestamp'].apply(lambda x: label_strong_flare_day2(x, df))
# target_day2 – 1, если хотя бы одна вспышка с "M" или "X" происходит в (24,48] часов, иначе 0

#############################################
# Добавляем колонку flare_class для истории
#############################################
df['flare_class'] = df['particulars'].apply(extract_flare_class)

#############################################
# Добавляем признаки по истории вспышек
#############################################
# История за последние 24 часов: окно (0,24)
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

df = add_flare_type_features(df, 24)

# Добавляем историю вспышек за промежуток 24-48 часов
df = add_flare_history_features(df, lower_hours=24, upper_hours=48, prefix="hist_24_48")
# Добавляем историю вспышек за последние 48 часов (0–48 часов)
df = add_flare_history_features(df, lower_hours=0, upper_hours=48, prefix="hist_48")

# Для удобства преобразуем категориальные признаки последних вспышек в числовой код
df['last_flare_24h_code'] = df['last_flare_24h'].astype('category').cat.codes
df['hist_24_48_last_flare_code'] = df['hist_24_48_last_flare'].astype('category').cat.codes
df['hist_48_last_flare_code'] = df['hist_48_last_flare'].astype('category').cat.codes

#############################################
# Формирование финального датасета для моделирования
#############################################
# Выбираем базовые временные признаки, длительность и признаки истории вспышек за разные окна
features_to_use = [
    'hour', 'weekday', 'month', 'duration',
    'last_flare_24h_code', 'count_A_24h', 'count_B_24h',
    'count_C_24h', 'count_M_24h', 'count_X_24h',
    'total_flare_count_24h', 'ratio_MX_24h',
    'hist_24_48_last_flare_code', 'hist_24_48_count_A', 'hist_24_48_count_B',
    'hist_24_48_count_C', 'hist_24_48_count_M', 'hist_24_48_count_X',
    'hist_24_48_total_flare_count', 'hist_24_48_ratio_MX',
    'hist_48_last_flare_code', 'hist_48_count_A', 'hist_48_count_B',
    'hist_48_count_C', 'hist_48_count_M', 'hist_48_count_X',
    'hist_48_total_flare_count', 'hist_48_ratio_MX'
]
target = 'target_day2'  # бинарный: 1 если сильная вспышка (M или X) происходит в (24,48] часов, иначе 0

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
# Обучение моделей (бинарное прогнозирование)
#############################################
# LightGBM
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_pred_prob_lgb >= 0.5).astype(int)

# RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_prob_xgb >= 0.5).astype(int)

# Энсамблирование – простое усреднение вероятностей
y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf + y_pred_prob_xgb) / 3
y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)

#############################################
# Вывод метрик
#############################################
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, "LightGBM (24-48 часов)")
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, "RandomForest (24-48 часов)")
print_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, "XGBoost (24-48 часов)")
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, "Ensemble (усреднение LGBM+RF+XGB)")

#############################################
# Сохранение моделей и данных
#############################################
lgb_model_filename = "../models/lightgbm_model_target_day2.pkl"
rf_model_filename = "../models/random_forest_model_target_day2.pkl"
xgb_model_filename = "../models/xgboost_model_target_day2.pkl"

joblib.dump(lgb_model, lgb_model_filename)
joblib.dump(rf_model, rf_model_filename)
joblib.dump(xgb_model, xgb_model_filename)

print(f"\nМодель LightGBM сохранена в {lgb_model_filename}")
print(f"Модель RandomForest сохранена в {rf_model_filename}")
print(f"Модель XGBoost сохранена в {xgb_model_filename}")

train_data_filename = "../models/train_data_target_day2.csv"
df_model.to_csv(train_data_filename, index=False)
print(f"Обучающие данные сохранены в {train_data_filename}")

full_data_filename = "../result_json/full_events_with_targets_day2.csv"
df.to_csv(full_data_filename, index=False)
print(f"Полный DataFrame с прогнозами сохранен в {full_data_filename}")
