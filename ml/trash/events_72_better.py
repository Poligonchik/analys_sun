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
import joblib
import xgboost as xgb

#############################################
# Загрузка и подготовка данных
#############################################

# Загрузка данных
input_file = "../../result_json/events.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Функция объединения даты и времени начала
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

# Преобразование времени в минуты от полуночи
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

# Вычисляем длительность события
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

# Удаляем строки с пропусками в ключевых признаках
df = df.dropna(subset=['begin', 'end', 'particulars', 'type'])

#############################################
# Таргет: предсказание сильной вспышки в ближайшие 72 часа
#############################################

def label_future_strong_flare(ts, horizon_hours, df):
    t_end = ts + timedelta(hours=horizon_hours)
    mask = (df['timestamp'] > ts) & (df['timestamp'] <= t_end) & (df['type'] == "XRA")
    strong = df.loc[mask, 'particulars'].dropna().astype(str)
    # Если среди вспышек в окне есть те, что начинаются с "M" или "X", считаем сильной
    return int(any(val.startswith(('M', 'X')) for val in strong))

df = df.sort_values("timestamp").reset_index(drop=True)
df['target_72'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 72, df))

#############################################
# Новые признаки на основе истории вспышек за последние 24 часа
#############################################

def extract_flare_class(particulars):
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"

df['flare_class'] = df['particulars'].apply(extract_flare_class)

def add_flare_type_features(df, window_hours):
    last_flare = []
    count_A = []
    count_B = []
    count_C = []
    count_M = []
    count_X = []
    total_count = []

    for i, current_time in enumerate(df['timestamp']):
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
    # Доля вспышек M или X среди всех вспышек за окно
    df[f'ratio_MX_{window_hours}h'] = (np.array(count_M) + np.array(count_X)) / (np.array(total_count) + 1e-5)
    return df

# Добавляем признаки о вспышках за последние 24 часа
df = add_flare_type_features(df, 24)

#############################################
# Формирование финального датасета для моделирования
#############################################

features_to_use = [
    'hour', 'weekday', 'month', 'duration',
    'last_flare_24h', 'count_A_24h', 'count_B_24h',
    'count_C_24h', 'count_M_24h', 'count_X_24h',
    'total_flare_count_24h', 'ratio_MX_24h'
]

df_model = df.copy()
# Удаляем ненужные для модели колонки
df_model = df_model.drop(columns=[
    'begin', 'end', 'particulars', 'loc_freq',
    'region', 'flare_class'
])

# Оставляем таргет только для 72 часов
df_model['target_72'] = df['target_72']

df_final = df_model[features_to_use + ['target_72']].dropna().reset_index(drop=True)

# Хронологическое разделение: 80% обучение, 20% тест
split_index = int(len(df_final) * 0.8)
train_df = df_final.iloc[:split_index]
test_df = df_final.iloc[split_index:]

X_train = train_df[features_to_use]
y_train = train_df["target_72"]
X_test = test_df[features_to_use]
y_test = test_df["target_72"]

# Преобразуем last_flare_24h в числовой код
X_train['last_flare_24h'] = X_train['last_flare_24h'].astype('category').cat.codes
X_test['last_flare_24h'] = X_test['last_flare_24h'].astype('category').cat.codes

#############################################
# Обучение моделей (72 часа)
#############################################

# LightGBM
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_pred_prob_lgb >= 0.5).astype(int)

# RandomForest (с балансировкой классов)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)

# XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, scale_pos_weight=1, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_prob_xgb >= 0.5).astype(int)

# Усреднение вероятностей (LightGBM + RF + XGB)
y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf + y_pred_prob_xgb) / 3
y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)

#############################################
# Функция для вывода метрик
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
# Вывод метрик для моделей (72 часа)
#############################################
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, "LightGBM (72 часов, бинарный)")
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, "Random Forest (72 часов, бинарный)")
print_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, "XGBoost (72 часов, бинарный)")
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, "Ensemble (усреднение LGBM+RF+XGB)")

#############################################
# Сохранение моделей и данных
#############################################
lgb_model_filename = "../models/lightgbm_model_target_72.pkl"
rf_model_filename = "../models/random_forest_model_target_72.pkl"
xgb_model_filename = "../models/xgboost_model_target_72.pkl"

joblib.dump(lgb_model, lgb_model_filename)
joblib.dump(rf_model, rf_model_filename)
joblib.dump(xgb_model, xgb_model_filename)

print(f"\nМодель LightGBM сохранена в {lgb_model_filename}")
print(f"Модель Random Forest сохранена в {rf_model_filename}")
print(f"Модель XGBoost сохранена в {xgb_model_filename}")

train_data_filename = "../models/train_data_target_72.csv"
df_final.to_csv(train_data_filename, index=False)
print(f"Обучающие данные сохранены в {train_data_filename}")

full_data_filename = "../../result_json/full_events_with_targets.csv"
df.to_csv(full_data_filename, index=False)
print(f"Полный DataFrame с прогнозами сохранен в {full_data_filename}")
