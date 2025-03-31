import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import joblib
import xgboost as xgb

# Загрузка данных
input_file = "../result_json/events.json"
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

# Преобразование строки времени в количество минут от полуночи
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

# Функция создания бинарного таргета для сильных вспышек
# (возвращает 1, если в течение horizon часов после ts происходит событие типа XRA,
# а в поле particulars начинается с "M" или "X")
def label_future_strong_flare(ts, horizon_hours, df):
    t_end = ts + timedelta(hours=horizon_hours)
    mask = (df['timestamp'] > ts) & (df['timestamp'] <= t_end) & (df['type'] == "XRA")
    strong = df.loc[mask, 'particulars'].dropna().astype(str)
    return int(any(val.startswith(('M', 'X')) for val in strong))

df = df.sort_values("timestamp").reset_index(drop=True)
# Для новой модели мы используем горизонт 48 часов:
df['target_48'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 48, df))
# (Также оставлены расчёты для 12 и 24 часов, если они понадобятся)
df['target_12'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 12, df))
df['target_24'] = df['timestamp'].apply(lambda x: label_future_strong_flare(x, 24, df))

# Функция агрегирования событий за заданное окно (в часах)
def add_aggregated_features(df, window_hours):
    agg_count = []
    agg_duration_sum = []
    agg_duration_mean = []
    agg_duration_var = []
    agg_duration_median = []
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
            dur_values = window_df['duration'].dropna()
            dur_sum = dur_values.sum()
            dur_mean = dur_values.mean()
            dur_var = dur_values.var() if len(dur_values) > 1 else 0
            dur_median = dur_values.median()
            xra_count = (window_df['type'] == 'XRA').sum()
        else:
            dur_sum, dur_mean, dur_var, dur_median, xra_count = 0, 0, 0, 0, 0
        agg_duration_sum.append(dur_sum)
        agg_duration_mean.append(dur_mean)
        agg_duration_var.append(dur_var)
        agg_duration_median.append(dur_median)
        agg_xra_count.append(xra_count)
    df[f'agg_count_{window_hours}h'] = agg_count
    df[f'agg_duration_sum_{window_hours}h'] = agg_duration_sum
    df[f'agg_duration_mean_{window_hours}h'] = agg_duration_mean
    df[f'agg_duration_var_{window_hours}h'] = agg_duration_var
    df[f'agg_duration_median_{window_hours}h'] = agg_duration_median
    df[f'agg_xra_count_{window_hours}h'] = agg_xra_count
    return df

# Добавляем агрегированные признаки за окна 6, 10, 24, 30 и 48 часов
for window in [6, 10, 24, 30, 48]:
    df = add_aggregated_features(df, window)

# Добавляем составные признаки
df['ratio_count_24_48'] = df[f'agg_count_24h'] / (df[f'agg_count_48h'] + 1e-5)
df['diff_mean_24_48'] = df[f'agg_duration_mean_24h'] - df[f'agg_duration_mean_48h']

# Определяем список признаков для модели
agg_features = []
for window in [6, 10, 24, 30, 48]:
    agg_features += [f'agg_count_{window}h', f'agg_duration_sum_{window}h',
                     f'agg_duration_mean_{window}h', f'agg_duration_var_{window}h',
                     f'agg_duration_median_{window}h', f'agg_xra_count_{window}h']
base_features = ['hour', 'weekday', 'month', 'duration'] + categorical_cols
features = base_features + agg_features + ['ratio_count_24_48', 'diff_mean_24_48']
cat_features = categorical_cols  # для LightGBM

# Удаляем строки с пропусками в признаках и в таргете target_48
df_clean = df.dropna(subset=features + ['target_48']).reset_index(drop=True)

# Преобразуем категориальные признаки в числовые коды для scikit-learn
for col in categorical_cols:
    df_clean[col] = df_clean[col].cat.codes

# Хронологическое разделение: 80% обучение, 20% тест
split_index = int(len(df_clean) * 0.8)
train_df = df_clean.iloc[:split_index]
test_df = df_clean.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["target_48"]
X_test = test_df[features]
y_test = test_df["target_48"]

#############################################
# Обучение моделей для бинарной классификации (target_48)
#############################################

# LightGBM с использованием всех ядер
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train, categorical_feature=cat_features)
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

# Стэкинг-ансамблирование (без гиперпараметрической оптимизации)
base_estimators = [
    ('lgb', lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1))
]
meta_estimator = LogisticRegression(random_state=42, max_iter=1000)
stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator,
                                     cv=5, n_jobs=-1)
stacking_model.fit(X_train, y_train)
y_pred_prob_stack = stacking_model.predict_proba(X_test)[:, 1]
y_pred_stack = (y_pred_prob_stack >= 0.5).astype(int)

# Простое усреднение вероятностей (энсамблирование)
y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf + y_pred_prob_xgb) / 3
y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)

# Дополнительный блок: Взвешенное голосование
w_lgb = 0.35
w_rf = 0.33
w_xgb = 0.32
y_pred_prob_weighted = (w_lgb * y_pred_prob_lgb + w_rf * y_pred_prob_rf + w_xgb * y_pred_prob_xgb)
y_pred_weighted = (y_pred_prob_weighted >= 0.5).astype(int)

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
# Вывод метрик для моделей (предсказание вспышки в течение 48 часов)
#############################################
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, "LightGBM (48 часов, бинарный)")
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, "Random Forest (48 часов, бинарный)")
print_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, "XGBoost (48 часов, бинарный)")
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, "Ensemble (усреднение LGBM+RF+XGB)")
print_metrics(y_test, y_pred_stack, y_pred_prob_stack, "Stacking (без GridSearchCV)")
print_metrics(y_test, y_pred_weighted, y_pred_prob_weighted, "Weighted Voting Ensemble (LGBM+RF+XGB)")

#############################################
# Сохранение моделей и обучающих данных
#############################################
lgb_model_filename = "../models/lightgbm_model_target_48.pkl"
rf_model_filename = "../models/random_forest_model_target_48.pkl"
xgb_model_filename = "../models/xgboost_model_target_48.pkl"
stack_model_filename = "../models/stacking_model_target_48.pkl"

joblib.dump(lgb_model, lgb_model_filename)
joblib.dump(rf_model, rf_model_filename)
joblib.dump(xgb_model, xgb_model_filename)
joblib.dump(stacking_model, stack_model_filename)

print(f"\nМодель LightGBM (48 часов, бинарная) сохранена в {lgb_model_filename}")
print(f"Модель Random Forest (48 часов, бинарная) сохранена в {rf_model_filename}")
print(f"Модель XGBoost (48 часов, бинарная) сохранена в {xgb_model_filename}")
print(f"Модель Stacking (48 часов, бинарная) сохранена в {stack_model_filename}")

train_data_filename = "../models/train_data_target_48.csv"
df_clean.to_csv(train_data_filename, index=False)
print(f"Обучающие данные сохранены в {train_data_filename}")

full_data_filename = "../models/full_events_with_targets.csv"
df.to_csv(full_data_filename, index=False)
print(f"Полный DataFrame с прогнозами сохранен в {full_data_filename}")
