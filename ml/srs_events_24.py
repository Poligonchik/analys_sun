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
import xgboost as xgb
import joblib

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
# Функция для извлечения класса вспышки из поля particulars
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
# Загрузка и агрегация данных SRS
#############################################
srs_input = "../unified_json/srs.json"
srs_df = pd.read_json(srs_input)
# Приводим числовые поля к числовому типу
for col in ['Lo', 'Area', 'LL', 'NN']:
    srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce')
# Преобразуем дату (формат "YYYY MM DD")
srs_df['date'] = pd.to_datetime(srs_df['date'], format="%Y %m %d")
# Группируем SRS по дате: число записей и усредняем числовые показатели
srs_agg = srs_df.groupby('date').agg({
    'Nmbr': 'count',
    'Area': 'mean',
    'NN': 'mean',
    'LL': 'mean'
}).rename(columns={
    'Nmbr': 'srs_count',
    'Area': 'srs_area_mean',
    'NN': 'srs_NN_mean',
    'LL': 'srs_LL_mean'
}).reset_index()


#############################################
# Загрузка и агрегация данных событий
#############################################
events_input = "../unified_json/events.json"
events_df = pd.read_json(events_input)
# Преобразуем дату (формат "YYYY MM DD")
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")

# Если ключ 'particulars' отсутствует – добавляем его как пустую строку
if 'particulars' not in events_df.columns:
    events_df['particulars'] = ""

# Вычисляем класс вспышки для каждого события
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)

# Определяем "сильное" событие: если type == "XRA" и flare_class в ["M", "X"]
def is_strong_row(row):
    event_type = str(row.get('type', "")).strip()
    flare_class = str(row.get('flare_class', "None")).strip()
    return 1 if event_type == "XRA" and flare_class in ["M", "X"] else 0

events_df['strong'] = events_df.apply(is_strong_row, axis=1)

# Агрегируем по дате: общее число событий и число сильных событий
events_agg = events_df.groupby('date').agg(
    events_count=('type', 'count'),
    strong_events=('strong', 'sum')
).reset_index()

# Агрегируем по классам вспышек (A, B, C, M, X)
flare_daily = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
for cl in ["A", "B", "C", "M", "X"]:
    if cl not in flare_daily.columns:
        flare_daily[cl] = 0

events_final = pd.merge(events_agg, flare_daily, on='date', how='outer')
events_final[['events_count','strong_events','A','B','C','M','X']] = \
    events_final[['events_count','strong_events','A','B','C','M','X']].fillna(0)


#############################################
# Объединение данных SRS и событий по дате
#############################################
# При outer-merge, если по дате нет SRS, поля srs_* будут 0.
merged = pd.merge(srs_agg, events_final, on='date', how='outer')

# Если по дате нет SRS, можно оставить srs_count=0 – при расчётах отношений условие будет обрабатывать этот случай.
cols_to_fill = ['srs_count','srs_area_mean','srs_NN_mean','srs_LL_mean',
                'events_count','strong_events','A','B','C','M','X']
merged[cols_to_fill] = merged[cols_to_fill].fillna(0)
merged = merged.sort_values('date').reset_index(drop=True)

# Рассчитываем скользящие суммы по событиям за 24 и 48 часов (так как данные агрегированы по дням)
merged['events_24h'] = merged['events_count'].rolling(window=1, min_periods=1).sum()
merged['strong_24h'] = merged['strong_events'].rolling(window=1, min_periods=1).sum()
merged['A_24h'] = merged['A'].rolling(window=1, min_periods=1).sum()
merged['B_24h'] = merged['B'].rolling(window=1, min_periods=1).sum()
merged['C_24h'] = merged['C'].rolling(window=1, min_periods=1).sum()
merged['M_24h'] = merged['M'].rolling(window=1, min_periods=1).sum()
merged['X_24h'] = merged['X'].rolling(window=1, min_periods=1).sum()

merged['events_48h'] = merged['events_count'].rolling(window=2, min_periods=1).sum()
merged['strong_48h'] = merged['strong_events'].rolling(window=2, min_periods=1).sum()
merged['A_48h'] = merged['A'].rolling(window=2, min_periods=1).sum()
merged['B_48h'] = merged['B'].rolling(window=2, min_periods=1).sum()
merged['C_48h'] = merged['C'].rolling(window=2, min_periods=1).sum()
merged['M_48h'] = merged['M'].rolling(window=2, min_periods=1).sum()
merged['X_48h'] = merged['X'].rolling(window=2, min_periods=1).sum()

#############################################
# Добавление дополнительных комбинационных признаков
#############################################
# Если SRS данных нет (srs_count==0), то задаём отношение равным events_count или strong_events
merged['ratio_events_to_srs'] = np.where(merged['srs_count'] > 0,
                                           merged['events_count'] / merged['srs_count'],
                                           merged['events_count'])
merged['ratio_strong_to_srs'] = np.where(merged['srs_count'] > 0,
                                           merged['strong_events'] / merged['srs_count'],
                                           merged['strong_events'])
merged['srs_events_interaction'] = merged['srs_count'] * merged['events_count']
merged['srs_area_strong_interaction'] = merged['srs_area_mean'] * merged['strong_events']
merged['srs_NN_ratio'] = merged['srs_NN_mean'] / (merged['srs_LL_mean'] + 1e-5)

# Отношения для каждого класса вспышек (относительно общего числа событий)
merged['ratio_A'] = merged['A'] / (merged['events_count'] + 1e-5)
merged['ratio_B'] = merged['B'] / (merged['events_count'] + 1e-5)
merged['ratio_C'] = merged['C'] / (merged['events_count'] + 1e-5)
merged['ratio_M'] = merged['M'] / (merged['events_count'] + 1e-5)
merged['ratio_X'] = merged['X'] / (merged['events_count'] + 1e-5)

# Отношение скользящих сумм за 48h к 24h (динамика активности)
merged['ratio_events_48h_to_24h'] = merged['events_48h'] / (merged['events_24h'] + 1e-5)
merged['ratio_strong_48h_to_24h'] = merged['strong_48h'] / (merged['strong_24h'] + 1e-5)
merged['diff_events_48h_24h'] = merged['events_48h'] - merged['events_24h']

#############################################
# Формирование таргета
#############################################
# Таргет: наличие хотя бы одного сильного события (strong_events > 0) в следующий календарный день
merged['target_24'] = merged['strong_events'].shift(-1).fillna(0).apply(lambda x: 1 if x > 0 else 0)

#############################################
# Формирование финального датасета для моделирования
#############################################
features = ['srs_count', 'srs_area_mean', 'srs_NN_mean', 'srs_LL_mean',
            'events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X',
            'events_24h', 'strong_24h', 'A_24h', 'B_24h', 'C_24h', 'M_24h', 'X_24h',
            'events_48h', 'strong_48h', 'A_48h', 'B_48h', 'C_48h', 'M_48h', 'X_48h',
            'ratio_events_to_srs', 'ratio_strong_to_srs', 'srs_events_interaction',
            'srs_area_strong_interaction', 'srs_NN_ratio', 'ratio_A', 'ratio_B', 'ratio_C', 'ratio_M', 'ratio_X',
            'ratio_events_48h_to_24h', 'ratio_strong_48h_to_24h', 'diff_events_48h_24h']

data_model = merged.dropna(subset=['target_24']).reset_index(drop=True)

# Разделение данных по хронологии: 80% для обучения, 20% для теста
split_idx = int(len(data_model) * 0.8)
train_data = data_model.iloc[:split_idx]
test_data = data_model.iloc[split_idx:]

X_train = train_data[features]
y_train = train_data['target_24']
X_test = test_data[features]
y_test = test_data['target_24']

#############################################
# Обучение трёх моделей
#############################################
# 1. LightGBM
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_pred_prob_lgb >= 0.5).astype(int)
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, model_name="LightGBM (24h)")

# 2. RandomForest (с балансировкой классов)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, model_name="RandomForest (24h)")

# 3. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, scale_pos_weight=1, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_prob_xgb >= 0.5).astype(int)
print_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, model_name="XGBoost (24h)")

# Энсамблирование: усреднение вероятностей
y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf + y_pred_prob_xgb) / 3
y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, model_name="Ensemble (LGBM+RF+XGB)")

#############################################
# Сохранение датасета и моделей
#############################################
data_model.to_csv("merged_events_srs.csv", index=False)
joblib.dump(lgb_model, "lgb_model_merged.pkl")
joblib.dump(rf_model, "rf_model_merged.pkl")
joblib.dump(xgb_model, "xgb_model_merged.pkl")

print("Финальный датасет сохранен в 'merged_events_srs.csv'")
print("Модель LightGBM сохранена в 'lgb_model_merged.pkl'")
print("Модель RandomForest сохранена в 'rf_model_merged.pkl'")
print("Модель XGBoost сохранена в 'xgb_model_merged.pkl'")
