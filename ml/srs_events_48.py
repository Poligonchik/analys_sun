import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#############################################
# 1) Функции вспомогательные
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

def extract_flare_class(particulars):
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"

MAG_TYPE_MAP = {
    'ALPHA': 1,
    'BETA': 1,
    'BETA-GAMMA': 2,
    'GAMMA': 2,
    'BETA-GAMMA-DELTA': 3,
    'BETA-DELTA': 3,
    'GAMMA-DELTA': 3,
    'DELTA': 3
}
def encode_mag_type(mag_str):
    if not isinstance(mag_str, str):
        return 0
    return MAG_TYPE_MAP.get(mag_str.upper().strip(), 0)

def is_strong_row(row):
    evt_type = str(row.get('type', "")).strip()
    cl = str(row.get('flare_class', "None")).strip()
    return 1 if (evt_type == "XRA" and cl in ["M", "X"]) else 0

#############################################
# 2) Чтение и обработка SRS данных
#############################################
srs_input = "../result_json/srs.json"
srs_df = pd.read_json(srs_input)
srs_df['date'] = pd.to_datetime(srs_df['date'], format="%Y %m %d")

# Приводим числовые поля к числовому типу
for col in ['Lo', 'Area', 'LL', 'NN']:
    srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce').fillna(0)

# Кодирование магнитного типа и определение "сложности"
srs_df['mag_code'] = srs_df['Mag_Type'].apply(encode_mag_type)
srs_df['is_complex'] = np.where(srs_df['mag_code'] >= 2, 1, 0)

# 2.1) Суммарная площадь сложных регионов
complex_mask = (srs_df['is_complex'] == 1)
complex_df = srs_df[complex_mask]
sum_complex_area = complex_df.groupby('date')['Area'].sum().rename("sum_complex_area")

# 2.2) Агрегация SRS по дате
srs_agg = srs_df.groupby('date').agg({
    'Nmbr': 'count',
    'Area': 'mean',
    'NN': 'mean',
    'LL': 'mean',
    'mag_code': 'sum',
    'is_complex': 'sum'
}).reset_index().rename(columns={
    'Nmbr': 'srs_count',
    'Area': 'srs_area_mean',
    'NN': 'srs_NN_mean',
    'LL': 'srs_LL_mean',
    'mag_code': 'mag_code_sum',
    'is_complex': 'complex_count'
})

# Присоединяем sum_complex_area
srs_agg = pd.merge(srs_agg, sum_complex_area, on='date', how='left')
srs_agg['sum_complex_area'] = srs_agg['sum_complex_area'].fillna(0)

#############################################
# 3) Чтение и обработка EVENTS данных
#############################################
events_input = "../result_json/events.json"
events_df = pd.read_json(events_input)
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")
events_df['particulars'] = events_df['particulars'].fillna("")
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)
events_df['strong'] = events_df.apply(is_strong_row, axis=1)

# Агрегация по дате: общее число событий и число сильных событий
evt_agg = events_df.groupby('date').agg(
    events_count=('type', 'count'),
    strong_events=('strong', 'sum')
).reset_index()

# Агрегация по классам вспышек
flare_daily = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
for cl in ["A", "B", "C", "M", "X"]:
    if cl not in flare_daily.columns:
        flare_daily[cl] = 0

events_final = pd.merge(evt_agg, flare_daily, on='date', how='outer').fillna(0)

#############################################
# 3.1) Учёт region -> mx_5d (вспышки M/X за 5 дней)
#############################################
# Приводим region в int
events_df['region'] = pd.to_numeric(events_df['region'], errors='coerce').fillna(0).astype(int)
srs_df['Nmbr'] = pd.to_numeric(srs_df['Nmbr'], errors='coerce').fillna(0).astype(int)

# Определяем is_MX = 1, если type=='XRA' и flare_class в ['M','X']
events_df['is_MX'] = np.where((events_df['type']=='XRA') & (events_df['flare_class'].isin(['M','X'])), 1, 0)
region_mx_daily = events_df.groupby(['date', 'region'])['is_MX'].sum().reset_index()
region_mx_daily = region_mx_daily.sort_values(['region', 'date'])
region_mx_daily['mx_5d'] = region_mx_daily.groupby('region')['is_MX'].rolling(5, min_periods=1).sum().values

# Объединяем с srs_df по дате и региону (Nmbr)
srs_df2 = pd.merge(srs_df, region_mx_daily[['date','region','mx_5d']],
                   left_on=['date','Nmbr'],
                   right_on=['date','region'],
                   how='left')
srs_df2['mx_5d'] = srs_df2['mx_5d'].fillna(0)
mx_5d_agg = srs_df2.groupby('date')['mx_5d'].max().reset_index()
srs_agg = pd.merge(srs_agg, mx_5d_agg, on='date', how='left')
srs_agg['mx_5d'] = srs_agg['mx_5d'].fillna(0)

#############################################
# 3.2) 7-дневные скользящие суммы для SRS
#############################################
srs_agg = srs_agg.sort_values('date').reset_index(drop=True)
srs_agg['sum_complex_area_7d'] = srs_agg['sum_complex_area'].rolling(7, min_periods=1).sum()
srs_agg['mag_code_sum_7d'] = srs_agg['mag_code_sum'].rolling(7, min_periods=1).sum()
srs_agg['complex_count_7d'] = srs_agg['complex_count'].rolling(7, min_periods=1).sum()
srs_agg['mx_5d_7d'] = srs_agg['mx_5d'].rolling(7, min_periods=1).sum()

#############################################
# 4) Финальное объединение SRS и EVENTS
#############################################
merged = pd.merge(srs_agg, events_final, on='date', how='outer').fillna(0)
merged = merged.sort_values('date').reset_index(drop=True)

#############################################
# 5) Вычисляем дополнительные rolling-столбцы в merged
#############################################
# rolling(24h) и rolling(48h) для событий (так как данные агрегированы по дням)
merged['events_24h'] = merged['events_count'].rolling(window=1, min_periods=1).sum()
merged['events_48h'] = merged['events_count'].rolling(window=2, min_periods=1).sum()

merged['strong_24h'] = merged['strong_events'].rolling(window=1, min_periods=1).sum()
merged['strong_48h'] = merged['strong_events'].rolling(window=2, min_periods=1).sum()

merged['A_48h'] = merged['A'].rolling(window=2, min_periods=1).sum()
merged['B_48h'] = merged['B'].rolling(window=2, min_periods=1).sum()
merged['C_48h'] = merged['C'].rolling(window=2, min_periods=1).sum()
merged['M_48h'] = merged['M'].rolling(window=2, min_periods=1).sum()
merged['X_48h'] = merged['X'].rolling(window=2, min_periods=1).sum()

# Вычисляем отношения и разности
merged['ratio_events_to_srs'] = np.where(merged['srs_count'] > 0,
                                           merged['events_count'] / merged['srs_count'],
                                           merged['events_count'])
merged['ratio_strong_to_srs'] = np.where(merged['srs_count'] > 0,
                                           merged['strong_events'] / merged['srs_count'],
                                           merged['strong_events'])
merged['srs_events_interaction'] = merged['srs_count'] * merged['events_count']
merged['srs_area_strong_interaction'] = merged['srs_area_mean'] * merged['strong_events']
merged['srs_NN_ratio'] = merged['srs_NN_mean'] / (merged['srs_LL_mean'] + 1e-5)

merged['ratio_A'] = merged['A'] / (merged['events_count'] + 1e-5)
merged['ratio_B'] = merged['B'] / (merged['events_count'] + 1e-5)
merged['ratio_C'] = merged['C'] / (merged['events_count'] + 1e-5)
merged['ratio_M'] = merged['M'] / (merged['events_count'] + 1e-5)
merged['ratio_X'] = merged['X'] / (merged['events_count'] + 1e-5)

merged['ratio_events_48h_to_24h'] = merged['events_48h'] / (merged['events_24h'] + 1e-5)
merged['ratio_strong_48h_to_24h'] = merged['strong_48h'] / (merged['strong_24h'] + 1e-5)
merged['diff_events_48h_24h'] = merged['events_48h'] - merged['events_24h']

#############################################
# 6) Добавляем таргет target_24
#############################################
merged['target_48'] = (merged['strong_events'].shift(-1).fillna(0) +
                         merged['strong_events'].shift(-2).fillna(0)).apply(lambda x: 1 if x > 0 else 0)


#############################################
# 7) Формирование финального датасета и списка признаков
#############################################
features = [
    # Из srs_agg:
    'srs_count', 'srs_area_mean', 'srs_NN_mean', 'srs_LL_mean', 'mag_code_sum', 'complex_count',
    'sum_complex_area', 'mx_5d', 'sum_complex_area_7d', 'mag_code_sum_7d', 'complex_count_7d', 'mx_5d_7d',
    # Из events_final:
    'events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X',
    # Новые rolling (24h/48h) и отношения:
    'events_24h', 'strong_48h', 'A_48h', 'B_48h', 'C_48h', 'M_48h', 'X_48h',
    'ratio_events_to_srs', 'ratio_strong_to_srs', 'srs_events_interaction',
    'srs_area_strong_interaction', 'srs_NN_ratio', 'ratio_A', 'ratio_B', 'ratio_C', 'ratio_M', 'ratio_X',
    'ratio_events_48h_to_24h', 'ratio_strong_48h_to_24h'
]

data_model = merged.dropna(subset=['target_48']).reset_index(drop=True)

split_idx = int(len(data_model) * 0.8)
train_data = data_model.iloc[:split_idx]
test_data = data_model.iloc[split_idx:]

X_train = train_data[features]
y_train = train_data['target_48']
X_test = test_data[features]
y_test = test_data['target_48']

#############################################
# 8) Обучение моделей
#############################################
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
pred_lgb = (prob_lgb >= 0.5).astype(int)
print_metrics(y_test, pred_lgb, prob_lgb, "LightGBM (48h)")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
prob_rf = rf_model.predict_proba(X_test)[:, 1]
pred_rf = rf_model.predict(X_test)
print_metrics(y_test, pred_rf, prob_rf, "RandomForest (48h)")

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, scale_pos_weight=1, n_jobs=-1)
xgb_model.fit(X_train, y_train)
prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
pred_xgb = (prob_xgb >= 0.5).astype(int)
print_metrics(y_test, pred_xgb, prob_xgb, "XGBoost (48h)")

# Ансамбль
prob_ensemble = (prob_lgb + prob_rf + prob_xgb) / 3
pred_ensemble = (prob_ensemble >= 0.5).astype(int)
print_metrics(y_test, pred_ensemble, prob_ensemble, "Ensemble (LGBM+RF+XGB)")

#############################################
# 9) Сохранение датасета и моделей
#############################################
#data_model.to_csv("merged_events_srs.csv", index=False)
joblib.dump(lgb_model, "../models/s_e_lgb_model_merged_48.pkl")
joblib.dump(rf_model, "../models/s_e_rf_model_merged_48.pkl")
joblib.dump(xgb_model, "../models/s_e_xgb_model_merged_48.pkl")

print("Финальный датасет сохранён в 'merged_events_srs.csv'.")
print("Модель LightGBM сохранена в 's_e_lgb_model_merged.pkl'.")
print("Модель RandomForest сохранена в 's_e_rf_model_merged.pkl'.")
print("Модель XGBoost сохранена в 's_e_xgb_model_merged.pkl'.")

#############################################
# 10) Корреляционная матрица и поиск сильно коррелирующих пар
#############################################
corr_matrix = data_model[features].corr()
print("Корреляционная матрица (численно):")
print(corr_matrix)

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=90)
plt.yticks(range(len(features)), features)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.show()

threshold = 0.9
corr_pairs = []
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
for col in upper_tri.columns:
    for row in upper_tri.index:
        val = upper_tri.loc[row, col]
        if abs(val) > threshold:
            corr_pairs.append((row, col, val))

if corr_pairs:
    print(f"\nПары признаков с корреляцией выше {threshold}:")
    for row, col, val in corr_pairs:
        print(f"{row} и {col}: корреляция = {val:.3f}")
else:
    print(f"\nНет пар признаков с корреляцией выше {threshold}.")

# Стэкинговая модель
meta_features_train = np.column_stack((
    lgb_model.predict_proba(X_train)[:, 1],
    rf_model.predict_proba(X_train)[:, 1],
    xgb_model.predict_proba(X_train)[:, 1]
))
meta_features_test = np.column_stack((prob_lgb, prob_rf, prob_xgb))

meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_features_train, y_train)

prob_stack = meta_model.predict_proba(meta_features_test)[:, 1]
pred_stack = (prob_stack >= 0.5).astype(int)
print_metrics(y_test, pred_stack, prob_stack, "Stacking Ensemble (48h)")

# Сохраняем стэкинг-модель
joblib.dump(meta_model, "../models/s_e_stacking_model_merged_48.pkl")
print("Стэкинговая модель сохранена в 's_e_stacking_model_merged_48.pkl'.")