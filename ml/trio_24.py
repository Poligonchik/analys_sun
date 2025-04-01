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
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#############################################
# Функция для вывода метрик
#############################################
def print_metrics(y_true, y_pred, y_prob, model_name="Model"):
    print(f"\nМетрики предсказания для {model_name}:")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.3f} ({acc * 100:.1f}%)")
    prec = precision_score(y_true, y_pred, zero_division=0)
    print(f"Precision: {prec:.3f}")
    rec = recall_score(y_true, y_pred, zero_division=0)
    print(f"Recall: {rec:.3f}")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"F1 Score: {f1:.3f}")
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"ROC AUC: {roc_auc:.3f}")
    except Exception:
        pass

    # Расчет TSS: TSS = Recall + Specificity - 1
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        tss = rec + specificity - 1
        print(f"TSS: {tss:.3f}")
    else:
        print("TSS: Не вычисляется для данной размерности матрицы ошибок.")

    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

#############################################
# Функции для извлечения класса вспышки и определения сильного события
#############################################
def extract_flare_class(particulars):
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"

def is_strong_row(row):
    event_type = str(row.get('type', "")).strip()
    flare_class = str(row.get('flare_class', "None")).strip()
    return 1 if (event_type == "XRA" and flare_class in ["M", "X"]) else 0

#############################################
# 1) Загрузка и подготовка данных
#############################################
events_input = "../result_json/events.json"
events_df = pd.read_json(events_input)
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")
# Удаляем "begin","end", если есть
cols_to_drop = ['begin', 'end']
events_df = events_df.drop(columns=[col for col in cols_to_drop if col in events_df.columns])

# Фильтруем строки, где 'loc_freq' содержит хотя бы одну букву
events_df = events_df[events_df['loc_freq'].astype(str).str.contains(r'[A-Za-z]', na=False)]

# Определяем класс вспышки и признак "strong"
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)
events_df['strong'] = events_df.apply(is_strong_row, axis=1)

# Агрегируем по датам
events_agg = events_df.groupby('date').agg(
    events_count=('type', 'count'),
    strong_events=('strong', 'sum')
).reset_index()

# Подсчитываем количество вспышек каждого класса
flare_counts = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
for cl in ["A", "B", "C", "M", "X"]:
    if cl not in flare_counts.columns:
        flare_counts[cl] = 0

events_final = pd.merge(events_agg, flare_counts, on='date', how='outer')
cols_event = ['events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X']
events_final[cols_event] = events_final[cols_event].fillna(0)

#############################################
# 2) Загрузка данных SRS (srs.json)
#############################################
srs_input = "../result_json/srs.json"
srs_df = pd.read_json(srs_input)
for col in ['Lo', 'Area', 'LL', 'NN']:
    srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce')
srs_df['date'] = pd.to_datetime(srs_df['date'], format="%Y %m %d")

# Переименуем для уникальности
srs_df = srs_df.rename(columns={
    'Lo': 'Lo_srs',
    'Area': 'Area_srs',
    'LL': 'LL_srs',
    'NN': 'NN_srs',
    'Mag_Type': 'Mag_Type_srs'
})
srs_single = srs_df.sort_values('date').drop_duplicates(subset=['date'], keep='first')

#############################################
# 3) Объединяем Events + SRS
#############################################
merged = pd.merge(events_final, srs_single, on='date', how='left')
srs_cols = ['Nmbr', 'Lo_srs', 'Area_srs', 'LL_srs', 'NN_srs', 'Mag_Type_srs']
for col in srs_cols:
    merged[col] = merged[col].fillna(0)

#############################################
# 4) Загрузка и обработка DSD (dsd.json)
#############################################
dsd_input = "../result_json/dsd.json"
with open(dsd_input, "r", encoding="utf-8") as f:
    dsd_data = json.load(f)
dsd_df = pd.DataFrame(dsd_data)
dsd_df['date'] = pd.to_datetime(dsd_df['date'], format="%Y %m %d")

# Интересующие признаки
new_features = [
    'radio_flux', 'sunspot_number', 'hemispheric_area', 'new_regions',
    'flares.C', 'flares.M', 'flares.X', 'flares.S'
]
# Объединяем с merged
merged_final = pd.merge(merged, dsd_df[new_features + ['date']], on='date', how='left')

#############################################
# 5) Целевой признак target_24 = наличие сильного события (M/X) завтра
#############################################
merged_final = merged_final.sort_values('date').reset_index(drop=True)
merged_final['target_24'] = merged_final['strong_events'].shift(-1).fillna(0).apply(lambda x: 1 if x > 0 else 0)

#############################################
# 6) Кодирование магнитного типа
#############################################
MAG_TYPE_MAP = {
    'ALPHA': 1, 'BETA': 1,
    'BETA-GAMMA': 2, 'GAMMA': 2,
    'BETA-GAMMA-DELTA': 3, 'BETA-DELTA': 3,
    'GAMMA-DELTA': 3, 'DELTA': 3
}
def encode_mag_type(x):
    if x == 0 or pd.isna(x):
        return 0
    return MAG_TYPE_MAP.get(str(x).strip().upper(), 0)

merged_final['Mag_Type_code'] = merged_final['Mag_Type_srs'].apply(encode_mag_type)

#############################################
# 7) Доп. комбинационные признаки
#############################################
merged_final['ratio_events_to_srs'] = np.where(merged_final['Nmbr'] > 0,
                                               merged_final['events_count'] / merged_final['Nmbr'],
                                               merged_final['events_count'])
merged_final['diff_events_srs'] = merged_final['events_count'] - merged_final['Nmbr']
merged_final['srs_events_interaction'] = merged_final['Nmbr'] * merged_final['events_count']
merged_final['area_strong_interaction'] = merged_final['Area_srs'] * merged_final['strong_events']
merged_final['NN_LL_ratio'] = merged_final['NN_srs'] / (merged_final['LL_srs'] + 1e-5)

#############################################
# 8) Дельты, рост
#############################################
merged_final = merged_final.sort_values('date').reset_index(drop=True)
merged_final['delta_radio_flux'] = merged_final['radio_flux'] - merged_final['radio_flux'].shift(1)
merged_final['delta_sunspot_number'] = merged_final['sunspot_number'] - merged_final['sunspot_number'].shift(1)
merged_final['delta_hemispheric_area'] = merged_final['hemispheric_area'] - merged_final['hemispheric_area'].shift(1)
merged_final['delta_new_regions'] = merged_final['new_regions'] - merged_final['new_regions'].shift(1)

merged_final['growth_radio_flux'] = (merged_final['delta_radio_flux'] / merged_final['radio_flux'].shift(1)).replace([np.inf, -np.inf], np.nan)
merged_final['growth_sunspot_number'] = (merged_final['delta_sunspot_number'] / merged_final['sunspot_number'].shift(1)).replace([np.inf, -np.inf], np.nan)
merged_final['growth_hemispheric_area'] = (merged_final['delta_hemispheric_area'] / merged_final['hemispheric_area'].shift(1)).replace([np.inf, -np.inf], np.nan)
merged_final['growth_new_regions'] = (merged_final['delta_new_regions'] / merged_final['new_regions'].shift(1)).replace([np.inf, -np.inf], np.nan)

cols_new = [
    'delta_radio_flux','delta_sunspot_number','delta_hemispheric_area','delta_new_regions',
    'growth_radio_flux','growth_sunspot_number','growth_hemispheric_area','growth_new_regions'
]
merged_final[cols_new] = merged_final[cols_new].fillna(0)

#############################################
# 9) Лаги и скользящие средние
#############################################
for col in ['radio_flux','sunspot_number','hemispheric_area']:
    for lag in [1,2,3]:
        merged_final[f'{col}_lag{lag}'] = merged_final[col].shift(lag).fillna(0)

for col in ['radio_flux','events_count','sunspot_number','hemispheric_area']:
    merged_final[f'{col}_2d_mean'] = merged_final[col].rolling(2, min_periods=1).mean()
    merged_final[f'{col}_3d_mean'] = merged_final[col].rolling(3, min_periods=1).mean()
    merged_final[f'{col}_5d_mean'] = merged_final[col].rolling(5, min_periods=1).mean()

merged_final['radio_flux_7d_std'] = merged_final['radio_flux'].rolling(7, min_periods=1).std().fillna(0)

#############################################
# 10) Формируем список признаков
#############################################
features = [
    # Из events
    'events_count','strong_events','A','B','C','M','X',
    'ratio_events_to_srs','diff_events_srs','srs_events_interaction',
    # Из SRS
    'Area_srs','Lo_srs','LL_srs','NN_srs','NN_LL_ratio',
    'area_strong_interaction','Mag_Type_code',
    # Из DSD
    'radio_flux','sunspot_number','hemispheric_area','new_regions',
    # Дельты и рост
    'delta_radio_flux','delta_sunspot_number','delta_hemispheric_area','delta_new_regions',
    'growth_radio_flux','growth_sunspot_number','growth_hemispheric_area','growth_new_regions',
    # Rolling aggregates
    'radio_flux_3d_mean','events_count_3d_mean',
    'hemispheric_area_2d_mean'
]
target = 'target_24'

data_model = merged_final.dropna(subset=[target]).reset_index(drop=True)
print("Общий датасет для моделирования:", data_model.shape)
print("Первые строки:")
print(data_model.head(5))

#############################################
# 11) Аугментация данных
#############################################
def augment_data(df, features, n_augments=2):
    augmented_rows = []
    for idx, row in df.iterrows():
        for i in range(n_augments):
            new_row = row.copy()
            for c in features:
                if pd.api.types.is_numeric_dtype(new_row[c]):
                    if c in ['events_count','strong_events','A','B','C','M','X','new_regions']:
                        noise = np.random.choice([-1,0,1])
                        new_val = int(new_row[c]) + noise
                        new_row[c] = new_val if new_val>=0 else 0
                    else:
                        factor = 0.05
                        noise = np.random.normal(0, factor*new_row[c]) if new_row[c]!=0 else np.random.normal(0, 0.1)
                        new_row[c] = new_row[c] + noise
            augmented_rows.append(new_row)
    return pd.DataFrame(augmented_rows)

augmented_data = augment_data(data_model, features, n_augments=2)
data_model_augmented = pd.concat([data_model, augmented_data], ignore_index=True)
print("После аугментации общий датасет для моделирования:", data_model_augmented.shape)

#############################################
# 12) Разделение на train/test по времени (80/20)
#############################################
data_model_augmented = data_model_augmented.sort_values('date').reset_index(drop=True)
split_idx = int(len(data_model_augmented)*0.8)
train_data = data_model_augmented.iloc[:split_idx]
test_data  = data_model_augmented.iloc[split_idx:]

X_train = train_data[features]
y_train = train_data[target]
X_test  = test_data[features]
y_test  = test_data[target]

print("Размер обучения:", X_train.shape)
print("Размер теста:", X_test.shape)

if X_train.empty or X_train.ndim!=2:
    raise ValueError("Обучающая выборка пуста или неверная размерность!")

#############################################
# 13) Балансировка SMOTE
#############################################
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("После SMOTE:", X_train_balanced.shape)

#############################################
# 14) Обучение трёх базовых моделей
#############################################
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train_balanced, y_train_balanced)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:,1]
y_pred_lgb = (y_pred_prob_lgb>=0.5).astype(int)
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, "LightGBM(24h)")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42,
                                  class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:,1]
y_pred_rf = (y_pred_prob_rf>=0.5).astype(int)
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, "RandomForest(24h)")

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=42, n_jobs=-1)
xgb_model.fit(X_train_balanced, y_train_balanced)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:,1]
y_pred_xgb = (y_pred_prob_xgb>=0.5).astype(int)
print_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, "XGBoost(24h)")

#############################################
# 15) Простейшее усреднение (Ensemble)
#############################################
y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf + y_pred_prob_xgb)/3
y_pred_ensemble = (y_pred_prob_ensemble>=0.5).astype(int)
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, "MeanEnsemble(24h)")

# Взвешенное
weights = [0.25,0.25,0.5]
y_pred_prob_weighted = (weights[0]*y_pred_prob_lgb + weights[1]*y_pred_prob_rf + weights[2]*y_pred_prob_xgb)
y_pred_weighted = (y_pred_prob_weighted>=0.5).astype(int)
print_metrics(y_test, y_pred_weighted, y_pred_prob_weighted, "WeightedEnsemble(24h)")

#############################################
# 16) Stacking
#############################################
meta_features_train = np.column_stack((
    lgb_model.predict_proba(X_train_balanced)[:,1],
    rf_model.predict_proba(X_train_balanced)[:,1],
    xgb_model.predict_proba(X_train_balanced)[:,1]
))
meta_features_test = np.column_stack((
    y_pred_prob_lgb,   # предсказания на X_test
    y_pred_prob_rf,
    y_pred_prob_xgb
))

meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_features_train, y_train_balanced)

y_pred_prob_stack = meta_model.predict_proba(meta_features_test)[:,1]
y_pred_stack = (y_pred_prob_stack>=0.5).astype(int)
print_metrics(y_test, y_pred_stack, y_pred_prob_stack, "StackingEnsemble(24h)")

#############################################
# 17) Сохраняем базовые модели и метамодель
#############################################
data_model_augmented.to_csv("merged_events_srs.csv", index=False)
joblib.dump(lgb_model, "../models/d_s_e_lgb_model_merged_24.pkl")
joblib.dump(rf_model,  "../models/d_s_e_rf_model_merged_24.pkl")
joblib.dump(xgb_model, "../models/d_s_e_xgb_model_merged_24.pkl")

# ВАЖНО: сохраняем и метамодель (стэкинг)
joblib.dump(meta_model,"../models/d_s_e_stacking_model_merged_24.pkl")

print("\nФинальный датасет сохранён в 'merged_events_srs.csv'")
print("LightGBM сохранён как 'd_s_e_lgb_model_merged_24.pkl'")
print("RandomForest сохранён как 'd_s_e_rf_model_merged_24.pkl'")
print("XGBoost сохранён как 'd_s_e_xgb_model_merged_24.pkl'")
print("Стэкинг (логист.регрессия) сохранён как 'd_s_e_stacking_model_merged_24.pkl'")

#############################################
# 18) Корреляционная матрица
#############################################
corr_matrix = data_model[features].corr()
print("Корреляционная матрица:")
print(corr_matrix)

plt.figure(figsize=(10,8))
plt.imshow(corr_matrix, interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=90)
plt.yticks(range(len(features)), features)
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.show()
