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
# Функция определения "сильного" события
#############################################
def is_strong_row(row):
    event_type = str(row.get('type', "")).strip()
    flare_class = str(row.get('flare_class', "None")).strip()
    return 1 if event_type == "XRA" and flare_class in ["M", "X"] else 0


#############################################
# Загрузка и подготовка данных событий (events.json)
#############################################
events_input = "../unified_json/events.json"
events_df = pd.read_json(events_input)
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")
cols_to_drop = ['begin', 'end']
events_df = events_df.drop(columns=[col for col in cols_to_drop if col in events_df.columns])
events_df = events_df[ events_df['loc_freq'].astype(str).str.contains(r'[A-Za-z]', na=False) ]
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)
events_df['strong'] = events_df.apply(is_strong_row, axis=1)
events_agg = events_df.groupby('date').agg(
    events_count=('type', 'count'),
    strong_events=('strong', 'sum')
).reset_index()
flare_counts = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
for cl in ["A", "B", "C", "M", "X"]:
    if cl not in flare_counts.columns:
        flare_counts[cl] = 0
events_final = pd.merge(events_agg, flare_counts, on='date', how='outer')
cols_event = ['events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X']
events_final[cols_event] = events_final[cols_event].fillna(0)

#############################################
# Загрузка данных SRS (srs.json)
#############################################
srs_input = "../unified_json/srs.json"
srs_df = pd.read_json(srs_input)
for col in ['Lo', 'Area', 'LL', 'NN']:
    srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce')
srs_df['date'] = pd.to_datetime(srs_df['date'], format="%Y %m %d")
# Оставляем Mag_Type как строку (без заполнения нулями)
srs_df = srs_df.rename(columns={
    'Lo': 'Lo_srs',
    'Area': 'Area_srs',
    'LL': 'LL_srs',
    'NN': 'NN_srs',
    'Mag_Type': 'Mag_Type_srs'
})
srs_single = srs_df.sort_values('date').drop_duplicates(subset=['date'], keep='first')

#############################################
# Объединение агрегированных событий и SRS по дате
#############################################
merged = pd.merge(events_final, srs_single, on='date', how='left')
# Для числовых столбцов заполним пропуски нулями, а для Mag_Type_srs оставим как есть
for col in ['Nmbr', 'Lo_srs', 'Area_srs', 'LL_srs', 'NN_srs']:
    merged[col] = merged[col].fillna(0)

#############################################
# Дополнительные комбинационные признаки
#############################################
merged['ratio_events_to_srs'] = np.where(merged['Nmbr'] > 0,
                                           merged['events_count'] / merged['Nmbr'],
                                           merged['events_count'])
merged['diff_events_srs'] = merged['events_count'] - merged['Nmbr']
merged['srs_events_interaction'] = merged['Nmbr'] * merged['events_count']
merged['area_strong_interaction'] = merged['Area_srs'] * merged['strong_events']
merged['NN_LL_ratio'] = merged['NN_srs'] / (merged['LL_srs'] + 1e-5)

#############################################
# Новые признаки по солнечным пятнам и магнитным полям
#############################################
merged['Nmbr'] = pd.to_numeric(merged['Nmbr'], errors='coerce').fillna(0)
merged['target_24_Nmbr'] = merged['Nmbr'].shift(-1).fillna(0)
merged['target_48_Nmbr'] = merged['Nmbr'].shift(-1).fillna(0) + merged['Nmbr'].shift(-2).fillna(0)
merged['change_in_Nmbr'] = merged['Nmbr'] - merged['Nmbr'].shift(1).fillna(0)
merged['sunspot_density'] = merged['Nmbr'] / (merged['Area_srs'] + 1e-5)
merged['change_in_density'] = merged['sunspot_density'] - merged['sunspot_density'].shift(1).fillna(0)

# Вычисление производной по площади активных областей
merged['change_in_Area_srs'] = merged['Area_srs'] - merged['Area_srs'].shift(1).fillna(0)

# Создание дамми‑переменных для сложности магнитного поля
mag_dummies = pd.get_dummies(merged['Mag_Type_srs'], prefix='mag')
merged = pd.concat([merged, mag_dummies], axis=1)

#############################################
# Новые признаки для прогноза на 48 часов (события)
#############################################
merged = merged.sort_values('date').reset_index(drop=True)
merged['target_48'] = (merged['strong_events'].shift(-1).fillna(0) + merged['strong_events'].shift(-2).fillna(0)).apply(lambda x: 1 if x > 0 else 0)
merged['target_24_A'] = merged['A'].shift(-1).fillna(0)
merged['target_24_B'] = merged['B'].shift(-1).fillna(0)
merged['target_24_C'] = merged['C'].shift(-1).fillna(0)
merged['target_24_M'] = merged['M'].shift(-1).fillna(0)
merged['target_24_X'] = merged['X'].shift(-1).fillna(0)
merged['target_48_A'] = (merged['A'].shift(-1).fillna(0) + merged['A'].shift(-2).fillna(0))
merged['target_48_B'] = (merged['B'].shift(-1).fillna(0) + merged['B'].shift(-2).fillna(0))
merged['target_48_C'] = (merged['C'].shift(-1).fillna(0) + merged['C'].shift(-2).fillna(0))
merged['target_48_M'] = (merged['M'].shift(-1).fillna(0) + merged['M'].shift(-2).fillna(0))
merged['target_48_X'] = (merged['X'].shift(-1).fillna(0) + merged['X'].shift(-2).fillna(0))
merged['rolling_events_2d'] = merged['events_count'].rolling(window=2, min_periods=1).sum()
merged['rolling_strong_2d'] = merged['strong_events'].rolling(window=2, min_periods=1).sum()
merged['change_in_events'] = merged['events_count'] - merged['events_count'].shift(1).fillna(0)
merged['change_in_strong'] = merged['strong_events'] - merged['strong_events'].shift(1).fillna(0)
merged['weekday'] = merged['date'].dt.weekday

#############################################
# Формирование финального датасета для моделирования
#############################################
# Основной целевой признак – прогноз сильных вспышек за 48 часов (target_48)
features = [
    'events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X',
    'ratio_events_to_srs', 'diff_events_srs', 'srs_events_interaction',
    'Area_srs', 'Lo_srs', 'LL_srs', 'NN_srs', 'NN_LL_ratio',
    'area_strong_interaction',
    'rolling_events_2d', 'rolling_strong_2d', 'change_in_events', 'change_in_strong', 'weekday',
    'Nmbr', 'change_in_Nmbr', 'sunspot_density', 'change_in_density',
    'change_in_Area_srs'
]
# Добавляем дамми-признаки для магнитного типа
mag_dummy_cols = list(mag_dummies.columns)
features.extend(mag_dummy_cols)

target = 'target_48'

data_model = merged.dropna(subset=[target]).reset_index(drop=True)
print("Общий датасет для моделирования:", data_model.shape)
print("Первые строки:")
print(data_model.head())

#############################################
# Аугментация данных (искусственное раздувание выборки)
#############################################
def augment_data(df, features, n_augments=5):
    augmented_rows = []
    for idx, row in df.iterrows():
        for i in range(n_augments):
            new_row = row.copy()
            for col in features:
                if pd.api.types.is_numeric_dtype(new_row[col]):
                    # Для счетных признаков, включая Nmbr, события и дамми-признаки, добавляем дискретный шум
                    if col in ['events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X', 'Nmbr']:
                        noise = np.random.choice([-1, 0, 1])
                        new_val = int(new_row[col]) + noise
                        new_row[col] = new_val if new_val >= 0 else 0
                    # Для площади активных областей изменяем с меньшим коэффициентом
                    elif col == 'Area_srs':
                        factor = 0.02
                        noise = np.random.normal(0, factor * new_row[col]) if new_row[col] != 0 else np.random.normal(0, 0.05)
                        new_row[col] = new_row[col] + noise
                    else:
                        factor = 0.05
                        noise = np.random.normal(0, factor * new_row[col]) if new_row[col] != 0 else np.random.normal(0, 0.1)
                        new_row[col] = new_row[col] + noise
            augmented_rows.append(new_row)
    return pd.DataFrame(augmented_rows)

augmented_data = augment_data(data_model, features, n_augments=5)
data_model_augmented = pd.concat([data_model, augmented_data], ignore_index=True)
print("После аугментации общий датасет для моделирования:", data_model_augmented.shape)

#############################################
# Разделение данных (хронологически, 80% обучение, 20% тест)
#############################################
data_model_augmented = data_model_augmented.sort_values('date').reset_index(drop=True)
split_idx = int(len(data_model_augmented) * 0.8)
train_data = data_model_augmented.iloc[:split_idx]
test_data = data_model_augmented.iloc[split_idx:]

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print("Размер обучения:", X_train.shape)
print("Размер теста:", X_test.shape)
if X_train.empty or X_train.ndim != 2:
    raise ValueError("Обучающая выборка пуста или имеет неверную размерность!")

#############################################
# Обучение моделей
#############################################
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_pred_prob_lgb >= 0.5).astype(int)
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, "LightGBM (48h)")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, "RandomForest (48h)")

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_prob_xgb >= 0.5).astype(int)
print_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, "XGBoost (48h)")

y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf + y_pred_prob_xgb) / 3
y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, "Ensemble (48h)")

#############################################
# Сохранение датасета и моделей
#############################################
data_model_augmented.to_csv("merged_events_srs_48h.csv", index=False)
joblib.dump(lgb_model, "lgb_model_merged_48h.pkl")
joblib.dump(rf_model, "rf_model_merged_48h.pkl")
joblib.dump(xgb_model, "xgb_model_merged_48h.pkl")

print("Финальный датасет сохранен в 'merged_events_srs_48h.csv'")
print("Модель LightGBM сохранена в 'lgb_model_merged_48h.pkl'")
print("Модель RandomForest сохранена в 'rf_model_merged_48h.pkl'")
print("Модель XGBoost сохранена в 'xgb_model_merged_48h.pkl'")
