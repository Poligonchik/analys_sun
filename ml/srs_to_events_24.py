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

# Преобразуем поле date (формат "YYYY MM DD")
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")

# Убираем поля с временем, так как нам они не нужны
# (если в исходном файле присутствуют begin и end, мы их просто отбрасываем)
cols_to_drop = ['begin', 'end']
events_df = events_df.drop(columns=[col for col in cols_to_drop if col in events_df.columns])

# Отбираем только те события, где в поле loc_freq присутствуют латинские буквы
events_df = events_df[ events_df['loc_freq'].astype(str).str.contains(r'[A-Za-z]', na=False) ]

# Вычисляем класс вспышки и метку "сильное событие"
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)
events_df['strong'] = events_df.apply(is_strong_row, axis=1)

# Группировка событий по дате:
# Для каждой даты считаем:
#   - общее число событий,
#   - число сильных событий,
#   - число вспышек по классам (A, B, C, M, X)
events_agg = events_df.groupby('date').agg(
    events_count=('type', 'count'),
    strong_events=('strong', 'sum')
).reset_index()

# Получаем число вспышек по классам
flare_counts = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
for cl in ["A", "B", "C", "M", "X"]:
    if cl not in flare_counts.columns:
        flare_counts[cl] = 0

# Объединяем агрегированные данные событий
events_final = pd.merge(events_agg, flare_counts, on='date', how='outer')
# Заполняем пропущенные значения нулями
cols_event = ['events_count','strong_events','A','B','C','M','X']
events_final[cols_event] = events_final[cols_event].fillna(0)

#############################################
# Загрузка данных SRS (srs.json)
#############################################
srs_input = "../unified_json/srs.json"
srs_df = pd.read_json(srs_input)
# Приводим числовые поля к числовому типу
for col in ['Lo', 'Area', 'LL', 'NN']:
    srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce')
srs_df['date'] = pd.to_datetime(srs_df['date'], format="%Y %m %d")
# Переименовываем столбцы SRS, чтобы не было конфликтов с данными событий
srs_df = srs_df.rename(columns={
    'Lo': 'Lo_srs',
    'Area': 'Area_srs',
    'LL': 'LL_srs',
    'NN': 'NN_srs',
    'Mag_Type': 'Mag_Type_srs'
})
# Если для одной даты несколько записей – оставляем первую
srs_single = srs_df.sort_values('date').drop_duplicates(subset=['date'], keep='first')

#############################################
# Объединение агрегированных событий и SRS по дате
#############################################
merged = pd.merge(events_final, srs_single, on='date', how='left')
# Если для даты нет данных SRS – заполняем соответствующие столбцы нулями
srs_cols = ['Nmbr', 'Lo_srs', 'Area_srs', 'LL_srs', 'NN_srs', 'Mag_Type_srs']
for col in srs_cols:
    merged[col] = merged[col].fillna(0)

#############################################
# Вычисление целевого признака
#############################################
# Таргет: наличие хотя бы одного сильного события (strong_events > 0) в следующий календарный день
merged = merged.sort_values('date').reset_index(drop=True)
merged['target_24'] = merged['strong_events'].shift(-1).fillna(0).apply(lambda x: 1 if x > 0 else 0)

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
# Формирование финального датасета для моделирования
#############################################
# Выбираем признаки из событий и SRS
features = [
    'events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X',
    'ratio_events_to_srs', 'diff_events_srs', 'srs_events_interaction',
    'Area_srs', 'Lo_srs', 'LL_srs', 'NN_srs', 'NN_LL_ratio',
    'area_strong_interaction'
]
target = 'target_24'

data_model = merged.dropna(subset=[target]).reset_index(drop=True)
print("Общий датасет для моделирования:", data_model.shape)
print("Первые строки:")
print(data_model.head())

#############################################
# Разделение данных (хронологически, 80% обучение, 20% тест)
#############################################
split_idx = int(len(data_model) * 0.8)
train_data = data_model.iloc[:split_idx]
test_data = data_model.iloc[split_idx:]

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
# 1. LightGBM
lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
y_pred_lgb = (y_pred_prob_lgb >= 0.5).astype(int)
print_metrics(y_test, y_pred_lgb, y_pred_prob_lgb, "LightGBM (24h)")

# 2. RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = rf_model.predict(X_test)
print_metrics(y_test, y_pred_rf, y_pred_prob_rf, "RandomForest (24h)")

# 3. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_prob_xgb >= 0.5).astype(int)
print_metrics(y_test, y_pred_xgb, y_pred_prob_xgb, "XGBoost (24h)")

# Энсамблирование: усреднение вероятностей
y_pred_prob_ensemble = (y_pred_prob_lgb + y_pred_prob_rf + y_pred_prob_xgb) / 3
y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)
print_metrics(y_test, y_pred_ensemble, y_pred_prob_ensemble, "Ensemble (24h)")

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
