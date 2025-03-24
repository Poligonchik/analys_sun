import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
import xgboost as xgb
import joblib

#############################################
# Функция для вывода метрик для многоклассовой задачи
#############################################
def print_multiclass_metrics(y_true, y_pred, model_name="Model"):
    exact_acc = accuracy_score(y_true, y_pred)
    tolerance_acc = np.mean(np.abs(y_true - y_pred) <= 1)
    print(f"\nМетрики предсказания для {model_name}:")
    print(f"Exact Accuracy: {exact_acc:.3f} ({exact_acc * 100:.1f}%)")
    print(f"Tolerance Accuracy (|pred - true| <= 1): {tolerance_acc:.3f} ({tolerance_acc * 100:.1f}%)")
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
    # Рассматриваем только релевантные классы: C, M, X.
    for cl in ["X", "M", "C"]:
        if part.startswith(cl):
            return cl
    return "None"


#############################################
# Функция для определения максимального (наивысшего) класса из релевантных
#############################################
def get_max_class_filtered(row):
    # Проверяем в порядке убывания: X > M > C.
    if row.get("X", 0) > 0:
        return "X"
    elif row.get("M", 0) > 0:
        return "M"
    elif row.get("C", 0) > 0:
        return "C"
    else:
        return "None"


#############################################
# Агрегация дополнительных признаков по событиям
#############################################
def add_extra_event_features(df):
    df['ratio_C'] = df.apply(lambda row: row['C'] / row['events_count'] if row['events_count'] > 0 else 0, axis=1)
    df['ratio_M'] = df.apply(lambda row: row['M'] / row['events_count'] if row['events_count'] > 0 else 0, axis=1)
    df['ratio_X'] = df.apply(lambda row: row['X'] / row['events_count'] if row['events_count'] > 0 else 0, axis=1)
    df['flare_score'] = df.apply(lambda row: (row['C']*1 + row['M']*2 + row['X']*3) / row['events_count'] if row['events_count'] > 0 else 0, axis=1)
    return df


#############################################
# Загрузка и подготовка данных событий (events.json)
#############################################
events_input = "../unified_json/events.json"
events_df = pd.read_json(events_input)

# Преобразуем поле date (формат "YYYY MM DD")
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")

# Убираем поля с временем (begin, end)
cols_to_drop = ['begin', 'end']
events_df = events_df.drop(columns=[col for col in cols_to_drop if col in events_df.columns])

# Отбираем только события, где в поле loc_freq присутствуют латинские буквы
events_df = events_df[events_df['loc_freq'].astype(str).str.contains(r'[A-Za-z]', na=False)]

# Вычисляем класс вспышки; оставляем только релевантные: C, M, X, иначе "None"
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)

# Группируем по дате: считаем только релевантные вспышки (где flare_class != "None")
events_agg = events_df.groupby('date').agg(
    events_count=('flare_class', lambda x: (x != "None").sum())
).reset_index()

# Получаем число вспышек по классам
flare_counts = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
for cl in ["C", "M", "X"]:
    if cl not in flare_counts.columns:
        flare_counts[cl] = 0

# Объединяем агрегированные данные событий
events_final = pd.merge(events_agg, flare_counts, on='date', how='outer')
cols_event = ['events_count', 'C', 'M', 'X']
events_final[cols_event] = events_final[cols_event].fillna(0)

# Добавляем дополнительные признаки, вычисленные по событиям
events_final = add_extra_event_features(events_final)

#############################################
# Загрузка данных SRS (srs.json)
#############################################
srs_input = "../unified_json/srs.json"
srs_df = pd.read_json(srs_input)
for col in ['Lo', 'Area', 'LL', 'NN']:
    srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce')
srs_df['date'] = pd.to_datetime(srs_df['date'], format="%Y %m %d")
srs_df = srs_df.rename(columns={
    'Lo': 'Lo_srs',
    'Area': 'Area_srs',
    'LL': 'LL_srs',
    'NN': 'NN_srs',
    'Mag_Type': 'Mag_Type_srs'
})
srs_single = srs_df.sort_values('date').drop_duplicates(subset=['date'], keep='first')

#############################################
# Объединение агрегированных данных событий и SRS по дате
#############################################
merged = pd.merge(events_final, srs_single, on='date', how='left')
srs_cols = ['Nmbr', 'Lo_srs', 'Area_srs', 'LL_srs', 'NN_srs', 'Mag_Type_srs']
for col in srs_cols:
    merged[col] = merged[col].fillna(0)

#############################################
# Вычисление дополнительных признаков из объединённых данных
#############################################
# Взвешенная интенсивность: (C*1 + M*2 + X*3) / events_count
merged['weighted_intensity'] = merged.apply(lambda row: (row['C']*1 + row['M']*2 + row['X']*3) / row['events_count']
                                              if row['events_count'] > 0 else 0, axis=1)
# Доля сильных вспышек (считаем сильными M и X)
merged['ratio_strong'] = merged.apply(lambda row: (row['M'] + row['X']) / row['events_count']
                                        if row['events_count'] > 0 else 0, axis=1)

#############################################
# Формирование целевого признака
#############################################
# Определяем максимальный класс вспышек в день (из релевантных: C, M, X)
merged['max_flare_class'] = merged.apply(get_max_class_filtered, axis=1)
# Целевой признак для текущей строки – это максимальный класс следующего дня
merged['target_class'] = merged['max_flare_class'].shift(-1)
merged = merged.reset_index(drop=True)
# Оставляем только строки, где target_class является "None", "C", "M" или "X"
merged = merged[merged['target_class'].isin(["None", "C", "M", "X"])].reset_index(drop=True)
# Преобразуем целевой признак в упорядоченную категорию:
# 0 - "None" (нет вспышки), 1 - "C", 2 - "M", 3 - "X"
cat_target = pd.Categorical(merged['target_class'], categories=["None", "C", "M", "X"], ordered=True)
merged['target_numeric'] = cat_target.codes

#############################################
# Дополнительные комбинационные признаки
#############################################
merged['ratio_events_to_srs'] = np.where(merged['Nmbr'] > 0,
                                         merged['events_count'] / merged['Nmbr'],
                                         merged['events_count'])
merged['diff_events_srs'] = merged['events_count'] - merged['Nmbr']
merged['srs_events_interaction'] = merged['Nmbr'] * merged['events_count']
merged['area_strong_interaction'] = merged['Area_srs'] * merged['C']  # например, взаимодействие SRS и числа вспышек C
merged['NN_LL_ratio'] = merged['NN_srs'] / (merged['LL_srs'] + 1e-5)

#############################################
# Формирование финального датасета для моделирования
#############################################
features = [
    'events_count', 'C', 'M', 'X', 'ratio_events_to_srs', 'diff_events_srs',
    'srs_events_interaction', 'Area_srs', 'Lo_srs', 'LL_srs', 'NN_srs', 'NN_LL_ratio',
    'area_strong_interaction', 'weighted_intensity', 'ratio_strong',
    'ratio_C', 'ratio_M', 'ratio_X', 'flare_score'
]
target = 'target_numeric'

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
            # Для каждого числового признака из списка features добавляем небольшой шум
            for col in features:
                if pd.api.types.is_numeric_dtype(new_row[col]):
                    if col in ['events_count', 'C', 'M', 'X']:
                        noise = np.random.choice([-1, 0, 1])
                        new_val = int(new_row[col]) + noise
                        new_row[col] = new_val if new_val >= 0 else 0
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
# Обучение моделей для многоклассовой классификации (4 класса: 0 - None, 1 - C, 2 - M, 3 - X)
#############################################
num_classes = 4

# 1. LightGBM (objective 'multiclass')
lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=num_classes, random_state=42, n_jobs=-1)
lgb_model.fit(X_train, y_train)
y_pred_prob_lgb = lgb_model.predict_proba(X_test)
y_pred_lgb = np.argmax(y_pred_prob_lgb, axis=1)
print_multiclass_metrics(y_test, y_pred_lgb, model_name="LightGBM (24h)")

# 2. RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print_multiclass_metrics(y_test, y_pred_rf, model_name="RandomForest (24h)")

# 3. XGBoost (objective 'multi:softprob')
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=num_classes, use_label_encoder=False,
                              eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)
y_pred_xgb = np.argmax(y_pred_prob_xgb, axis=1)
print_multiclass_metrics(y_test, y_pred_xgb, model_name="XGBoost (24h)")

# Энсамблирование: усреднение вероятностей от всех моделей
y_pred_prob_ensemble = (y_pred_prob_lgb + rf_model.predict_proba(X_test) + y_pred_prob_xgb) / 3
y_pred_ensemble = np.argmax(y_pred_prob_ensemble, axis=1)
print_multiclass_metrics(y_test, y_pred_ensemble, model_name="Ensemble (24h)")

#############################################
# Дополнительная метрика: Tolerance Accuracy (|pred - true| <= 1)
#############################################
def tolerance_accuracy(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) <= 1)

tol_acc_lgb = tolerance_accuracy(y_test, y_pred_lgb)
tol_acc_rf  = tolerance_accuracy(y_test, y_pred_rf)
tol_acc_xgb = tolerance_accuracy(y_test, y_pred_xgb)
tol_acc_ens = tolerance_accuracy(y_test, y_pred_ensemble)

print("\nTolerance Accuracy (|pred - true| <= 1):")
print(f"LightGBM: {tol_acc_lgb:.3f}")
print(f"RandomForest: {tol_acc_rf:.3f}")
print(f"XGBoost: {tol_acc_xgb:.3f}")
print(f"Ensemble: {tol_acc_ens:.3f}")

#############################################
# Сохранение датасета и моделей
#############################################
data_model_augmented.to_csv("merged_events_srs.csv", index=False)
joblib.dump(lgb_model, "lgb_model_merged.pkl")
joblib.dump(rf_model, "rf_model_merged.pkl")
joblib.dump(xgb_model, "xgb_model_merged.pkl")

print("Финальный датасет сохранен в 'merged_events_srs.csv'")
print("Модель LightGBM сохранена в 'lgb_model_merged.pkl'")
print("Модель RandomForest сохранена в 'rf_model_merged.pkl'")
print("Модель XGBoost сохранена в 'xgb_model_merged.pkl'")
