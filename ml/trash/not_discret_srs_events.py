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
    # Точное совпадение
    exact_acc = accuracy_score(y_true, y_pred)

    # Допустимая погрешность: считаем предсказание верным, если абсолютная разница между кодами не больше 1.
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
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"


#############################################
# Функция определения "сильного" события (используется ранее, но для многоклассовой цели не требуется)
#############################################
def is_strong_row(row):
    event_type = str(row.get('type', "")).strip()
    flare_class = str(row.get('flare_class', "None")).strip()
    return 1 if event_type == "XRA" and flare_class in ["M", "X"] else 0


#############################################
# Функция вычисления взвешенной интенсивности вспышек для одной даты
#############################################
def weighted_intensity(row):
    total = row['A'] * 1 + row['B'] * 2 + row['C'] * 3 + row['M'] * 4 + row['X'] * 5
    return total / row['events_count'] if row['events_count'] > 0 else 0


#############################################
# Функция для определения максимального (наивысшего) класса вспышек по порядку
#############################################
def get_max_class(row):
    # Порядок по возрастанию интенсивности: A < B < C < M < X
    for cl in ['X', 'M', 'C', 'B', 'A']:  # идем с конца – первая найденная (если > 0) будет максимальной
        if row.get(cl, 0) > 0:
            return cl
    return "None"


#############################################
# Загрузка и подготовка данных событий (events.json)
#############################################
events_input = "../result_json/events.json"
events_df = pd.read_json(events_input)

# Преобразуем поле date (формат "YYYY MM DD")
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")

# Убираем поля с временем (begin, end)
cols_to_drop = ['begin', 'end']
events_df = events_df.drop(columns=[col for col in cols_to_drop if col in events_df.columns])

# Отбираем только события, где в поле loc_freq присутствуют латинские буквы
events_df = events_df[events_df['loc_freq'].astype(str).str.contains(r'[A-Za-z]', na=False)]

# Вычисляем класс вспышки
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)
# Вычисляем метку "сильное событие" (используется для агрегации, если нужно)
events_df['strong'] = events_df.apply(is_strong_row, axis=1)

# Группировка событий по дате:
# Считаем общее число событий, число сильных событий и число вспышек по классам
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
cols_event = ['events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X']
events_final[cols_event] = events_final[cols_event].fillna(0)

#############################################
# Загрузка данных SRS (srs.json)
#############################################
srs_input = "../result_json/srs.json"
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
# Объединение агрегированных событий и SRS по дате
#############################################
merged = pd.merge(events_final, srs_single, on='date', how='left')
srs_cols = ['Nmbr', 'Lo_srs', 'Area_srs', 'LL_srs', 'NN_srs', 'Mag_Type_srs']
for col in srs_cols:
    merged[col] = merged[col].fillna(0)

#############################################
# Вычисление дополнительных признаков
#############################################
# Взвешенная интенсивность (среднее значение по классам, где вес = порядковый номер)
merged['weighted_intensity'] = merged.apply(weighted_intensity, axis=1)
# Отношение сильных событий к общему числу событий
merged['ratio_strong'] = merged['strong_events'] / (merged['events_count'] + 1e-5)

#############################################
# Формирование целевого признака для следующего дня
#############################################
# Для каждой даты определим максимальный класс вспышек, произошедший в этот день
merged['max_flare_class'] = merged.apply(get_max_class, axis=1)
# Целевой признак для текущей строки – это максимальный класс следующего дня
merged['target_class'] = merged['max_flare_class'].shift(-1)
# Оставляем только те строки, где target_class является одним из A, B, C, M, X
merged = merged[merged['target_class'].isin(["A", "B", "C", "M", "X"])].reset_index(drop=True)

# Преобразуем целевой признак в категорию с порядком
cat_target = pd.Categorical(merged['target_class'], categories=["A", "B", "C", "M", "X"], ordered=True)
merged['target_numeric'] = cat_target.codes  # 0 для A, 1 для B, ..., 4 для X

#############################################
# Дополнительные комбинационные признаки (по желанию)
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
# Выбираем признаки из событий и SRS, а также новые признаки
features = [
    'events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X',
    'ratio_events_to_srs', 'diff_events_srs', 'srs_events_interaction',
    'Area_srs', 'Lo_srs', 'LL_srs', 'NN_srs', 'NN_LL_ratio',
    'area_strong_interaction', 'weighted_intensity', 'ratio_strong'
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
                    # Для счетных признаков (целые) – если это счетчики событий
                    if col in ['events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X']:
                        noise = np.random.choice([-1, 0, 1])
                        new_val = int(new_row[col]) + noise
                        new_row[col] = new_val if new_val >= 0 else 0
                    else:
                        # Для непрерывных признаков добавляем гауссовский шум (примерно 5% от значения)
                        factor = 0.05
                        noise = np.random.normal(0, factor * new_row[col]) if new_row[col] != 0 else np.random.normal(0,
                                                                                                                      0.1)
                        new_row[col] = new_row[col] + noise
            augmented_rows.append(new_row)
    return pd.DataFrame(augmented_rows)


# Создаем дополнительные синтетические строки (например, 5 копий для каждой исходной строки)
augmented_data = augment_data(data_model, features, n_augments=5)
# Объединяем оригинальные данные и синтетически созданные
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
# Обучение моделей для многоклассовой классификации
#############################################
num_classes = 5  # A, B, C, M, X → 0,1,2,3,4

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

# Энсамблирование: усреднение вероятностей
y_pred_prob_ensemble = (y_pred_prob_lgb + rf_model.predict_proba(X_test) + y_pred_prob_xgb) / 3
y_pred_ensemble = np.argmax(y_pred_prob_ensemble, axis=1)
print_multiclass_metrics(y_test, y_pred_ensemble, model_name="Ensemble (24h)")


#############################################
# Вычисление дополнительных метрик: совпадение по доп. правилу (±1 класс)
#############################################
def tolerance_accuracy(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) <= 1)


tol_acc_lgb = tolerance_accuracy(y_test, y_pred_lgb)
tol_acc_rf = tolerance_accuracy(y_test, y_pred_rf)
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
