#!/usr/bin/env python3
import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
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
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        tss = rec + specificity - 1
        print(f"TSS: {tss:.3f}")
    else:
        print("TSS: Не вычисляется для данной матрицы ошибок.")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


#############################################
# Функции для обработки данных
#############################################
def combine_datetime(row):
    try:
        if isinstance(row['date'], pd.Timestamp):
            date_str = row['date'].strftime("%Y %m %d")
        else:
            date_str = str(row['date']).strip()
        begin_str = str(row['begin']).strip().zfill(4)
        dt_str = date_str + " " + begin_str
        return datetime.strptime(dt_str, "%Y %m %d %H%M")
    except Exception as e:
        print(f"Ошибка в combine_datetime для {row[['date', 'begin']]}, {e}")
        return pd.NaT


def time_str_to_minutes(time_str):
    if pd.isna(time_str):
        return np.nan
    clean = re.sub(r'\D', '', str(time_str))
    if clean == "":
        return np.nan
    clean = clean.zfill(4)
    try:
        hours = int(clean[:2])
        minutes = int(clean[2:4])
        return hours * 60 + minutes
    except Exception:
        return np.nan


def compute_duration(row):
    b = row.get('begin_mins')
    e = row.get('end_mins')
    if pd.notna(b) and pd.notna(e):
        diff = e - b
        if diff < 0:
            diff += 1440
        return diff
    return np.nan


def extract_flare_class(particulars):
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"


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


def compute_daily_flare_features(df):
    df['event_date'] = df['timestamp'].dt.date
    daily = df[df['flare_class'] != "None"].groupby('event_date').size().rename("daily_flare_count").reset_index()
    daily['yesterday_count'] = daily['daily_flare_count'].shift(1)
    daily['daybefore_count'] = daily['daily_flare_count'].shift(2)
    daily['growth_flare'] = (daily['yesterday_count'] - daily['daybefore_count']) / (daily['daybefore_count'] + 1e-5)
    daily['date'] = pd.to_datetime(daily['event_date'])
    return daily[['date', 'yesterday_count', 'daybefore_count', 'growth_flare']]


#############################################
# Функция формирования признаков для объединённых данных
#############################################
def prepare_prediction_features():
    # Загрузка событий
    EVENTS_FILE = Path("../files_for_predict/tables/events_download.json")
    df_events = pd.read_json(EVENTS_FILE)
    df_events = df_events.dropna(subset=['date', 'begin', 'end', 'particulars'])
    df_events['date'] = pd.to_datetime(df_events['date'], format="%Y %m %d", errors='coerce')
    df_events['timestamp'] = df_events.apply(combine_datetime, axis=1)
    df_events = df_events.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df_events['hour'] = df_events['timestamp'].dt.hour
    df_events['weekday'] = df_events['timestamp'].dt.weekday
    df_events['month'] = df_events['timestamp'].dt.month
    df_events['begin_mins'] = df_events['begin'].apply(time_str_to_minutes)
    df_events['end_mins'] = df_events['end'].apply(time_str_to_minutes)
    df_events['duration'] = df_events.apply(compute_duration, axis=1)
    df_events['flare_class'] = df_events['particulars'].apply(extract_flare_class)
    df_events = add_flare_type_features(df_events, 24)
    daily_feats = compute_daily_flare_features(df_events)
    df_events = pd.merge(df_events, daily_feats, on='date', how='left')
    df_events['last_flare_24h'] = df_events['last_flare_24h'].astype('category').cat.codes

    # Агрегация событий по датам
    events_agg = df_events.groupby('date').agg(
        events_count=('type', 'count'),
        strong_events=('flare_class', lambda x: sum([1 for fl in x if fl in ["M", "X"]])
                       )).reset_index()
    flare_counts = df_events.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
    for cl in ["A", "B", "C", "M", "X"]:
        if cl not in flare_counts.columns:
            flare_counts[cl] = 0
    events_final = pd.merge(events_agg, flare_counts, on='date', how='outer').fillna(0)

    # Загрузка данных SRS
    SRS_FILE = Path("../files_for_predict/tables/srs_download.json")
    df_srs = pd.read_json(SRS_FILE)
    df_srs['date'] = pd.to_datetime(df_srs['date'], format="%Y %m %d", errors='coerce')
    for col in ['Lo', 'Area', 'LL', 'NN']:
        df_srs[col] = pd.to_numeric(df_srs[col], errors='coerce').fillna(0)
    print("SRS столбцы до переименования:", df_srs.columns.tolist())

    def encode_mag_type(x):
        MAG_TYPE_MAP = {
            'ALPHA': 1, 'BETA': 1, 'BETA-GAMMA': 2, 'GAMMA': 2,
            'BETA-GAMMA-DELTA': 3, 'BETA-DELTA': 3, 'GAMMA-DELTA': 3, 'DELTA': 3
        }
        if not isinstance(x, str):
            return 0
        return MAG_TYPE_MAP.get(x.upper().strip(), 0)

    df_srs['Mag_Type_code'] = df_srs['Mag_Type'].apply(encode_mag_type)
    df_srs = df_srs.rename(columns={
        'Lo': 'Lo_srs',
        'Area': 'Area_srs',
        'LL': 'LL_srs',
        'NN': 'NN_srs',
        'Mag_Type': 'Mag_Type_srs'
    })
    print("SRS столбцы после переименования:", df_srs.columns.tolist())
    srs_single = df_srs.sort_values('date').drop_duplicates(subset=['date'], keep='first')

    # Загрузка данных DSD
    DSD_FILE = Path("../files_for_predict/tables/dsd_download.json")
    with open(DSD_FILE, "r", encoding="utf-8") as f:
        dsd_data = json.load(f)
    df_dsd = pd.DataFrame(dsd_data)
    df_dsd['date'] = pd.to_datetime(df_dsd['date'], format="%Y %m %d", errors='coerce')

    # Определяем список признаков из DSD
    new_features = [
        'radio_flux', 'sunspot_number', 'hemispheric_area', 'new_regions',
        'flares.C', 'flares.M', 'flares.X', 'flares.S'
    ]

    # Объединяем: сначала Events + SRS, затем + DSD
    merged = pd.merge(events_final, srs_single, on='date', how='outer').fillna(0)
    print("Столбцы после объединения событий и SRS:", merged.columns.tolist())
    merged_final = pd.merge(merged, df_dsd[new_features + ['date']], on='date', how='left').fillna(0)
    merged_final = merged_final.sort_values('date').reset_index(drop=True)
    print("Столбцы итогового объединенного датасета:", merged_final.columns.tolist())

    # Целевой признак: наличие сильного события (M/X) на следующий день
    merged_final['target_24'] = merged_final['strong_events'].shift(-1).fillna(0).apply(lambda x: 1 if x > 0 else 0)

    # Дополнительные комбинационные признаки
    merged_final['ratio_events_to_srs'] = np.where(merged_final['Nmbr'] > 0,
                                                   merged_final['events_count'] / merged_final['Nmbr'],
                                                   merged_final['events_count'])
    merged_final['diff_events_srs'] = merged_final['events_count'] - merged_final['Nmbr']
    merged_final['srs_events_interaction'] = merged_final['Nmbr'] * merged_final['events_count']
    merged_final['area_strong_interaction'] = merged_final['Area_srs'] * merged_final['strong_events']
    merged_final['NN_LL_ratio'] = merged_final['NN_srs'] / (merged_final['LL_srs'] + 1e-5)

    # Признаки на основе дельт и темпов роста (DSD)
    merged_final['delta_radio_flux'] = merged_final['radio_flux'] - merged_final['radio_flux'].shift(1)
    merged_final['delta_sunspot_number'] = merged_final['sunspot_number'] - merged_final['sunspot_number'].shift(1)
    merged_final['delta_hemispheric_area'] = merged_final['hemispheric_area'] - merged_final['hemispheric_area'].shift(
        1)
    merged_final['delta_new_regions'] = merged_final['new_regions'] - merged_final['new_regions'].shift(1)
    merged_final['growth_radio_flux'] = (
                merged_final['delta_radio_flux'] / merged_final['radio_flux'].shift(1)).replace([np.inf, -np.inf],
                                                                                                np.nan)
    merged_final['growth_sunspot_number'] = (
                merged_final['delta_sunspot_number'] / merged_final['sunspot_number'].shift(1)).replace(
        [np.inf, -np.inf], np.nan)
    merged_final['growth_hemispheric_area'] = (
                merged_final['delta_hemispheric_area'] / merged_final['hemispheric_area'].shift(1)).replace(
        [np.inf, -np.inf], np.nan)
    merged_final['growth_new_regions'] = (
                merged_final['delta_new_regions'] / merged_final['new_regions'].shift(1)).replace([np.inf, -np.inf],
                                                                                                  np.nan)
    for col in ['delta_radio_flux', 'delta_sunspot_number', 'delta_hemispheric_area', 'delta_new_regions',
                'growth_radio_flux', 'growth_sunspot_number', 'growth_hemispheric_area', 'growth_new_regions']:
        merged_final[col] = merged_final[col].fillna(0)

    # Лаги для показателей DSD
    for col in ['radio_flux', 'sunspot_number', 'hemispheric_area']:
        for lag in [1, 2, 3]:
            merged_final[f'{col}_lag{lag}'] = merged_final[col].shift(lag).fillna(0)

    # Скользящие средние (3-дневное среднее)
    for col in ['radio_flux', 'events_count', 'sunspot_number', 'hemispheric_area']:
        merged_final[f'{col}_3d_mean'] = merged_final[col].rolling(3, min_periods=1).mean()

    merged_final['radio_flux_7d_std'] = merged_final['radio_flux'].rolling(7, min_periods=1).std().fillna(0)

    # Формирование финального списка признаков (32 признака, как в обучении)
    final_features = [
        'events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X',
        'ratio_events_to_srs', 'diff_events_srs', 'srs_events_interaction',
        'Area_srs', 'Lo_srs', 'LL_srs', 'NN_srs', 'NN_LL_ratio',
        'area_strong_interaction', 'Mag_Type_code',
        'radio_flux', 'sunspot_number', 'hemispheric_area', 'new_regions',
        'delta_radio_flux', 'delta_sunspot_number', 'delta_hemispheric_area', 'delta_new_regions',
        'growth_radio_flux', 'growth_sunspot_number', 'growth_hemispheric_area', 'growth_new_regions',
        'radio_flux_3d_mean', 'events_count_3d_mean', 'hemispheric_area_2d_mean'
    ]
    # Если признака hemispheric_area_2d_mean нет, вычисляем как 2-дневное среднее по hemispheric_area
    if 'hemispheric_area_2d_mean' not in merged_final.columns:
        merged_final['hemispheric_area_2d_mean'] = merged_final['hemispheric_area'].rolling(2, min_periods=1).mean()

    merged_final = merged_final.fillna(0).sort_values('date').reset_index(drop=True)
    print("Последние 5 строк итогового датасета:")
    print(merged_final.tail(5))
    return merged_final, final_features


#############################################
# Функция предсказания для горизонтов 24h и 48h с использованием стэкинга
#############################################
def predict_tomorrow():
    merged_final, features = prepare_prediction_features()
    # Выбираем последнюю запись по списку признаков
    prediction_row = merged_final.iloc[-1:][features].fillna(0).reset_index(drop=True)
    if prediction_row.empty:
        print("Нет данных для прогнозирования!")
        return
    print("Признаки для прогноза (последняя запись):")
    print(prediction_row)

    # Пути к моделям для target_24
    LGB_MODEL_FILE_24 = Path("../models/d_s_e_lgb_model_merged_24.pkl")
    RF_MODEL_FILE_24 = Path("../models/d_s_e_rf_model_merged_24.pkl")
    XGB_MODEL_FILE_24 = Path("../models/d_s_e_xgb_model_merged_24.pkl")
    STACK_MODEL_FILE_24 = Path("../models/d_s_e_stacking_model_merged_24.pkl")
    # Пути к моделям для target_48
    LGB_MODEL_FILE_48 = Path("../models/d_s_e_lgb_model_merged_48.pkl")
    RF_MODEL_FILE_48 = Path("../models/d_s_e_rf_model_merged_48.pkl")
    XGB_MODEL_FILE_48 = Path("../models/d_s_e_xgb_model_merged_48.pkl")

    # Загрузка моделей для 24 часов
    lgb_model_24 = joblib.load(LGB_MODEL_FILE_24)
    rf_model_24 = joblib.load(RF_MODEL_FILE_24)
    xgb_model_24 = joblib.load(XGB_MODEL_FILE_24)
    stack_model_24 = joblib.load(STACK_MODEL_FILE_24)
    # Загрузка моделей для 48 часов
    lgb_model_48 = joblib.load(LGB_MODEL_FILE_48)
    rf_model_48 = joblib.load(RF_MODEL_FILE_48)
    xgb_model_48 = joblib.load(XGB_MODEL_FILE_48)

    # Предсказания для горизонта 24 часов
    prob_lgb_24 = lgb_model_24.predict_proba(prediction_row)[:, 1][0]
    prob_rf_24 = rf_model_24.predict_proba(prediction_row)[:, 1][0]
    prob_xgb_24 = xgb_model_24.predict_proba(prediction_row)[:, 1][0]
    prob_ens_24 = (prob_lgb_24 + prob_rf_24 + prob_xgb_24) / 3.0
    meta_features_24 = np.array([[prob_lgb_24, prob_rf_24, prob_xgb_24]])
    prob_stack_24 = stack_model_24.predict_proba(meta_features_24)[:, 1][0]

    # Предсказания для горизонта 48 часов
    prob_lgb_48 = lgb_model_48.predict_proba(prediction_row)[:, 1][0]
    prob_rf_48 = rf_model_48.predict_proba(prediction_row)[:, 1][0]
    prob_xgb_48 = xgb_model_48.predict_proba(prediction_row)[:, 1][0]
    prob_ens_48 = (prob_lgb_48 + prob_rf_48 + prob_xgb_48) / 3.0
    meta_features_48 = np.array([[prob_lgb_48, prob_rf_48, prob_xgb_48]])

    print("\nПрогноз на 24 часа (вероятность наличия хотя бы одного сильного события):")
    print(f"LightGBM: {prob_lgb_24:.3f}")
    print(f"RandomForest: {prob_rf_24:.3f}")
    print(f"XGBoost: {prob_xgb_24:.3f}")
    print(f"Ensemble (усреднение): {prob_ens_24:.3f}")
    print(f"Stacking Ensemble: {prob_stack_24:.3f}")

    print("\nПрогноз на 48 часов (вероятность наличия хотя бы одного сильного события):")
    print(f"LightGBM: {prob_lgb_48:.3f}")
    print(f"RandomForest: {prob_rf_48:.3f}")
    print(f"XGBoost: {prob_xgb_48:.3f}")
    print(f"Ensemble (усреднение): {prob_ens_48:.3f}")

    print(
        "\nРекомендуется ориентироваться на прогноз стэкинг-модели, так как она продемонстрировала лучшие результаты на тестовой выборке.")


if __name__ == "__main__":
    # Для предсказания обучаем модели и сохраняем их (если требуется, этот блок можно не выполнять)
    # Здесь предполагается, что модели уже обучены и сохранены по указанным путям.
    predict_tomorrow()
