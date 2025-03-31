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

###########################################
# Пути к файлам с текущей активностью (events_download.json)
###########################################
EVENTS_FILE = Path("../files_for_predict/tables/events_download.json")

###########################################
# Пути к обученным моделям (target_24)
###########################################
LGB_MODEL_FILE = Path("../models/e_lightgbm_model_target_24.pkl")
RF_MODEL_FILE = Path("../models/e_random_forest_model_target_24.pkl")
XGB_MODEL_FILE = Path("../models/e_xgboost_model_target_24.pkl")
LGB_MODEL_FILE_48 = Path("../models/e_lightgbm_model_target_48.pkl")
RF_MODEL_FILE_48 = Path("../models/e_random_forest_model_target_48.pkl")
XGB_MODEL_FILE_48 = Path("../models/e_xgboost_model_target_48.pkl")

###########################################
# Функции для подготовки признаков (как при обучении)
###########################################
def combine_datetime(row):
    try:
        # Если 'date' уже Timestamp, преобразуем в строку
        if isinstance(row['date'], pd.Timestamp):
            date_str = row['date'].strftime("%Y %m %d")
        else:
            date_str = str(row['date']).strip()
        # Приводим begin к строке и дополняем ведущими нулями
        begin_str = str(row['begin']).strip().zfill(4)
        dt_str = date_str + " " + begin_str
        return datetime.strptime(dt_str, "%Y %m %d %H%M")
    except Exception as e:
        print(f"Ошибка при combine_datetime для строки: {row[['date', 'begin']]}, {e}")
        return pd.NaT


def time_str_to_minutes(time_str):
    """
    Преобразует значение времени (формат HHMM) в минуты от полуночи.
    Приводит входное значение к строке.
    """
    if pd.isna(time_str):
        return np.nan
    time_str = str(time_str)
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


def compute_duration(row):
    """
    Вычисляет длительность события с учетом перехода через полночь.
    """
    b = row['begin_mins']
    e = row['end_mins']
    if pd.notna(b) and pd.notna(e):
        diff = e - b
        if diff < 0:
            diff += 1440
        return diff
    return np.nan


def extract_flare_class(particulars):
    """
    Извлекает класс вспышки из particulars.
    """
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"


def add_flare_type_features(df, window_hours):
    """
    Вычисляет признаки за последние window_hours часов:
      - last_flare_{window_hours}h: класс последней вспышки,
      - count_A_{window_hours}h, ..., count_X_{window_hours}h,
      - total_flare_count_{window_hours}h: общее число вспышек,
      - ratio_MX_{window_hours}h: (count_M + count_X) / total_flare_count.
    """
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
    df[f'ratio_MX_{window_hours}h'] = (np.array(count_M) + np.array(count_X)) / (np.array(total_count) + 1e-5)
    return df


def compute_daily_flare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет суточные признаки: yesterday_count, daybefore_count, growth_flare,
    на основе группировки по календарной дате (из timestamp).
    """
    df['event_date'] = df['timestamp'].dt.date
    daily = df[df['flare_class'] != "None"].groupby('event_date').size().rename("daily_flare_count").reset_index()
    daily['yesterday_count'] = daily['daily_flare_count'].shift(1)
    daily['daybefore_count'] = daily['daily_flare_count'].shift(2)
    daily['growth_flare'] = (daily['yesterday_count'] - daily['daybefore_count']) / (daily['daybefore_count'] + 1e-5)
    daily['date'] = pd.to_datetime(daily['event_date'])
    return daily[['date', 'yesterday_count', 'daybefore_count', 'growth_flare']]


def prepare_prediction_data() -> pd.DataFrame:
    """
    Загружает данные из EVENTS_FILE, вычисляет timestamp и все признаки для прогнозирования target_24.
    Используется events_download.json.
    Возвращает последнюю запись с 15 признаками:
      ['hour', 'weekday', 'month', 'duration', 'last_flare_24h',
       'count_A_24h', 'count_B_24h', 'count_C_24h', 'count_M_24h', 'count_X_24h',
       'total_flare_count_24h', 'ratio_MX_24h', 'yesterday_count', 'daybefore_count', 'growth_flare']
    """
    df = pd.read_json(EVENTS_FILE)
    print("После загрузки, df.shape =", df.shape)

    df = df.dropna(subset=['date', 'begin', 'end', 'particulars'])
    print("После dropna, df.shape =", df.shape)

    df['date'] = df['date'].apply(lambda d: d.strip() if isinstance(d, str) else d)

    df['timestamp'] = df.apply(combine_datetime, axis=1)
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    print("После вычисления timestamp, df.shape =", df.shape)

    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    df['month'] = df['timestamp'].dt.month

    df['begin_mins'] = df['begin'].apply(time_str_to_minutes)
    df['end_mins'] = df['end'].apply(time_str_to_minutes)
    df['duration'] = df.apply(compute_duration, axis=1)

    df['flare_class'] = df['particulars'].apply(extract_flare_class)

    df = add_flare_type_features(df, 24)

    daily_feats = compute_daily_flare_features(df)
    print("Суточные признаки, daily_feats.shape =", daily_feats.shape)

    df = pd.merge(df, daily_feats, on='date', how='left')
    print("После слияния с суточными признаками, df.shape =", df.shape)

    # Преобразуем категориальный признак last_flare_24h в числовой код
    df['last_flare_24h'] = df['last_flare_24h'].astype('category').cat.codes

    features_to_use = ['hour', 'weekday', 'month', 'duration',
                       'last_flare_24h', 'count_A_24h', 'count_B_24h',
                       'count_C_24h', 'count_M_24h', 'count_X_24h',
                       'total_flare_count_24h', 'ratio_MX_24h',
                       'yesterday_count', 'daybefore_count', 'growth_flare']

    df_final = df[features_to_use].dropna().reset_index(drop=True)
    print("Финальный набор признаков для прогнозирования, df_final.shape =", df_final.shape)

    return df_final.iloc[-1:].copy()


###########################################
# Прогноз выводом
###########################################
def predict_tomorrow():
    input_row = prepare_prediction_data()
    if input_row.empty:
        print("Нет данных для прогнозирования!")
        return
    print("Признаки для прогноза (последняя запись):")
    print(input_row)

    # Загрузка обученных моделей
    lgb_model = joblib.load(LGB_MODEL_FILE)
    rf_model = joblib.load(RF_MODEL_FILE)
    xgb_model = joblib.load(XGB_MODEL_FILE)

    lgb_model_48 = joblib.load(LGB_MODEL_FILE_48)
    rf_model_48 = joblib.load(RF_MODEL_FILE_48)
    xgb_model_48 = joblib.load(XGB_MODEL_FILE_48)

    # Получаем прогнозные вероятности
    prob_lgb = lgb_model.predict_proba(input_row)[:, 1][0]
    prob_rf = rf_model.predict_proba(input_row)[:, 1][0]
    prob_xgb = xgb_model.predict_proba(input_row)[:, 1][0]
    prob_lgb_48 = lgb_model_48.predict_proba(input_row)[:, 1][0]
    prob_rf_48 = rf_model_48.predict_proba(input_row)[:, 1][0]
    prob_xgb_48 = xgb_model_48.predict_proba(input_row)[:, 1][0]

    prob_ensemble_48 = (prob_lgb_48 + prob_rf_48 + prob_xgb_48) / 3.0
    prob_ensemble = (prob_lgb + prob_rf + prob_xgb) / 3.0
    prob_weighted_48 = 0.5 * prob_lgb_48 + 0.25 * prob_rf_48 + 0.25 * prob_xgb_48
    prob_weighted = 0.5 * prob_lgb + 0.25 * prob_rf + 0.25 * prob_xgb
    print("\nПрогноз для завтрашнего дня (вероятность наличия хотя бы одного сильного события):")
    print(f"LightGBM: {prob_lgb:.3f}")
    print(f"RandomForest: {prob_rf:.3f}")
    print(f"XGBoost: {prob_xgb:.3f}")
    print(f"Ensemble (усреднение): {prob_ensemble:.3f}")
    print(f"Ensemble (взвешенное): {prob_weighted:.3f}")

    print("\nПрогноз для следующих 48 часов (вероятность наличия хотя бы одного сильного события):")
    print(f"LightGBM: {prob_lgb_48:.3f}")
    print(f"RandomForest: {prob_rf_48:.3f}")
    print(f"XGBoost: {prob_xgb_48:.3f}")
    print(f"Ensemble (усреднение): {prob_ensemble_48:.3f}")
    print(f"Ensemble (взвешенное): {prob_weighted_48:.3f}")

    # На основании предыдущих метрик рекомендуем ориентироваться на модель LightGBM
    print(
        "\nРекомендуется ориентироваться на прогноз модели LightGBM, так как она демонстрировала лучшие показатели (ROC AUC, F1 Score) на обучении.")


if __name__ == "__main__":
    predict_tomorrow()
