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

EVENTS_FILE = Path("../files_for_predict/tables/events_download.json")
SRS_FILE = Path("../files_for_predict/tables/srs_download.json")

# Пути к обученным моделям
LGB_MODEL_FILE_24 = Path("../models/s_e_lgb_model_merged_24.pkl")
RF_MODEL_FILE_24 = Path("../models/s_e_rf_model_merged_24.pkl")
XGB_MODEL_FILE_24 = Path("../models/s_e_xgb_model_merged_24.pkl")
STACK_MODEL_FILE_24 = Path("../models/s_e_stacking_model_merged_24.pkl")

LGB_MODEL_FILE_48 = Path("../models/s_e_lgb_model_merged_48.pkl")
RF_MODEL_FILE_48 = Path("../models/s_e_rf_model_merged_48.pkl")
XGB_MODEL_FILE_48 = Path("../models/s_e_xgb_model_merged_48.pkl")
STACK_MODEL_FILE_48 = Path("../models/s_e_stacking_model_merged_48.pkl")

def encode_mag_type(mag_str):
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
    if not isinstance(mag_str, str):
        return 0
    return MAG_TYPE_MAP.get(mag_str.upper().strip(), 0)


def extract_flare_class(particulars):
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"


def combine_date_str(date_str, time_str):
    try:
        date_str = date_str.strip()
        time_str = str(time_str).strip().zfill(4)
        dt_str = date_str + " " + time_str
        return datetime.strptime(dt_str, "%Y %m %d %H%M")
    except Exception as e:
        print(f"Ошибка при combine_date_str: {e}")
        return pd.NaT


def time_str_to_minutes(time_str):
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
    b = row['begin_mins']
    e = row['end_mins']
    if pd.notna(b) and pd.notna(e):
        diff = e - b
        if diff < 0:
            diff += 1440
        return diff
    return np.nan


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


# Функции формирования признаков для объединённых данных
def prepare_features():
    # Загрузка SRS данных
    srs_df = pd.read_json(SRS_FILE)
    srs_df['date'] = pd.to_datetime(srs_df['date'], format="%Y %m %d")
    # Приведение числовых полей
    for col in ['Lo', 'Area', 'LL', 'NN']:
        srs_df[col] = pd.to_numeric(srs_df[col], errors='coerce').fillna(0)
    srs_df['mag_code'] = srs_df['Mag_Type'].apply(encode_mag_type)
    srs_df['is_complex'] = np.where(srs_df['mag_code'] >= 2, 1, 0)
    complex_df = srs_df[srs_df['is_complex'] == 1]
    sum_complex_area = complex_df.groupby('date')['Area'].sum().rename("sum_complex_area")
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
    srs_agg = pd.merge(srs_agg, sum_complex_area, on='date', how='left')
    srs_agg['sum_complex_area'] = srs_agg['sum_complex_area'].fillna(0)

    events_df = pd.read_json(EVENTS_FILE)
    events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")
    events_df['particulars'] = events_df['particulars'].fillna("")
    events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)
    # Флаг сильного события
    events_df['strong'] = np.where((events_df['type'].str.strip() == "XRA") &
                                   (events_df['flare_class'].isin(['M', 'X'])), 1, 0)
    evt_agg = events_df.groupby('date').agg(
        events_count=('type', 'count'),
        strong_events=('strong', 'sum')
    ).reset_index()
    flare_daily = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
    for cl in ["A", "B", "C", "M", "X"]:
        if cl not in flare_daily.columns:
            flare_daily[cl] = 0
    events_final = pd.merge(evt_agg, flare_daily, on='date', how='outer').fillna(0)

    # Обработка region и mx_5d
    events_df['region'] = pd.to_numeric(events_df['region'], errors='coerce').fillna(0).astype(int)
    srs_df['Nmbr'] = pd.to_numeric(srs_df['Nmbr'], errors='coerce').fillna(0).astype(int)
    events_df['is_MX'] = np.where((events_df['type'] == 'XRA') &
                                  (events_df['flare_class'].isin(['M', 'X'])), 1, 0)
    region_mx_daily = events_df.groupby(['date', 'region'])['is_MX'].sum().reset_index()
    region_mx_daily = region_mx_daily.sort_values(['region', 'date'])
    region_mx_daily['mx_5d'] = region_mx_daily.groupby('region')['is_MX'].rolling(5, min_periods=1).sum().values
    srs_df2 = pd.merge(srs_df, region_mx_daily[['date', 'region', 'mx_5d']],
                       left_on=['date', 'Nmbr'],
                       right_on=['date', 'region'],
                       how='left')
    srs_df2['mx_5d'] = srs_df2['mx_5d'].fillna(0)
    mx_5d_agg = srs_df2.groupby('date')['mx_5d'].max().reset_index()
    srs_agg = pd.merge(srs_agg, mx_5d_agg, on='date', how='left')
    srs_agg['mx_5d'] = srs_agg['mx_5d'].fillna(0)

    # 7-дневные скользящие суммы для SRS
    srs_agg = srs_agg.sort_values('date').reset_index(drop=True)
    srs_agg['sum_complex_area_7d'] = srs_agg['sum_complex_area'].rolling(7, min_periods=1).sum()
    srs_agg['mag_code_sum_7d'] = srs_agg['mag_code_sum'].rolling(7, min_periods=1).sum()
    srs_agg['complex_count_7d'] = srs_agg['complex_count'].rolling(7, min_periods=1).sum()
    srs_agg['mx_5d_7d'] = srs_agg['mx_5d'].rolling(7, min_periods=1).sum()

    # Объединяем SRS и события
    merged = pd.merge(srs_agg, events_final, on='date', how='outer').fillna(0)
    merged = merged.sort_values('date').reset_index(drop=True)

    # Дополнительные признаки
    merged['events_24h'] = merged['events_count'].rolling(window=1, min_periods=1).sum()
    merged['events_48h'] = merged['events_count'].rolling(window=2, min_periods=1).sum()
    merged['strong_24h'] = merged['strong_events'].rolling(window=1, min_periods=1).sum()
    merged['strong_48h'] = merged['strong_events'].rolling(window=2, min_periods=1).sum()
    merged['A_48h'] = merged['A'].rolling(window=2, min_periods=1).sum()
    merged['B_48h'] = merged['B'].rolling(window=2, min_periods=1).sum()
    merged['C_48h'] = merged['C'].rolling(window=2, min_periods=1).sum()
    merged['M_48h'] = merged['M'].rolling(window=2, min_periods=1).sum()
    merged['X_48h'] = merged['X'].rolling(window=2, min_periods=1).sum()
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

    # Список признаков
    features = [
        'srs_count', 'srs_area_mean', 'srs_NN_mean', 'srs_LL_mean', 'mag_code_sum', 'complex_count',
        'sum_complex_area', 'mx_5d', 'sum_complex_area_7d', 'mag_code_sum_7d', 'complex_count_7d', 'mx_5d_7d',
        'events_count', 'strong_events', 'A', 'B', 'C', 'M', 'X',
        'events_24h', 'strong_48h', 'A_48h', 'B_48h', 'C_48h', 'M_48h', 'X_48h',
        'ratio_events_to_srs', 'ratio_strong_to_srs', 'srs_events_interaction',
        'srs_area_strong_interaction', 'srs_NN_ratio', 'ratio_A', 'ratio_B', 'ratio_C', 'ratio_M', 'ratio_X',
        'ratio_events_48h_to_24h', 'ratio_strong_48h_to_24h'
    ]

    # Возвращаем последнюю запись
    data_features = merged.dropna().reset_index(drop=True)
    if data_features.empty:
        print("Нет данных для формирования признаков!")
        return None
    return data_features.iloc[-1:][features].copy()


# Прогнозирование на завтра
def predict_tomorrow():
    input_row = prepare_features()
    if input_row is None or input_row.empty:
        print("Нет данных для прогнозирования!")
        return
    # print("Признаки для прогноза (последняя запись):")
    # print(input_row)
    # Загрузка моделей
    lgb_24 = joblib.load(LGB_MODEL_FILE_24)
    rf_24 = joblib.load(RF_MODEL_FILE_24)
    xgb_24 = joblib.load(XGB_MODEL_FILE_24)
    stack_24 = joblib.load(STACK_MODEL_FILE_24)

    lgb_48 = joblib.load(LGB_MODEL_FILE_48)
    rf_48 = joblib.load(RF_MODEL_FILE_48)
    xgb_48 = joblib.load(XGB_MODEL_FILE_48)
    stack_48 = joblib.load(STACK_MODEL_FILE_48)

    # Предсказания
    prob_lgb_24 = lgb_24.predict_proba(input_row)[:, 1][0]
    prob_rf_24 = rf_24.predict_proba(input_row)[:, 1][0]
    prob_xgb_24 = xgb_24.predict_proba(input_row)[:, 1][0]
    prob_ens_24 = (prob_lgb_24 + prob_rf_24 + prob_xgb_24) / 3
    prob_stack_24 = stack_24.predict_proba(np.column_stack((
        prob_lgb_24, prob_rf_24, prob_xgb_24)).reshape(1, -1))[:, 1][0]

    prob_lgb_48 = lgb_48.predict_proba(input_row)[:, 1][0]
    prob_rf_48 = rf_48.predict_proba(input_row)[:, 1][0]
    prob_xgb_48 = xgb_48.predict_proba(input_row)[:, 1][0]
    prob_ens_48 = (prob_lgb_48 + prob_rf_48 + prob_xgb_48) / 3
    prob_stack_48 = stack_48.predict_proba(np.column_stack((
        prob_lgb_48, prob_rf_48, prob_xgb_48)).reshape(1, -1))[:, 1][0]

    print("\nПрогноз на 24 часа (вероятность хотя бы одного сильного события):")
    print(f"LightGBM: {prob_lgb_24:.3f}")
    print(f"RandomForest: {prob_rf_24:.3f}")
    print(f"XGBoost: {prob_xgb_24:.3f}")
    print(f"Ensemble (усреднение): {prob_ens_24:.3f}")
    print(f"Stacking Ensemble: {prob_stack_24:.3f}")

    print("\nПрогноз на 48 часов (вероятность хотя бы одного сильного события):")
    print(f"LightGBM: {prob_lgb_48:.3f}")
    print(f"RandomForest: {prob_rf_48:.3f}")
    print(f"XGBoost: {prob_xgb_48:.3f}")
    print(f"Ensemble (усреднение): {prob_ens_48:.3f}")
    print(f"Stacking Ensemble: {prob_stack_48:.3f}")

    print(
        "\nРекомендуется ориентироваться на прогноз стэккинг-модели.")


if __name__ == "__main__":
    predict_tomorrow()
