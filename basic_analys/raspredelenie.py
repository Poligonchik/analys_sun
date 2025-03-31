import os
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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
# Функция определения "сильного" события (т.е. вспышки классов M и X)
#############################################
def is_strong_row(row):
    event_type = str(row.get('type', "")).strip()
    flare_class = str(row.get('flare_class', "None")).strip()
    # Здесь extract_flare_class() уже вернул "M", даже если было "M1.2"
    return 1 if event_type == "XRA" and flare_class in ["M", "X"] else 0

#############################################
# Загрузка и подготовка данных событий (events.json)
#############################################
events_input = "../result_json/events.json"
events_df = pd.read_json(events_input)
events_df['date'] = pd.to_datetime(events_df['date'], format="%Y %m %d")
cols_to_drop = ['begin', 'end']
events_df = events_df.drop(columns=[col for col in cols_to_drop if col in events_df.columns])
events_df = events_df[events_df['loc_freq'].astype(str).str.contains(r'[A-Za-z]', na=False)]
events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)
events_df['strong'] = events_df.apply(is_strong_row, axis=1)

# Агрегация событий по дате
events_agg = events_df.groupby('date').agg(
    events_count=('type', 'count'),
    strong_events=('strong', 'sum')
).reset_index()

# Подсчёт количества вспышек по классам для каждой даты
flare_counts = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0).reset_index()
for cl in ["A", "B", "C", "M", "X"]:
    if cl not in flare_counts.columns:
        flare_counts[cl] = 0

events_final = pd.merge(events_agg, flare_counts, on='date', how='outer')
cols_event = ['events_count','strong_events','A','B','C','M','X']
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
# Объединение агрегированных данных событий и SRS по дате
#############################################
merged = pd.merge(events_final, srs_single, on='date', how='left')
srs_cols = ['Nmbr', 'Lo_srs', 'Area_srs', 'LL_srs', 'NN_srs', 'Mag_Type_srs']
for col in srs_cols:
    merged[col] = merged[col].fillna(0)

#############################################
# Загрузка и обработка данных DSD (dsd.json)
#############################################
dsd_input = "../result_json/dsd.json"
with open(dsd_input, "r", encoding="utf-8") as f:
    dsd_data = json.load(f)
dsd_df = pd.DataFrame(dsd_data)
dsd_df['date'] = pd.to_datetime(dsd_df['date'], format="%Y %m %d")

# Выбираем интересующие признаки из DSD, исключая flares.M и flares.X
new_features = ['radio_flux', 'sunspot_number', 'hemispheric_area', 'new_regions',
                'flares.C', 'flares.S']

#############################################
# Объединение данных DSD с merged по дате
#############################################
merged_final = pd.merge(merged, dsd_df[new_features + ['date']], on='date', how='left')

#############################################
# Вычисление целевой переменной target_24
#############################################
merged_final = merged_final.sort_values('date').reset_index(drop=True)
# Целевая переменная: если в следующий день (сдвиг -1) было зафиксировано хотя бы одно сильное событие, target_24 = 1, иначе 0.
merged_final['target_24'] = merged_final['strong_events'].shift(-1).fillna(0).apply(lambda x: 1 if x > 0 else 0)

#############################################
# Дополнительные комбинационные признаки
#############################################
merged_final['ratio_events_to_srs'] = np.where(merged_final['Nmbr'] > 0,
                                               merged_final['events_count'] / merged_final['Nmbr'],
                                               merged_final['events_count'])
merged_final['diff_events_srs'] = merged_final['events_count'] - merged_final['Nmbr']
merged_final['srs_events_interaction'] = merged_final['Nmbr'] * merged_final['events_count']
merged_final['area_strong_interaction'] = merged_final['Area_srs'] * merged_final['strong_events']
merged_final['NN_LL_ratio'] = merged_final['NN_srs'] / (merged_final['LL_srs'] + 1e-5)

#############################################
# Признаки на основе временных изменений
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
cols_new = ['delta_radio_flux', 'delta_sunspot_number', 'delta_hemispheric_area', 'delta_new_regions',
            'growth_radio_flux', 'growth_sunspot_number', 'growth_hemispheric_area', 'growth_new_regions']
merged_final[cols_new] = merged_final[cols_new].fillna(0)

#############################################
# Формирование финального датасета для моделирования
#############################################
# Из агрегированных данных событий исключаем столбцы "M" и "X"
features = [
    'events_count', 'strong_events', 'A', 'B', 'C',
    'ratio_events_to_srs', 'diff_events_srs', 'srs_events_interaction',
    'Area_srs', 'Lo_srs', 'LL_srs', 'NN_srs', 'NN_LL_ratio',
    'area_strong_interaction'
] + new_features + cols_new
target = 'target_24'

data_model = merged_final.dropna(subset=[target]).reset_index(drop=True)
print("Общий датасет для моделирования:", data_model.shape)
print("Первые строки датасета для моделирования:")
print(data_model.head())

#############################################
# Вывод информации о целевой переменной
#############################################
target_counts = data_model[target].value_counts()
print("\nРаспределение целевой переменной (target_24):")
print(target_counts)
# Здесь, если target_24 == 1, значит, в следующий день была зафиксирована хотя бы одна сильная вспышка (M или X).
print("\nДоля 1 на все:")
print(target_counts[1]/target_counts.sum())
#############################################
# Подсчет общего количества вспышек по классам (на основе всех событий)
#############################################
total_flares = events_df['flare_class'].value_counts()
print("\nОбщее количество вспышек по классам:")
print(total_flares)

#############################################
# Подсчет количества вспышек по классам для каждого дня
#############################################
flares_by_day = events_df.groupby('date')['flare_class'].value_counts().unstack(fill_value=0)
print("\nКоличество вспышек по классам за каждый день:")
print(flares_by_day)
