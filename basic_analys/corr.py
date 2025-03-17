import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ---------- 1) ЗАГРУЗКА DSD.JSON ----------
with open('../unified_json/dsd.json', 'r', encoding='utf-8') as f:
    dsd_data = json.load(f)

df_dsd = pd.DataFrame(dsd_data)
df_dsd['date'] = pd.to_datetime(df_dsd['date'], format='%Y %m %d')

# ---------- 2) ЗАГРУЗКА EVENTS.JSON ----------
with open('../unified_json/events.json', 'r', encoding='utf-8') as f:
    events_data = json.load(f)

df_events = pd.DataFrame(events_data)
df_events['date'] = pd.to_datetime(df_events['date'], format='%Y %m %d')

# «Разворачиваем» (explode) список events
df_events_expanded = df_events.explode('events').reset_index(drop=True)

# Раскладываем вложенные словари (events) в отдельные столбцы
df_events_expanded = pd.concat(
    [
        df_events_expanded.drop(columns=['events']),
        pd.json_normalize(df_events_expanded['events'])
    ],
    axis=1
)

# Считаем, сколько всего событий за каждую дату (event_count)
df_events_grouped = df_events_expanded.groupby('date').size().reset_index(name='event_count')

# ---------- 3) ЗАГРУЗКА SGAS.JSON ----------
with open('../unified_json/sgas.json', 'r', encoding='utf-8') as f:
    sgas_data = json.load(f)

df_sgas = pd.DataFrame(sgas_data)
df_sgas['date'] = pd.to_datetime(df_sgas['date'], format='%Y %m %d')

# «Разворачиваем» (explode) список energetic_events
df_sgas_expanded = df_sgas.explode('energetic_events').reset_index(drop=True)

# Раскладываем вложенные словари (energetic_events) в отдельные столбцы
df_sgas_expanded = pd.concat(
    [
        df_sgas_expanded.drop(columns=['energetic_events']),
        pd.json_normalize(df_sgas_expanded['energetic_events'])
    ],
    axis=1
)

# Считаем, сколько энергетических событий за каждую дату
df_sgas_grouped = df_sgas_expanded.groupby('date').size().reset_index(name='energetic_count')

# ---------- 4) ЗАГРУЗКА SRS.JSON ----------
with open('../unified_json/srs.json', 'r', encoding='utf-8') as f:
    srs_data = json.load(f)

df_srs = pd.DataFrame(srs_data['regions_with_sunspots'])
df_srs['date'] = pd.to_datetime(df_srs['date'], format='%Y %m %d')

# Суммарная площадь пятен (Area) и кол-во регионов (region_count)
df_srs_grouped = (
    df_srs
    .groupby('date')
    .agg(
        total_area=('Area', lambda x: pd.to_numeric(x, errors='coerce').sum()),
        region_count=('Nmbr', 'count')
    )
    .reset_index()
)

# ---------- 5) ОБЪЕДИНЕНИЕ ВСЕХ ДАННЫХ ПО ДАТЕ ----------
df_merged = pd.merge(df_dsd, df_events_grouped, on='date', how='outer')
df_merged = pd.merge(df_merged, df_sgas_grouped, on='date', how='outer')
df_merged = pd.merge(df_merged, df_srs_grouped, on='date', how='outer')

# ---------- 6) ВЫБОР ЧИСЛОВЫХ СТОЛБЦОВ ДЛЯ КОРРЕЛЯЦИИ ----------
df_numeric = df_merged.select_dtypes(include=[np.number])

# ---------- 7) СЧИТАЕМ КОРРЕЛЯЦИЮ И СТРОИМ ТЕПЛОВУЮ КАРТУ ----------
corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation heatmap (all datasets merged by date)")
plt.tight_layout()
plt.show()
