import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка SRS данных
srs_df = pd.read_json("../result_json/srs.json")
srs_df['Lo'] = pd.to_numeric(srs_df['Lo'], errors='coerce')
srs_df['Area'] = pd.to_numeric(srs_df['Area'], errors='coerce')
srs_df['LL'] = pd.to_numeric(srs_df['LL'], errors='coerce')
srs_df['NN'] = pd.to_numeric(srs_df['NN'], errors='coerce')

srs_agg = srs_df.groupby('date').agg({
    'Lo': 'mean',
    'Area': 'mean',
    'LL': 'mean',
    'NN': 'mean',
    'Nmbr': 'count'
}).rename(columns={'Nmbr': 'srs_count'}).reset_index()

# Загрузка событий
events_df = pd.read_json("../result_json/events.json")

events_df['begin'] = pd.to_numeric(events_df['begin'], errors='coerce')
events_df['max'] = pd.to_numeric(events_df['max'], errors='coerce')
events_df['end'] = pd.to_numeric(events_df['end'], errors='coerce')

# Извлекаем класс вспышки из particulars
def extract_flare_class(particulars):
    if pd.isna(particulars):
        return "None"
    part = str(particulars).strip().upper()
    for cl in ["X", "M", "C", "B", "A"]:
        if part.startswith(cl):
            return cl
    return "None"

events_df['flare_class'] = events_df['particulars'].apply(extract_flare_class)

# Считаем количество вспышек по классам по каждой дате
flare_counts = events_df[events_df['flare_class'].isin(["A", "B", "C", "M", "X"])]
flare_counts = flare_counts.groupby(['date', 'flare_class']).size().unstack(fill_value=0).reset_index()

# Дополнительно агрегируем числовые поля
def parse_loc_freq(freq_str):
    try:
        parts = freq_str.split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return float(freq_str)
    except:
        return None

events_df['loc_freq_mean'] = events_df['loc_freq'].apply(parse_loc_freq)

events_agg = events_df.groupby('date').agg({
    'begin': 'mean',
    'max': 'mean',
    'end': 'mean',
    'loc_freq_mean': 'mean'
}).reset_index()

# Объединяем всё
events_full = pd.merge(events_agg, flare_counts, on='date', how='left').fillna(0)
merged_df = pd.merge(srs_agg, events_full, on='date', how='inner')

print("Объединённые данные (первые 5 строк):")
print(merged_df.head())

# Корреляционная матрица
corr_matrix = merged_df.corr()
print("\nКорреляционная матрица:")
print(corr_matrix)

# Визуализация
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Корреляции SRS и Flare Classes")
plt.tight_layout()
plt.show()
