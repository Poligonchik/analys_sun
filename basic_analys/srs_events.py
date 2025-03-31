import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

# --- Загрузка данных ---
# Загружаем SRS данные (обработанный srs.json, который является списком записей)
srs_df = pd.read_json("../result_json/srs.json")  # скорректируйте путь при необходимости

# Приводим числовые столбцы SRS к числовому типу
srs_df['Lo'] = pd.to_numeric(srs_df['Lo'], errors='coerce')
srs_df['Area'] = pd.to_numeric(srs_df['Area'], errors='coerce')
srs_df['LL'] = pd.to_numeric(srs_df['LL'], errors='coerce')
srs_df['NN'] = pd.to_numeric(srs_df['NN'], errors='coerce')

# Для SRS сгруппируем по дате: вычислим средние значения по числовым полям и количество записей за дату
srs_agg = srs_df.groupby('date').agg({
    'Lo': 'mean',
    'Area': 'mean',
    'LL': 'mean',
    'NN': 'mean',
    'Nmbr': 'count'
}).rename(columns={'Nmbr': 'srs_count'}).reset_index()

# Загружаем данные о событиях (events)
events_df = pd.read_json("../result_json/events.json")  # измените путь, если требуется

# Приводим поля begin, max, end к числовому типу
events_df['begin'] = pd.to_numeric(events_df['begin'], errors='coerce')
events_df['max'] = pd.to_numeric(events_df['max'], errors='coerce')
events_df['end'] = pd.to_numeric(events_df['end'], errors='coerce')

# Функция для обработки поля loc_freq (например, "020-140")
def parse_loc_freq(freq_str):
    try:
        parts = freq_str.split('-')
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
        else:
            return None
    except:
        return None

events_df['loc_freq_mean'] = events_df['loc_freq'].apply(parse_loc_freq)

# Группируем события по дате: считаем число событий и средние значения числовых полей
events_agg = events_df.groupby('date').agg({
    'begin': 'mean',
    'max': 'mean',
    'end': 'mean',
    'loc_freq_mean': 'mean',
    'type': 'count'
}).rename(columns={'type': 'events_count'}).reset_index()

# --- Объединение данных по дате ---
merged_df = pd.merge(srs_agg, events_agg, on='date', how='inner')
print("Объединённые данные (первые 5 строк):")
print(merged_df.head())

# --- Расчёт корреляционной матрицы ---
corr_matrix = merged_df.corr()
print("\nКорреляционная матрица:")
print(corr_matrix)

# --- Визуализация: тепловая карта корреляций ---
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Aggregated SRS and Events Data")
plt.tight_layout()
plt.show()
