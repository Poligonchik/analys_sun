import json
import pandas as pd
import numpy as np
from collections import Counter

json_path = "../processed_results/combined_events_all.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Для каждого файла с датой и годом берем каждое событие и добавляем дату и год
flat_events = []
for file_record in data:
    file_date = file_record.get("date")
    file_year = file_record.get("year")
    events = file_record.get("events", [])
    for ev in events:
        event_record = ev.copy()
        event_record["date"] = file_date
        event_record["year"] = file_year
        flat_events.append(event_record)

df = pd.DataFrame(flat_events)

# Приводим пустые строки к NaN
df.replace("", np.nan, inplace=True)

print("Общая информация о DataFrame:")
print(df.info())
print("\nПервые 5 строк:")
print(df.head())

# Функция для подсчёта типов в столбце
def count_types(series):
    type_counter = Counter()
    for x in series.dropna():
        type_counter[type(x).__name__] += 1
    return dict(type_counter)

print("\nАнализ каждого столбца:")
for col in df.columns:
    print(f"\nСтолбец: {col}")
    total = len(df)
    missing = df[col].isna().sum()
    print(f"  Всего значений: {total}, Пропущено: {missing} ({missing/total:.2%})")
    unique_count = df[col].nunique(dropna=True)
    print(f"  Уникальных значений: {unique_count}")
    if df[col].dtype == "object":
        print("  Топ-5 значений:")
        print(df[col].value_counts().head(10))
        print("  Топ-10 наименее распространённых значений:")
        bottom_counts = df[col].value_counts(ascending=True).head(10) # Берём все значения и сортируем по возрастанию частоты
        print(bottom_counts)
    else:
        print("  Статистика:")
        print(df[col].describe())
    type_counts = count_types(df[col])
    print(f"  Распределение типов: {type_counts}")

# Если есть числовые столбцы - посчитаем корреляции
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    print("\nКорреляция между числовыми столбцами:")
    print(df[num_cols].corr())

if "year" in df.columns:
    events_per_year = df.groupby("year").size()
    print("\nКоличество событий по годам:")
    print(events_per_year)
