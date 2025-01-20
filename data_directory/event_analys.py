import json
import pandas as pd

# Загрузка JSON файла
file_path = "processed_results/combined_events.json"
with open(file_path, "r") as file:
    events_data = json.load(file)

# Разворачивание событий из вложенной структуры
all_events = []
for day_data in events_data:
    date = day_data.get("date", "Unknown")  # Дата
    for event in day_data.get("events", []):  # Список событий
        event["date"] = date  # Добавляем дату в каждое событие
        all_events.append(event)

# Преобразование в DataFrame
events_df = pd.DataFrame(all_events)

# Базовый анализ
# 1. Общее количество событий
total_events = len(events_df)
print(f"Общее количество событий: {total_events}")

# 2. Распределение по типам событий
if "type" in events_df.columns:
    event_type_distribution = events_df["type"].value_counts()
    print("\nРаспределение событий по типам:")
    print(event_type_distribution)

# 3. Топ-5 регионов с наибольшим количеством событий
if "region" in events_df.columns:
    top_regions = events_df["region"].value_counts().head(5)
    print("\nТоп-5 регионов с наибольшим количеством событий:")
    print(top_regions)

# 4. События с наибольшей интенсивностью
if "particulars" in events_df.columns:
    def extract_intensity(particulars):
        if isinstance(particulars, str) and "E" in particulars:
            try:
                # Преобразуем строку в число, например: B6.9 -> 6.9E-04 -> 0.00069
                parts = particulars.split("E")
                base = float(parts[0]) if parts[0] else 0
                exponent = int(parts[1]) if len(parts) > 1 else 0
                return base * (10 ** exponent)
            except (ValueError, IndexError):
                return None
        return None

    # Извлечение и обработка интенсивности
    events_df["intensity"] = events_df["particulars"].apply(extract_intensity)
    events_df["intensity"] = pd.to_numeric(events_df["intensity"], errors="coerce")  # Преобразование в числовой тип

    # Топ-5 событий с наибольшей интенсивностью
    if not events_df["intensity"].isnull().all():  # Проверка на наличие данных
        top_intensity_events = events_df.nlargest(5, "intensity")
        print("\nСобытия с наибольшей интенсивностью:")
        print(top_intensity_events[["event", "begin", "end", "intensity"]])
    else:
        print("\nНет данных о интенсивности для анализа.")

# 5. Распределение событий по часам
if "begin" in events_df.columns:
    events_df["hour"] = events_df["begin"].str[:2]
    hour_distribution = events_df["hour"].value_counts().sort_index()
    print("\nРаспределение событий по часам:")
    print(hour_distribution)

# Сохранение анализа в файл
analysis_results = {
    "total_events": total_events,
    "event_type_distribution": event_type_distribution.to_dict() if "type" in events_df.columns else None,
    "top_regions": top_regions.to_dict() if "region" in events_df.columns else None,
    "hour_distribution": hour_distribution.to_dict() if "begin" in events_df.columns else None,
}

output_path = "events_analysis.json"
with open(output_path, "w") as analysis_file:
    json.dump(analysis_results, analysis_file, indent=4)

print(f"\nАнализ завершен. Результаты сохранены в '{output_path}'.")
