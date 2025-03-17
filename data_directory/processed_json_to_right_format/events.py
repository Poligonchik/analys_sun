import os
import json
import re
from datetime import datetime
import pandas as pd

def parse_date(date_str):
    """
    Преобразует дату в различных форматах в стандартный формат "YYYY MM DD".
    Если дата уже в нужном формате (например, "1996 10 05"), возвращает её без изменений.
    """
    date_formats = [
        "%Y %m %d",  # "1996 10 05"
        "%Y %b %d",  # "2000 Oct 14"
        "%Y-%m-%d",  # "1996-10-05"
        "%Y/%m/%d",  # "1996/10/05"
        "%d %b %Y",  # "05 Oct 1996"
        "%d %m %Y"   # "05 10 1996"
    ]
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y %m %d")
        except Exception:
            continue
    return date_str

def process_events_list(events_list):
    """
    Обрабатывает список записей из combined_events_all.json.
    Для каждого объекта, если есть поле "date", приводит его к формату "YYYY MM DD".
    Также удаляет поле "year". Для каждого события удаляются поля "obs" и "q".
    """
    for record in events_list:
        if "date" in record:
            record["date"] = parse_date(record["date"])
        record.pop("year", None)
        if "events" in record and isinstance(record["events"], list):
            for event in record["events"]:
                event.pop("year", None)
                event.pop("obs", None)
                event.pop("q", None)
                event.pop("event", None)
    return events_list

def process_events_file(input_path, output_path):
    """
    Загружает JSON-файл с событиями, обрабатывает каждый объект (приводит дату к формату "YYYY MM DD" и удаляет поле "year" и поля "obs", "q"),
    сохраняет результат в новый JSON и возвращает данные.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        print("Обработка списка событий (combined_events_all.json)")
        data = process_events_list(data)
    else:
        print("Ожидался список объектов, а получен другой тип.")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Обработанный файл событий сохранен по пути: {output_path}")
    return data

def analyze_list_data(data_list, sample_size=50):
    """
    Выполняет анализ для верхнеуровневых ключей каждого объекта.
    Для каждого ключа выводит общее число записей, число пропусков и количество уникальных значений.
    Если значение является списком, оно преобразуется в строку через json.dumps.
    Для ключа "events" уникальные значения не выводятся полностью.
    """
    print("\nАнализ данных (верхнеуровневые ключи для каждого объекта в списке)")
    total = len(data_list)
    keys = set()
    for record in data_list:
        keys.update(record.keys())
    for key in keys:
        missing = 0
        unique_values = set()
        for record in data_list:
            val = record.get(key)
            if val in [None, ""]:
                missing += 1
            else:
                if isinstance(val, list):
                    try:
                        val_str = json.dumps(val, sort_keys=True)
                    except Exception:
                        val_str = str(val)
                    unique_values.add(val_str)
                else:
                    unique_values.add(val)
        if key == "events":
            print(f"Ключ: '{key}' - Всего: {total}, Пропусков: {missing}, Уникальных значений: {len(unique_values)}")
            print(f"  (Список уникальных значений для '{key}' не выводится, так как он очень объёмный)")
        else:
            unique_list = list(unique_values)
            if len(unique_list) > sample_size:
                unique_list = unique_list[:sample_size]
            print(f"Ключ: '{key}' - Всего: {total}, Пропусков: {missing}, Уникальных значений: {len(unique_values)}")
            print(f"  Первые {sample_size} уникальных значений: {unique_list}")

def analyze_events_data(data_list, sample_size=50):
    """
    Выполняет анализ для вложенных объектов (событий) из поля "events".
    Для каждого ключа выводит общее число событий, число пропусков (учитываются None, пустая строка и "////")
    и количество уникальных значений.
    """
    print("\nАнализ данных для событий (вложенных объектов в поле 'events')")
    total = len(data_list)
    keys = set()
    for event in data_list:
        keys.update(event.keys())
    for key in keys:
        missing = 0
        unique_values = set()
        for event in data_list:
            val = event.get(key)
            if val in [None, "", "////", "///"]:
                missing += 1
            else:
                unique_values.add(val)
        unique_list = list(unique_values)
        if len(unique_list) > sample_size:
            unique_list = unique_list[:sample_size]
        print(f"Ключ: '{key}' - Всего событий: {total}, Пропусков: {missing}, Уникальных значений: {len(unique_values)}")
        print(f"  Первые {sample_size} уникальных значений: {unique_list}")

def time_str_to_minutes(time_str):
    """
    Преобразует строку с временем (например, "1234") в количество минут от полуночи.
    Из строки удаляются все нецифровые символы.
    Если строка пуста или не содержит цифр, возвращается None.
    """
    if pd.isna(time_str):
        return None
    clean = re.sub(r'\D', '', time_str)
    if clean == "":
        return None
    clean = clean.zfill(4)
    try:
        hours = int(clean[:2])
        minutes = int(clean[2:4])
        return hours * 60 + minutes
    except Exception:
        return None

def minutes_to_time_str(minutes):
    """
    Преобразует количество минут от полуночи в строку формата HHMM.
    Если значение NaN, возвращает пустую строку.
    """
    if pd.isna(minutes):
        return ""
    minutes = int(round(minutes)) % 1440
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}{m:02d}"

if __name__ == '__main__':
    # Пути: исходный JSON и куда сохранить обработанный JSON
    input_file = "../processed_results/combined_events_all.json"
    output_file = "../../unified_json/events.json"
    output_filled_file = "../../unified_json/events.json"

    # Обработка файла: приведение даты к формату "YYYY MM DD" и удаление поля "year" и полей "obs", "q"
    processed_events = process_events_file(input_file, output_file)

    # Анализ верхнеуровневых данных
    analyze_list_data(processed_events)

    # «Разворачиваем» поле events: собираем все события в один список
    all_events = []
    for record in processed_events:
        date_value = record.get("date")
        events = record.get("events", [])
        if isinstance(events, list):
            for event in events:
                if "date" not in event and date_value:
                    event["date"] = date_value
                all_events.append(event)
    print(f"\nОбщее количество событий (развернуто): {len(all_events)}")

    # Анализ вложенных событий до замены пропусков
    analyze_events_data(all_events)

    # Создаем DataFrame из всех событий (без заполнения пропусков)
    df_events = pd.DataFrame(all_events)

    # Заменяем строки "////" и "///" на NaN
    df_events.replace({"////": pd.NA, "///": pd.NA}, inplace=True)

    # --- Специальная обработка столбцов 'begin' и 'end' ---
    # Преобразуем строки времени в минуты и затем обратно в формат HHMM, удаляя буквенные префиксы
    df_events['begin_mins'] = df_events['begin'].apply(time_str_to_minutes)
    df_events['end_mins'] = df_events['end'].apply(time_str_to_minutes)

    # Функция для вычисления интервала (с учетом перехода через полночь)
    def compute_interval(row):
        b = row['begin_mins']
        e = row['end_mins']
        if pd.notna(b) and pd.notna(e):
            diff = e - b
            if diff < 0:
                diff += 1440
            return diff
        else:
            return None

    df_events['interval'] = df_events.apply(compute_interval, axis=1)
    # Здесь НЕ заполняем пропуски – оставляем исходные значения
    # Преобразуем минуты обратно в строковый формат HHMM
    df_events['begin'] = df_events['begin_mins'].apply(minutes_to_time_str)
    df_events['end'] = df_events['end_mins'].apply(minutes_to_time_str)

    # Удаляем временные столбцы
    df_events.drop(columns=['begin_mins', 'end_mins', 'interval'], inplace=True)
    # --- Конец специальной обработки begin/end ---

    print("\nПосле специальной обработки begin/end (без заполнения пропусков):")
    filled_events = df_events.to_dict(orient="records")
    analyze_events_data(filled_events)

    # Сохраняем результат в JSON для дальнейшего анализа
    with open(output_filled_file, "w", encoding="utf-8") as f:
        json.dump(filled_events, f, ensure_ascii=False, indent=4)
    print(f"\nВложенные данные событий сохранены в {output_filled_file}")
