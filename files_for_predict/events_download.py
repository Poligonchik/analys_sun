
import requests
import json
import os
import pandas as pd
from datetime import datetime

def download_events():
    """
    Загружает данные по URL и сохраняет их в файл events_download.json.
    """
    url = "https://services.swpc.noaa.gov/json/edited_events.json"
    filename = "tables/events_download.json"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на статус 200 OK
    except requests.RequestException as e:
        print(f"Ошибка при загрузке данных: {e}")
        return False

    data = response.json()

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Файл успешно сохранён как: {os.path.abspath(filename)}")
    return True

def extract_time(dt_str):
    """
    Извлекает время из строки ISO datetime и возвращает его в формате HHMM.
    """
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%H%M")
    except Exception:
        return "ND"

def extract_date(dt_str):
    """
    Извлекает дату из строки ISO datetime и возвращает её в формате "YYYY MM DD".
    """
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y %m %d")
    except Exception:
        return "ND"

def format_region(region):
    """
    Форматирует значение region как 4-значную строку с ведущими нулями.
    Если region отсутствует, возвращает "0000".
    """
    if region is None:
        return "0000"
    try:
        return f"{int(region):04d}"
    except Exception:
        return str(region)

def transform_event(event):
    """
    Преобразует событие из исходного формата в новый формат с полями:
    begin, max, end, type, loc_freq, particulars, date, region.
    """
    new_event = {}
    new_event["begin"] = extract_time(event.get("begin_datetime", ""))
    new_event["max"] = extract_time(event.get("max_datetime", ""))
    new_event["end"] = extract_time(event.get("end_datetime", ""))
    new_event["type"] = event.get("type", "ND")
    new_event["loc_freq"] = event.get("frequency", "ND")
    new_event["particulars"] = event.get("particulars1", "")
    new_event["date"] = extract_date(event.get("begin_datetime", ""))
    new_event["region"] = format_region(event.get("region"))
    return new_event

def time_str_to_minutes(t):
    """
    Преобразует строку времени формата HHMM в минуты от начала дня.
    """
    if pd.isna(t):
        return pd.NA
    try:
        t_str = str(t).zfill(4)
        hh = int(t_str[:2])
        mm = int(t_str[2:])
        return hh * 60 + mm
    except Exception:
        return pd.NA

def minutes_to_time_str(m):
    """
    Преобразует минуты от начала дня в строку формата HHMM.
    """
    try:
        m = int(round(m))
        hh = m // 60
        mm = m % 60
        return f"{hh:02d}{mm:02d}"
    except Exception:
        return pd.NA

def compute_interval(row):
    """
    Интервал между begin и end
    """
    b = row['begin_mins']
    e = row['end_mins']
    if pd.notna(b) and pd.notna(e):
        return e - b if e >= b else e + 1440 - b
    else:
        return pd.NA

def transform_events():
    input_filename = "tables/events_download.json"
    output_filename = "tables/events_download.json"

    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    transformed = [transform_event(e) for e in data]

    # заменяем пустые строки и "ND" на pd.NA
    df = pd.DataFrame(transformed)
    df = df.replace({"": pd.NA, "ND": pd.NA})

    # время из строк в минуты от начала дня
    df['begin_mins'] = df['begin'].apply(time_str_to_minutes)
    df['max_mins']   = df['max'].apply(time_str_to_minutes)
    df['end_mins']   = df['end'].apply(time_str_to_minutes)

    # Вычисляем интервал между begin и end
    df['interval'] = df.apply(compute_interval, axis=1)

    # Вычисляем медианный интервал
    valid_intervals = df['interval'].dropna()
    median_interval = valid_intervals.median() if not valid_intervals.empty else 13
    print("Медианный интервал (минут):", median_interval)

    # Если отсутствует end, заменяем его как begin + медианный интервал
    mask_end_missing = df['begin_mins'].notna() & df['end_mins'].isna()
    df.loc[mask_end_missing, 'end_mins'] = (df.loc[mask_end_missing, 'begin_mins'] + median_interval) % 1440

    # Если отсутствует begin, заменяем его как end - медианный интервал
    mask_begin_missing = df['end_mins'].notna() & df['begin_mins'].isna()
    df.loc[mask_begin_missing, 'begin_mins'] = (df.loc[mask_begin_missing, 'end_mins'] - median_interval) % 1440

    df['interval'] = df.apply(compute_interval, axis=1)

    # Функция для вычисления доли прохождения интервала от начала до max
    def compute_progress(row):
        b = row['begin_mins']
        e = row['end_mins']
        m = row['max_mins']
        if pd.notna(b) and pd.notna(e) and pd.notna(m) and pd.notna(row['interval']) and row['interval'] != 0:
            diff = m - b if m >= b else m + 1440 - b
            return diff / row['interval']
        else:
            return pd.NA

    df['progress'] = df.apply(compute_progress, axis=1)
    valid_progress = df['progress'].dropna()
    average_progress = valid_progress.mean() if not valid_progress.empty else 0.5
    print("Среднее значение progress (доля):", average_progress)

    # Если max отсутствует, вычисляем его как begin + (average_progress * interval)
    mask_max_missing = df['max_mins'].isna() & df['begin_mins'].notna() & df['end_mins'].notna()
    df.loc[mask_max_missing, 'max_mins'] = (df.loc[mask_max_missing, 'begin_mins'] +
                                             average_progress * df.loc[mask_max_missing, 'interval']) % 1440

    # Преобразуем минуты обратно в строковый формат HHMM
    df['begin'] = df['begin_mins'].apply(minutes_to_time_str)
    df['end']   = df['end_mins'].apply(minutes_to_time_str)
    df['max']   = df['max_mins'].apply(minutes_to_time_str)

    # Удаляем временные столбцы
    df.drop(columns=['begin_mins', 'max_mins', 'end_mins', 'interval', 'progress'], inplace=True)

    # Еще раз заменяем  оставшиеся "ND" или пустые строки на pd.NA
    df = df.replace({"": pd.NA, "ND": pd.NA})

    final_json = df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print(f"Преобразование завершено. Итоговый файл: {output_filename}")

    missing_counts = df.isna().sum()
    print("\nКоличество пропущенных значений по столбцам:")
    print(missing_counts)

def main():
    if download_events():
        transform_events()

if __name__ == "__main__":
    main()
