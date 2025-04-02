import os
import json
import datetime
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def reformat_date(date_str):
    """
    Преобразует дату из формата "YYYY-MM-DD" в формат "YYYY MM DD".
    """
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y %m %d")
    except Exception as e:
        print(f"Ошибка при обработке даты '{date_str}': {e}")
        return date_str

def clean_value(value):
    """
    Чищу немного
    """
    if isinstance(value, str) and value.strip() == "*":
        return "NaN"
    if value is None or (isinstance(value, str) and value.strip().lower() == "none") or \
       (isinstance(value, str) and value.strip() == ""):
        return "NaN"
    elif isinstance(value, list):
        return [clean_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items() if k != "op_10cm"}
    else:
        return value

def process_dsd_records(records):
    """
    Приводит значение поля "date" к формату "YYYY MM DD".
    """
    for record in records:
        if "date" in record:
            record["date"] = reformat_date(record["date"])
        for key, value in record.items():
            record[key] = clean_value(value)
    return records

def flatten_flares(records):
    """
    Переносит вложенные поля словаря "flares" на уровень верхнего уровня с префиксом "flares.".
    """
    for record in records:
        if "flares" in record and isinstance(record["flares"], dict):
            for key, value in record["flares"].items():
                record[f"flares.{key}"] = value
            del record["flares"]
    return records

def shift_flares(records):
    """
    Сдвигает значения для вспышек следующим образом:
      - Значение из поля "background" переносится в "flares.C".
      - Старое значение "flares.C" записывается в "flares.M".
      - Старое значение "flares.M" записывается в "flares.X".
      - Старое значение "flares.X" записывается в "flares.S".
    После сдвига удаляет поле "background".
    Потому что я случайно его добавила на 1 этапе и поздно заметила, что для модели это ключевой признак самый, а
    ключевой на самом деле flares.C и парсила не так
    """
    for record in records:
        background_val = record.get("background", 0.0)
        flares_C_val = record.get("flares.C", 0.0)
        flares_M_val = record.get("flares.M", 0.0)
        flares_X_val = record.get("flares.X", 0.0)
        # Сдвиг: background -> flares.C, flares.C -> flares.M, flares.M -> flares.X, flares.X -> flares.S
        record["flares.C"] = background_val
        record["flares.M"] = flares_C_val
        record["flares.X"] = flares_M_val
        record["flares.S"] = flares_X_val
        if "background" in record:
            del record["background"]
    return records


def analyze_field(records, field, sample_size=5):
    """
    Выводит статистическую информацию по указанному полю:
      - Общее число записей,
      - Количество значений "NaN",
      - Количество уникальных значений
      - sample_size наиболее частых значений.
    """
    values = [record.get(field, "NaN") for record in records]
    nan_count = sum(1 for v in values if v == "NaN")

    def to_hashable(val):
        if isinstance(val, (dict, list)):
            return json.dumps(val, sort_keys=True)
        else:
            return val

    hashable_values = list(map(to_hashable, values))
    unique_vals = set(hashable_values)
    most_common = Counter(hashable_values).most_common(sample_size)

    print(f"Поле: '{field}'")
    print(f"Общее число записей: {len(records)}")
    print(f"Количество значений 'NaN': {nan_count}")
    print(f"Количество уникальных значений: {len(unique_vals)}")
    print(f"{sample_size} наиболее частых значений: {most_common}")

def analyze_nested_field(records, parent_field, sample_size=5):
    nested_values = {}
    for record in records:
        nested = record.get(parent_field)
        if isinstance(nested, dict):
            for key, value in nested.items():
                nested_values.setdefault(key, []).append(value)
    for key, values in nested_values.items():
        nan_count = sum(1 for v in values if v == "NaN")

        def to_hashable(val):
            if isinstance(val, (dict, list)):
                return json.dumps(val, sort_keys=True)
            else:
                return val

        hashable_values = list(map(to_hashable, values))
        unique_vals = set(hashable_values)
        most_common = Counter(hashable_values).most_common(sample_size)
        print(f"Вложенное поле: '{parent_field}.{key}'")
        print(f"Общее число записей: {len(values)}")
        print(f"Количество значений 'NaN': {nan_count}")
        print(f"Количество уникальных значений: {len(unique_vals)}")
        print(f"{sample_size} наиболее частых значений: {most_common}")

def build_correlation_table(records, title="Тепловая карта корреляций числовых полей"):
    """
    Создаёт DataFrame из записей, заменяет строковые "NaN" на np.nan, приводит столбцы к числовому типу,
    выбирает числовые столбцы, вычисляет корреляционную матрицу и строит тепловую карту.
    """
    df = pd.DataFrame(records)
    df.replace("NaN", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    df_numeric = df.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr()

    print("\nОбщая таблица корреляций для числовых полей:")
    print(corr_matrix)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def impute_top_level_fields(records):
    """
    Заполняет пропуски в полях верхнего уровня:
      - radio_flux: заполняется медианой (если есть пропуски).
      - background: заполняется медианой.
      - solar_field: оставляем без изменений.
    Возвращает обновлённый список записей.
    """
    df = pd.DataFrame(records)
    df.replace("NaN", np.nan, inplace=True)
    for col in ["radio_flux"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    return df.to_dict(orient="records")

def impute_nested_flares(records):
    """
    Для вложенного словаря 'flares', для каждого ключа (C, M, X, S) заменяет значения "NaN"
    на моду (наиболее частое значение) среди ненулевых значений. Если ненулевых значений нет,
    заменяет на 0.0 (так как вспышек нет).
    """
    keys = ["C", "M", "X", "S"]
    flares_values = {k: [] for k in keys}
    for record in records:
        flares = record.get("flares")
        if isinstance(flares, dict):
            for k in keys:
                flares_values[k].append(flares.get(k, "NaN"))
    flares_mode = {}
    for k, values in flares_values.items():
        non_nan = [v for v in values if v != "NaN"]
        if non_nan:
            flares_mode[k] = Counter(non_nan).most_common(1)[0][0]
        else:
            flares_mode[k] = 0.0
    for record in records:
        flares = record.get("flares")
        if isinstance(flares, dict):
            for k in keys:
                if flares.get(k, "NaN") == "NaN":
                    record["flares"][k] = flares_mode[k]
    return records

def impute_flattened_flares(records):
    """
    Для полей "flares.C", "flares.M", "flares.X" и "flares.S", если значение равно "NaN" или np.nan,
    заменяет его на 0.0 (так как отсутствуют вспышки).
    """
    keys = ["flares.C", "flares.M", "flares.X", "flares.S"]
    for record in records:
        for k in keys:
            val = record.get(k, "NaN")
            if pd.isna(val) or val == "NaN":
                record[k] = 0.0
    return records

if __name__ == '__main__':
    input_file = "../processed_results/DSD_all.json"
    output_file = "../../result_json/dsd.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Приведение даты к единому формату
    processed_data = process_dsd_records(data)

    # Развернём вложенные поля "flares"
    processed_data = flatten_flares(processed_data)

    # Выполняем сдвиг значений: background -> flares.C, flares.C -> flares.M, flares.M -> flares.X, flares.X -> flares.S
    # Потому что я только потом заметила, что парсила неверно
    processed_data = shift_flares(processed_data)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"Обработанный файл сохранен по пути: {output_file}")

    print("Корреляционная матрица до импьютации:")
    build_correlation_table(processed_data, title="До импьютации")

    print("Статистика по каждому полю до импьютации:")
    fields = set()
    for record in processed_data:
        fields.update(record.keys())
    for field in fields:
        analyze_field(processed_data, field, sample_size=5)

    processed_data = impute_top_level_fields(processed_data)

    processed_data = impute_flattened_flares(processed_data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f"Итоговый обработанный файл сохранен по пути: {output_file}")

    print("Корреляционная матрица после импьютации:")
    build_correlation_table(processed_data, title="После импьютации")

    print("Статистика по каждому полю после импьютации:")
    fields = set()
    for record in processed_data:
        fields.update(record.keys())
    for field in fields:
        analyze_field(processed_data, field, sample_size=5)
