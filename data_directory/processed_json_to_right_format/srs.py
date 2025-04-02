import os
import json
from datetime import datetime
from collections import Counter


def parse_issued_date(issued_str):
    """
    Преобразует строку вида '2000 Oct 14 0030 UTC'
    в объект datetime, удаляя 'UTC' и парся по формату.
    """
    try:
        issued_clean = issued_str.replace(" UTC", "")
        dt = datetime.strptime(issued_clean, "%Y %b %d %H%M")
        return dt
    except Exception as e:
        print(f"Ошибка при парсинге даты '{issued_str}': {e}")
        return None


def process_section(section):
    """
    Извлекает дату из поля 'issued' и добавляет новое поле 'date' в формате 'YYYY MM DD'
    Если поле 'Mag_Type' отсутствует или пустое, заменяет его на 'Beta' (самое частое)
    Удаляет поля 'Z' и 'product'
    """
    for record in section:
        issued = record.get("issued")
        if issued:
            dt = parse_issued_date(issued)
            if dt:
                record["date"] = dt.strftime("%Y %m %d")
            else:
                record["date"] = "unknown"
        else:
            record["date"] = "unknown"
        record.pop("issued", None)
        if not record.get("Mag_Type"):
            record["Mag_Type"] = "Beta"
        record.pop("Z", None)
        record.pop("product", None)
    return section


def impute_mode_for_field(section, field_name):
    """
    Для заданного поля в секции вычисляет моду
    и заменяет пропуск на это значение.
    Выводит моду и количество её повторений.
    """
    values = [record.get(field_name) for record in section if record.get(field_name) not in [None, ""]]
    if not values:
        print(f"Поле '{field_name}' не содержит значимых значений для вычисления моды.")
        return section
    counter = Counter(values)
    mode_value, mode_count = counter.most_common(1)[0]
    print(f"Для поля '{field_name}': мода = {mode_value}, встречается {mode_count} раз.")
    for record in section:
        if record.get(field_name) in [None, ""]:
            record[field_name] = mode_value
    return section


def fill_missing_values_for_section(section):
    """
    Проходит по всем ключам в секции и для каждого поля заменяет пропуски
    на моду этого поля.
    """
    keys = set()
    for record in section:
        keys.update(record.keys())
    for key in keys:
        section = impute_mode_for_field(section, key)
    return section


def flatten_section(section):
    """
    Если секция является словарем, раскрывает вложенность
    и возвращает плоский список записей.
    """
    if isinstance(section, dict):
        flat_list = []
        for sub in section.values():
            if isinstance(sub, list):
                flat_list.extend(sub)
            else:
                flat_list.append(sub)
        return flat_list
    return section


def process_srs_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Извлекаем только нужную секцию
    regions = data.get("regions_with_sunspots", [])

    print("Обработка секции: regions_with_sunspots")
    regions = process_section(regions)
    regions = fill_missing_values_for_section(regions)
    regions = flatten_section(regions)

    result = regions

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Обработанный файл сохранен по пути: {output_path}")
    return result


def analyze_section(section, section_name="", sample_size=50):
    """
    Общее число записей,
    Для каждого ключа: число пропусков и количество уникальных значений.
    Выводит первые sample_size уникальных значений.
    """
    print(f"\nАнализ секции: {section_name}")
    total = len(section)
    keys = set()
    for record in section:
        keys.update(record.keys())
    for key in keys:
        missing = sum(1 for record in section if key not in record or record[key] in [None, ""])
        unique_values = {record.get(key) for record in section if record.get(key) not in [None, ""]}
        unique_list = list(unique_values)
        if len(unique_list) > sample_size:
            unique_list = unique_list[:sample_size]
        print(f"Ключ: '{key}' - Всего: {total}, Пропусков: {missing}, Уникальных значений: {len(unique_values)}")
        print(f"  Первые {sample_size} уникальных значений: {unique_list}")


def analyze_data(data):
    if isinstance(data, list):
        analyze_section(data, section_name="regions_with_sunspots")
    elif isinstance(data, dict):
        for key, section in data.items():
            analyze_section(section, section_name=key)


if __name__ == '__main__':
    input_file = "../processed_results/combined_srs_all.json"
    output_file = "../../result_json/srs.json"

    processed_data = process_srs_file(input_file, output_file)
    analyze_data(processed_data)

