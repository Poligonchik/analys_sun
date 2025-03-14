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
    Для каждой записи в секции (например, 'regions_with_sunspots'):
      - Извлекает дату из поля 'issued' и добавляет новое поле 'date' в формате 'YYYY MM DD'
      - Удаляет поле 'issued'
      - Если поле 'Mag_Type' отсутствует или пустое, заменяет его на 'Beta'
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
        # Удаляем поле "issued"
        if "issued" in record:
            del record["issued"]
        # Обработка поля "Mag_Type": если отсутствует или пустое, заменяем на "Beta"
        if "Mag_Type" in record:
            if record["Mag_Type"] in [None, ""]:
                record["Mag_Type"] = "Beta"
        else:
            record["Mag_Type"] = "Beta"
    return section

def impute_mode_for_field(section, field_name):
    """
    Для заданного поля в секции вычисляет моду (наиболее часто встречающееся значение)
    и заменяет пропуски (None или пустые строки) на это значение.
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

def process_srs_file(input_path, output_path):
    """
    Загружает SRS JSON, обрабатывает все секции:
      - Приводит дату к единому формату и удаляет поле 'issued'
      - Сначала обрабатывает специальные поля (например, 'Mag_Type')
      - Затем заполняет пропуски для всех ключей модой
    Сохраняет результат в новый JSON.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in data.keys():
        if isinstance(data[key], list):
            print(f"Обработка секции: {key}")
            data[key] = process_section(data[key])
            # Заполнение пропусков для всех полей данной секции
            data[key] = fill_missing_values_for_section(data[key])
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Обработанный файл сохранен по пути: {output_path}")
    return data

def analyze_section(section, section_name="", sample_size=50):
    """
    Для списка записей (секции) выводит анализ:
      - Общее число записей,
      - Для каждого ключа: число пропусков и количество уникальных значений.
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
    """
    Вызывает analyze_section для каждой секции в data.
    """
    for key, section in data.items():
        if isinstance(section, list):
            analyze_section(section, section_name=key)

if __name__ == '__main__':
    # Пути: исходный файл и куда положить результат
    input_file = "../processed_results/combined_srs_all.json"
    output_file = "../../unified_json/srs.json"

    processed_data = process_srs_file(input_file, output_file)
    analyze_data(processed_data)

'''
все пропуски заполняются модой для значений, дял которых нет медианы
'''