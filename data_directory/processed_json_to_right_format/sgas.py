import os
import json
from datetime import datetime


def parse_issued_date(issued_str):
    """
    Преобразует строку вида "2013 jul 24 0245 utc" или "2005 apr 22 0250 ut"
    в объект datetime.
    Для этого переводим строку в нижний регистр, удаляем окончание " utc" или " ut",
    затем приводим название месяца к виду с заглавной буквой и парсим дату.
    """
    try:
        issued_clean = issued_str.lower().strip()
        if issued_clean.endswith(" utc"):
            issued_clean = issued_clean[:-4]
        elif issued_clean.endswith(" ut"):
            issued_clean = issued_clean[:-3]
        parts = issued_clean.split()
        if len(parts) < 4:
            print(f"Некорректный формат даты: {issued_str}")
            return None
        parts[1] = parts[1].capitalize()
        new_str = " ".join(parts)
        dt = datetime.strptime(new_str, "%Y %b %d %H%M")
        return dt
    except Exception as e:
        print(f"Ошибка при парсинге даты '{issued_str}': {e}")
        return None


def process_records(records):
    """
    Для каждого словаря:
      - Парсит дату из поля "issued" и сохраняет её в виде нового поля "date"
      - Удаляет поле "issued"
    """
    for record in records:
        issued = record.get("issued")
        if issued:
            dt = parse_issued_date(issued)
            record["date"] = dt.strftime("%Y %m %d") if dt else "unknown"
        else:
            record["date"] = "unknown"
        if "issued" in record:
            del record["issued"]
    return records


def clean_value(value):
    """
    Если значение пустое (пустая строка, None или строка "none" в любом регистре),
    заменяет его на строку "NaN".
    Если значение является списком или словарем, применяется рекурсивно.
    Иначе возвращает значение без изменений.
    """
    if value is None or (isinstance(value, str) and (value.strip() == "" or value.strip().lower() == "none")):
        return "NaN"
    elif isinstance(value, list):
        return [clean_value(item) for item in value]
    elif isinstance(value, dict):
        # При обработке вложенных словарей удаляем ключ "op_10cm"
        return {k: clean_value(v) for k, v in value.items() if k != "op_10cm"}
    else:
        return value


def clean_record(record):
    """
    - Удаляет поля "geomagnetic_activity_summary", "proton_events", "op_10cm", "product" и "sgas_number"
      (из верхнего уровня), так как они не используются для прогнозирования.
    - Для остальных полей заменяет пустые значения (пустая строка, None или "none") на "NaN"
      с помощью clean_value.
    """
    for key in ["geomagnetic_activity_summary", "proton_events", "op_10cm", "product", "sgas_number"]:
        if key in record:
            del record[key]
    for key in list(record.keys()):
        record[key] = clean_value(record[key])
    return record


def clean_all_records(records):
    """
    Применяет функцию clean_record ко всем записям.
    """
    return [clean_record(record) for record in records]


def analyze_records(records, sample_size=50):
    """
    Выводит анализ списка записей по всем ключам (за исключением energetic_events).
    "NaN" считается заполненным значением.
    """
    print(f"\nВсего записей: {len(records)}")
    keys = set()
    for record in records:
        keys.update(record.keys())
    if "energetic_events" in keys:
        keys.remove("energetic_events")
    for key in keys:
        missing = sum(1 for record in records if key not in record or record[key] in [None, ""])
        unique_values = {record.get(key) for record in records if record.get(key) not in [None, ""]}
        unique_list = list(unique_values)
        if len(unique_list) > sample_size:
            unique_list = unique_list[:sample_size]
        print(
            f"\nКлюч: '{key}' - Всего: {len(records)}, Пропусков: {missing}, Уникальных значений: {len(unique_values)}")
        print(f"  Первые {sample_size} уникальных значений: {unique_list}")


def analyze_energetic_events(records, sample_size=50):
    """
    Для поля energetic_events (если оно является списком словарей)
    собирает все вложенные словари и анализирует их ключи.
    "NaN" считается заполненным значением.
    """
    all_events = []
    for record in records:
        events = record.get("energetic_events")
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    all_events.append(event)
    if not all_events:
        print("Нет вложенных данных в 'energetic_events'")
        return
    keys = set()
    for event in all_events:
        keys.update(event.keys())
    for key in keys:
        missing = sum(1 for event in all_events if key not in event or event[key] in [None, ""])
        unique_values = {event.get(key) for event in all_events if event.get(key) not in [None, ""]}
        unique_list = list(unique_values)
        if len(unique_list) > sample_size:
            unique_list = unique_list[:sample_size]
        print(
            f"\n[energetic_events] Ключ: '{key}' - Всего событий: {len(all_events)}, Пропусков: {missing}, Уникальных значений: {len(unique_values)}")
        print(f"  Первые {sample_size} уникальных значений: {unique_list}")


if __name__ == '__main__':
    input_file = "../processed_results/sgas_all.json"
    output_file = "../../unified_json/sgas.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = process_records(data)
    cleaned_data = clean_all_records(processed_data)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"Обработанный файл сохранен по пути: {output_file}")

    print("\nАнализ основных полей после замены пустых значений на 'NaN' и удаления 'op_10cm', 'product' и 'sgas_number':")
    analyze_records(cleaned_data)

    print("\nАнализ вложенных полей в energetic_events после замены:")
    analyze_energetic_events(cleaned_data)


'''
пропуски NaN
'''