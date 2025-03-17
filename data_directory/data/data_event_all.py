import os
import re
import json
from pathlib import Path


# Функция для преобразования текстового обозначения месяца в число
def convert_month(month_abbr):
    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
    }
    return month_map.get(month_abbr, "00")


# Функция для извлечения даты из имени файла, если оно соответствует шаблону YYYYMMDDevents.txt
def extract_date_from_filename(filename):
    match = re.match(r"(\d{4})(\d{2})(\d{2})events\.txt", filename)
    if match:
        year, month, day = match.groups()
        return f"{year} {month} {day}"
    return None


# Функция для приведения значения времени к формату HHMM:
# если первый символ является буквой, удаляем его.
def fix_time(time_str):
    if time_str and not time_str[0].isdigit():
        return time_str[1:]
    return time_str


# Функция для парсинга одного файла событий
def parse_event_file(filepath):
    try:
        filename = filepath.name
        with open(filepath, 'r') as file:
            lines = file.readlines()
        content = "".join(lines)

        # Попытка извлечения даты из имени файла
        date = extract_date_from_filename(filename)

        # Если не удалось извлечь дату из имени, проверяем содержимое файла
        if not date:
            if "Space Environment Center" in content:
                edited_match = re.search(r"EDITED EVENTS for (\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})", content)
                if edited_match:
                    year = edited_match.group(1)
                    month_abbr = edited_match.group(2)
                    day = edited_match.group(3).zfill(2)
                    month = convert_month(month_abbr)
                    date = f"{year} {month} {day}"
                else:
                    date = "Unknown"
            else:
                date_match = re.search(r":Date:\s*(\d{4} \d{2} \d{2})", content)
                if date_match:
                    date = date_match.group(1)
                else:
                    date = "Unknown"

        events = []
        event_section_started = False

        for line in lines:
            if "Event" in line and "Begin" in line:
                event_section_started = True
                continue

            if event_section_started:
                if line.strip() == "" or line.startswith("#") or "----" in line:
                    continue

                # Разбиваем строку по 2 и более пробелам
                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 7:
                    # Если в поле "end" содержатся два значения (например, "B0546 ////"), разделяем их:
                    if " " in parts[3]:
                        end_obs = parts[3].split()
                        if len(end_obs) >= 2:
                            end_field = fix_time(end_obs[0])
                            obs_field = end_obs[1]  # для obs не обрабатываем время
                        else:
                            end_field = fix_time(parts[3])
                            obs_field = parts[4] if len(parts) > 4 else None
                        base_index = 4
                    else:
                        end_field = fix_time(parts[3])
                        obs_field = parts[4]
                        base_index = 5

                    # Обрабатываем поля Begin и Max как время
                    begin_field = fix_time(parts[1])
                    max_field = fix_time(parts[2])

                    q_field = parts[base_index] if len(parts) > base_index else None
                    type_field = parts[base_index + 1] if len(parts) > base_index + 1 else None
                    loc_freq_field = parts[base_index + 2] if len(parts) > base_index + 2 else None
                    particulars_field = parts[base_index + 3] if len(parts) > base_index + 3 else None

                    # Ищем значение для region после particulars
                    region_field = None
                    if len(parts) > base_index + 4:
                        candidate = parts[base_index + 4]
                        # Если кандидат выглядит как экспоненциальное число, то, возможно, за ним следует искомое значение
                        if re.match(r'^[+-]?\d+(\.\d+)?[eE][+-]?\d+$', candidate):
                            if len(parts) > base_index + 5:
                                candidate2 = parts[base_index + 5]
                                if re.match(r'^\d{3,5}$', candidate2):
                                    region_field = candidate2
                        else:
                            if re.match(r'^\d{3,5}$', candidate):
                                region_field = candidate

                    event_dict = {
                        "event": parts[0],  # символ "+" остаётся, если присутствует
                        "begin": begin_field,
                        "max": max_field,
                        "end": end_field,
                        "obs": obs_field,
                        "q": q_field,
                        "type": type_field,
                        "loc_freq": loc_freq_field,
                        "particulars": particulars_field
                    }
                    if region_field is not None:
                        event_dict["region"] = region_field

                    events.append(event_dict)

        return {
            "date": date,
            "events": events
        }
    except Exception as e:
        print(f"Ошибка обработки файла {filepath}: {e}")
        return None


all_events = []

# Перебор годов от 1996 до 2024
for year in range(1996, 2025):
    input_directory = Path(f"../ftp_data/{year}/{year}_events")
    if input_directory.exists():
        for filepath in input_directory.glob("*.txt"):
            print(f"Обработка файла: {filepath}")
            parsed_data = parse_event_file(filepath)
            if parsed_data:
                parsed_data["year"] = year  # добавляем год для отладки
                all_events.append(parsed_data)
    else:
        print(f"Папка {input_directory} не существует.")

# Сохранение объединённых данных в JSON-файл
output_directory = Path("../processed_results")
output_directory.mkdir(parents=True, exist_ok=True)
output_json = output_directory / "combined_events_all.json"
with open(output_json, "w") as json_file:
    json.dump(all_events, json_file, indent=4)

print(f"Все обработано и сохранено в файл: {output_json}")
