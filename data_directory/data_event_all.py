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


# Функция для парсинга одного файла событий
def parse_event_file(filepath):
    try:
        filename = filepath.name
        with open(filepath, 'r') as file:
            lines = file.readlines()
        content = "".join(lines)

        # Попытка извлечения даты из имени файла
        date = extract_date_from_filename(filename)

        # Если не удалось извлечь дату из имени, то проверяем содержимое файла
        if not date:
            if "Space Environment Center" in content:
                # Для формата 1996–1998
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
                # Стандартный формат: ищем строку с тегом :Date:
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

                parts = re.split(r"\s{2,}", line.strip())
                if len(parts) >= 7:
                    events.append({
                        "event": parts[0],
                        "begin": parts[1],
                        "max": parts[2],
                        "end": parts[3],
                        "obs": parts[4],
                        "q": parts[5],
                        "type": parts[6],
                        "loc_freq": parts[7] if len(parts) > 7 else None,
                        "particulars": parts[8] if len(parts) > 8 else None,
                        "region": parts[9] if len(parts) > 9 else None
                    })

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
    input_directory = Path(f"./ftp_data/{year}/{year}_events")
    if input_directory.exists():
        for filepath in input_directory.glob("*.txt"):
            print(f"Обработка файла: {filepath}")
            parsed_data = parse_event_file(filepath)
            if parsed_data:
                parsed_data["year"] = year  # добавляем год для отладки
                all_events.append(parsed_data)
    else:
        print(f"Папка {input_directory} не существует.")

# Сохранение объединенных данных в JSON-файл
output_directory = Path("./processed_results")
output_directory.mkdir(parents=True, exist_ok=True)
output_json = output_directory / "combined_events_all.json"
with open(output_json, "w") as json_file:
    json.dump(all_events, json_file, indent=4)

print(f"Все обработано и сохранено в файл: {output_json}")
