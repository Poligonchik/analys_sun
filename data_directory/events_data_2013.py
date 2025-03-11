import os
import re
import json
from pathlib import Path

input_directory = Path("./ftp_data/2013/2013_events")
output_directory = Path("./processed_results")
output_directory.mkdir(parents=True, exist_ok=True)

# Функция для парсинга одного файла
def parse_event_file(filepath):
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        # Извлечение заголовков
        date_match = re.search(r":Date:\s*(\d{4} \d{2} \d{2})", "".join(lines))
        if date_match:
            date = date_match.group(1)
        else:
            date = "Unknown"
        events = []
        event_section_started = False

        for line in lines:
            if "Event" in line and "Begin" in line:  # Начало секции событий
                event_section_started = True
                continue

            if event_section_started:
                # Игнорируем пустые строки или разделители
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
for filepath in input_directory.glob("*.txt"):
    print(f"Обработка файла: {filepath}")
    parsed_data = parse_event_file(filepath)
    if parsed_data:
        all_events.append(parsed_data)

# Сохранение объединенных данных в JSON
output_json = output_directory / "combined_events.json"
with open(output_json, "w") as json_file:
    json.dump(all_events, json_file, indent=4)

print(f"Все обработано и сохранено в файл: {output_json}")
