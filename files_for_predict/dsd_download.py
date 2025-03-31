#!/usr/bin/env python3
import requests
import os
import json
from pathlib import Path

def download_dsd_file() -> Path:
    """
    Скачивает файл с ежедневными солнечными индексами и сохраняет его как "dsd.txt".
    """
    url = "https://services.swpc.noaa.gov/text/daily-solar-indices.txt"
    filename = "tables/dsd.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Ошибка при загрузке: {e}")
        return None

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Файл успешно сохранён как: {os.path.abspath(filename)}")
    return Path(filename)

def parse_dsd_file(filepath: Path) -> list:
    """
    Парсит файл DSD.txt и возвращает список записей.
    Предполагается, что каждая строка данных содержит минимум 15 полей:
      Year Month Day radio_flux sunspot_number hemispheric_area new_regions solar_field
      x_ray_flux c_flares m_flares x_flares s_flares optical_flare1 optical_flare2 [optical_flare3]
    Если значение равно "*" – заменяется на 0.0 для числовых полей или "A0.0" для x_ray_flux.
    Поле solar_field и optical_flare* не включаются в итоговый результат.
    """
    data_entries = []
    with filepath.open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Пропускаем заголовки и комментарии
            if not line or line.startswith('#') or line.startswith(':'):
                continue

            parts = line.split()
            if len(parts) < 15:
                continue

            # Формирование даты: Year, Month, Day
            year, month, day = parts[0], parts[1], parts[2]
            date = f"{year} {month} {day.zfill(2)}"

            # radio_flux
            try:
                radio_flux = float(parts[3])
            except ValueError:
                radio_flux = None

            # sunspot_number
            try:
                sunspot_number = int(parts[4])
            except ValueError:
                sunspot_number = None

            # hemispheric_area
            try:
                hemispheric_area = float(parts[5])
            except ValueError:
                hemispheric_area = None

            # new_regions
            try:
                new_regions = int(parts[6])
            except ValueError:
                new_regions = None

            # Пропускаем поле solar_field (parts[7])

            # x_ray_flux: если значение равно "*" – устанавливаем "A0.0", иначе оставляем как есть
            x_ray_flux = "A0.0" if parts[8] == "*" else parts[8]

            # Обработка вспышек (поля 9-12); если значение равно "*" – считаем его 0.0
            try:
                c_flares = 0.0 if parts[9] == "*" else float(parts[9])
            except ValueError:
                c_flares = 0.0
            try:
                m_flares = 0.0 if parts[10] == "*" else float(parts[10])
            except ValueError:
                m_flares = 0.0
            try:
                x_flares = 0.0 if parts[11] == "*" else float(parts[11])
            except ValueError:
                x_flares = 0.0
            try:
                s_flares = 0.0 if parts[12] == "*" else float(parts[12])
            except ValueError:
                s_flares = 0.0

            entry = {
                "date": date,
                "radio_flux": radio_flux,
                "sunspot_number": sunspot_number,
                "hemispheric_area": hemispheric_area,
                "new_regions": new_regions,
                "x_ray_flux": x_ray_flux,
                "flares": {
                    "C": c_flares,
                    "M": m_flares,
                    "X": x_flares,
                    "S": s_flares
                }
            }
            data_entries.append(entry)
    return data_entries

def flatten_flares(records: list) -> list:
    """
    Переносит вложенные поля словаря "flares" на уровень верхнего уровня с префиксом "flares."
    и удаляет исходное поле "flares".
    """
    for record in records:
        flares = record.get("flares")
        if isinstance(flares, dict):
            for key, value in flares.items():
                record[f"flares.{key}"] = value
            del record["flares"]
    return records

def save_to_json(data: list, output_path: Path):
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Данные сохранены в {output_path}")

def main():
    # Скачиваем файл
    dsd_filepath = download_dsd_file()
    if dsd_filepath is None:
        return
    # Парсим файл
    records = parse_dsd_file(dsd_filepath)
    print(f"Обработано записей: {len(records)}")
    # Разворачиваем вложенный словарь "flares"
    records = flatten_flares(records)
    # Сохраняем итоговый JSON
    output_file = Path("tables/dsd_download.json")
    save_to_json(records, output_file)

if __name__ == "__main__":
    main()
