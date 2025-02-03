import json
from pathlib import Path
import re


def parse_dsd_file(filepath: Path) -> list:
    """
    Парсит файл Daily Solar Data (DSD) и возвращает список записей в формате словаря.
    """
    data_entries = []

    with filepath.open('r', encoding='utf-8') as file:
        lines = file.readlines()

    # Идентифицировать строку начала данных (после заголовков)
    data_start = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # Пропустить пустые строки и комментарии
        if re.match(r'\d{4} \d{2} \d{2}', line):
            data_start = True
        if data_start:
            # Разбить строку на части
            parts = line.split()
            if len(parts) < 15:
                # Проверка на достаточное количество полей
                continue
            # Извлечь дату
            year, month, day = parts[0], parts[1], parts[2]
            date = f"{year}-{month}-{day}"

            # Извлечь остальные поля
            try:
                radio_flux = float(parts[3])
            except ValueError:
                radio_flux = None

            try:
                sunspot_number = int(parts[4])
            except ValueError:
                sunspot_number = None

            try:
                hemispheric_area = float(parts[5])
            except ValueError:
                hemispheric_area = None

            try:
                new_regions = int(parts[6])
            except ValueError:
                new_regions = None

            try:
                solar_field = float(parts[7])
            except ValueError:
                solar_field = None

            x_ray_flux = parts[8]  # Строковое значение (например, B3.1)

            try:
                background = int(parts[9])
            except ValueError:
                background = None

            try:
                x_flares = int(parts[10])
            except ValueError:
                x_flares = None

            try:
                m_flares = int(parts[11])
            except ValueError:
                m_flares = None

            try:
                s_flares = int(parts[12])
            except ValueError:
                s_flares = None

            try:
                optical_1 = int(parts[13])
            except ValueError:
                optical_1 = None

            try:
                optical_2 = int(parts[14])
            except ValueError:
                optical_2 = None

            # Опционально: если есть дополнительные поля
            optical_3 = int(parts[15]) if len(parts) > 15 else None

            # Создать словарь записи
            entry = {
                "date": date,
                "radio_flux": radio_flux,
                "sunspot_number": sunspot_number,
                "hemispheric_area": hemispheric_area,
                "new_regions": new_regions,
                "solar_field": solar_field,
                "x_ray_flux": x_ray_flux,
                "background": background,
                "flares": {
                    "X": x_flares,
                    "M": m_flares,
                    "S": s_flares
                },
                "optical_flares": {
                    "1": optical_1,
                    "2": optical_2,
                    "3": optical_3
                }
            }

            data_entries.append(entry)

    return data_entries


def save_to_json(data: list, output_path: Path):
    """
    Сохраняет данные в формате JSON.
    """
    with output_path.open('w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"Данные сохранены в {output_path}")


def main():
    input_file = Path('./ftp_data/2013/2013_DSD.txt')
    output_file = Path('./processed_results/DSD_2013.json')

    if not input_file.exists():
        print(f"Файл {input_file} не найден.")
        return

    parsed_data = parse_dsd_file(input_file)
    if parsed_data:
        save_to_json(parsed_data, output_file)
    else:
        print("Нет данных для сохранения.")


if __name__ == "__main__":
    main()
