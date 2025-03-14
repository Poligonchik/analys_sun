import json
from pathlib import Path
import re

# Словарь для преобразования сокращённого названия месяца в номер
month_map = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}


def parse_dsd_file(filepath: Path) -> list:
    """
    Парсит файл Daily Solar Data (DSD) и возвращает список записей в формате словаря.
    Поддерживаются два формата:
      1. Формат с датой вида "01 Jan 96" и 12 полей (например, для 1996 года).
      2. Формат с датой вида "1998 01 01" и минимум 16 полей (например, для 1998 года).
    Если в строке недостаточно данных – она пропускается.
    """
    data_entries = []
    file_format = None  # будет 'dmy' или 'ymd'

    with filepath.open('r', encoding='utf-8') as file:
        lines = file.readlines()

    data_start = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # пропускаем пустые строки и комментарии

        parts = line.split()

        # Определяем формат, если ещё не выбран
        if not data_start:
            # Если первая часть – двухзначное число и вторая – название месяца
            if re.match(r'^\d{2}$', parts[0]) and parts[1] in month_map:
                file_format = 'dmy'
                data_start = True
            # Если первая часть – четырёхзначное число (год)
            elif re.match(r'^\d{4}$', parts[0]):
                file_format = 'ymd'
                data_start = True
            else:
                continue

        if file_format == 'dmy':
            # Для формата dmy ожидаем ровно 12 полей (или больше, но берем первые 12)
            if len(parts) < 12:
                continue

            # Парсим дату в формате "01 Jan 96"
            day, mon_str, yr = parts[0], parts[1], parts[2]
            month = month_map.get(mon_str, mon_str)
            full_year = "19" + yr  # Предполагаем данные за 20 век
            date = f"{full_year}-{month}-{day.zfill(2)}"

            # Остальные поля
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

            x_ray_flux = parts[8]  # строковое значение

            # Флейры – предполагаем, что поля 9, 10, 11 соответствуют классам C, M, X
            try:
                c_flares = int(parts[9]) if parts[9] != "*" else None
            except ValueError:
                c_flares = None
            try:
                m_flares = int(parts[10]) if parts[10] != "*" else None
            except ValueError:
                m_flares = None
            try:
                x_flares = int(parts[11]) if parts[11] != "*" else None
            except ValueError:
                x_flares = None

            entry = {
                "date": date,
                "radio_flux": radio_flux,
                "sunspot_number": sunspot_number,
                "hemispheric_area": hemispheric_area,
                "new_regions": new_regions,
                "solar_field": solar_field,
                "x_ray_flux": x_ray_flux,
                "background": None,          # отсутствует в этом формате
                "flares": {
                    "C": c_flares,
                    "M": m_flares,
                    "X": x_flares
                },
                "optical_flares": None       # отсутствует в этом формате
            }
            data_entries.append(entry)

        elif file_format == 'ymd':
            # Для формата ymd ожидаем минимум 16 полей
            if len(parts) < 16:
                continue

            # Парсим дату в формате "1998 01 01"
            year, month, day = parts[0], parts[1], parts[2]
            date = f"{year}-{month}-{day.zfill(2)}"

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
                sf = float(parts[7])
                solar_field = sf if sf != -999 else None
            except ValueError:
                solar_field = None

            x_ray_flux = parts[8]

            try:
                bg = int(parts[9])
                background = bg if bg != -999 else None
            except ValueError:
                background = None

            try:
                c_flares = int(parts[10])
            except ValueError:
                c_flares = None
            try:
                m_flares = int(parts[11])
            except ValueError:
                m_flares = None
            try:
                x_flares = int(parts[12])
            except ValueError:
                x_flares = None
            try:
                s_flares = int(parts[13])
            except ValueError:
                s_flares = None

            # Оптические вспышки – следующие поля (может быть 2 или 3 поля)
            optical = {}
            if len(parts) >= 15:
                try:
                    optical["1"] = int(parts[14])
                except ValueError:
                    optical["1"] = None
            else:
                optical["1"] = None
            if len(parts) >= 16:
                try:
                    optical["2"] = int(parts[15])
                except ValueError:
                    optical["2"] = None
            else:
                optical["2"] = None
            if len(parts) >= 17:
                try:
                    optical["3"] = int(parts[16])
                except ValueError:
                    optical["3"] = None
            else:
                optical["3"] = None

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
                    "C": c_flares,
                    "M": m_flares,
                    "X": x_flares,
                    "S": s_flares
                },
                "optical_flares": optical
            }
            data_entries.append(entry)

    return data_entries


def save_to_json(data: list, output_path: Path):
    with output_path.open('w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"Данные сохранены в {output_path}")


def main():
    all_data = []
    # Обрабатываем годы от 1996 до 2024
    for year in range(1996, 2025):
        input_file = Path(f'../ftp_data/{year}/{year}_DSD.txt')
        if not input_file.exists():
            print(f"Файл {input_file} не найден для года {year}.")
            continue

        yearly_data = parse_dsd_file(input_file)
        if yearly_data:
            print(f"Год {year}: обработано {len(yearly_data)} записей")
            all_data.extend(yearly_data)
        else:
            print(f"Нет данных для года {year}.")

    if all_data:
        output_file = Path('../processed_results/DSD_all.json')
        save_to_json(all_data, output_file)
    else:
        print("Нет данных для сохранения.")


if __name__ == "__main__":
    main()
