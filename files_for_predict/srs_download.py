#!/usr/bin/env python3
import requests
import os
import json
import re
import pandas as pd


def download_srs_json(url: str, filename: str) -> bool:
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Ошибка при загрузке: {e}")
        return False

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Файл успешно сохранён как: {os.path.abspath(filename)}")
    return True


# Функция для вычисления поля Lo из строки location с обработкой None
def extract_lo(location) -> str:
    if not location:
        return "ND"
    location = str(location)
    match = re.search(r"([NS])(\d+)([EW])(\d+)", location)
    if match:
        # Извлекаем группы: lat_dir, lat, lon_dir, lon – нам нужна только долгота
        _, _, lon_dir, lon = match.groups()
        lon = int(lon)
        if lon_dir == 'E':
            lo = (360 - lon) % 360
        else:
            lo = lon
        return f"{lo:03d}"
    return "ND"


# Маппинг для поля Mag (будет переименовано в Mag_Type)
mag_mapping = {
    "G": "Gamma",
    "GD": "Gamma-Delta",
    "BD": "Beta-Delta",
    "BGD": "Beta-Gamma-Delta",
    "BG": "Beta-Gamma",
    "A": "Alpha",
    "B": "Beta"
}


# Функция для форматирования значений в двухзначное целое число
def format_two_digits(x):
    try:
        return f"{int(float(x)):02d}"
    except (ValueError, TypeError):
        return "ND"


def process_srs_file(input_filename: str, output_filename: str):
    # Загрузка JSON-данных из файла
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Преобразуем данные в DataFrame
    df = pd.DataFrame(data)

    # Приводим дату из формата "2025-03-30" к формату "2025 03 30"
    df["observed_date"] = df["observed_date"].astype(str).str.replace("-", " ")

    # Создаем новые поля согласно схеме:
    # Nmbr     = region
    # Location = location
    # Lo       = вычисляем через extract_lo(location)
    # Area     = area
    # Z        = spot_class
    # LL       = extent, форматируем как двухзначное целое число (например, 5 → "05")
    # NN       = number_spots, форматируем как двухзначное целое число
    # Mag_Type = mag_class, преобразуем по mag_mapping (если нет соответствия, будет "ND")
    # Type     = "f" (фиксированное значение)
    df["Nmbr"] = df["region"]
    df["Location"] = df["location"].fillna("ND")
    df["Lo"] = df["Location"].apply(extract_lo)
    df["Area"] = df["area"].fillna("ND")
    df["Z"] = df["spot_class"].fillna("ND")
    df["LL"] = df["extent"].fillna("ND")
    df["NN"] = df["number_spots"].fillna("ND")

    # Форматируем LL и NN как двухзначные числа
    df["LL"] = df["LL"].apply(format_two_digits)
    df["NN"] = df["NN"].apply(format_two_digits)

    df["Mag_Type"] = df["mag_class"].fillna("ND").map(mag_mapping).fillna("ND")
    df["Type"] = "f"

    # Убираем события, где Mag_Type равно "ND"
    df = df[df["Mag_Type"] != "ND"]

    # Оставляем нужные столбцы, включая observed_date
    final_df = df[["observed_date", "Nmbr", "Location", "Lo", "Area", "Z", "LL", "NN", "Mag_Type", "Type"]]

    # Удаляем поля "Z" и "Type"
    final_df = final_df.drop(columns=["Z", "Type"])

    # Переставляем столбцы так, чтобы observed_date оказался в конце
    final_df = final_df[["Nmbr", "Location", "Lo", "Area", "LL", "NN", "Mag_Type", "observed_date"]]

    # Переименовываем поле observed_date в date
    final_df = final_df.rename(columns={"observed_date": "date"})

    # Преобразуем итоговый DataFrame в список словарей и сохраняем в JSON-файл
    final_json = final_df.to_dict(orient="records")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False)

    print(f"Файл {output_filename} успешно создан.")

    # Подсчет количества значений "ND" по каждому столбцу
    print("\nКоличество 'ND' по столбцам:")
    for col in final_df.columns:
        nd_count = (final_df[col] == "ND").sum()
        print(f"{col}: {nd_count}")



def main():
    download_url = "https://services.swpc.noaa.gov/json/solar_regions.json"
    input_filename = "./tables/srs_download.json"
    output_filename = "./tables/srs_download.json"

    # Скачиваем файл, если загрузка успешна, переходим к обработке
    if download_srs_json(download_url, input_filename):
        process_srs_file(input_filename, output_filename)


if __name__ == "__main__":
    main()
