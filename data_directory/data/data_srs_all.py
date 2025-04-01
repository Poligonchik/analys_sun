import pandas as pd
import re
from pathlib import Path
import json

RE_SDF_NUMBER = re.compile(r"SRS Number\s+(\d+)", re.IGNORECASE)  # иногда используется SRS Number
RE_SOLAR_ACTIVITY_LEVEL = re.compile(
    r"Solar activity (?:has been at|was)[\s:]+([\w\s\-]+)(?: levels)?",
    re.IGNORECASE
)
RE_LARGEST_EVENT = re.compile(
    r"The largest solar event of the period was a\s+([CMX]\d?(?:\.\d+)?)\s+event observed at\s+(\d{2}/\d{4}Z)\s+from Region\s+(\d+)\s*\(([^)]+)\)",
    re.IGNORECASE
)
RE_NUM_SUNSPOT_REGIONS = re.compile(
    r"There (?:is|are) currently (\d+) numbered sunspot regions?",
    re.IGNORECASE
)
RE_GEOMAG_LEVEL = re.compile(
    r"The geomagnetic field (?:has been at|was)[\s:]+([\w\s\-]+) levels?",
    re.IGNORECASE
)
RE_SOLAR_WIND_SPEED = re.compile(
    r"reached a peak speed of\s+(\d+)\s+km/s at\s+(\d{2}/\d{4}Z)",
    re.IGNORECASE
)
RE_TOTAL_IMF = re.compile(
    r"Total IMF reached\s+([\d\.]+)\s+nT at\s+(\d{2}/\d{4}Z)",
    re.IGNORECASE
)
RE_BZ_MIN = re.compile(
    r"The maximum southward component of Bz reached\s+(-?[\d\.]+)\s+nT at\s+(\d{2}/\d{4}Z)",
    re.IGNORECASE
)
RE_ELECTRON_FLUX = re.compile(
    r"Electrons greater than 2 MeV.*?reached a peak level of\s+(\d+)\s+pfu",
    re.IGNORECASE
)


def extract_section(text: str, section_title: str) -> list:
    """
    Извлекает строки секции из текста.
    Ищет содержимое после строки с section_title до появления новой секции (начинающейся с
    заглавных букв) или пустой строки.
    Возвращает список строк секции.
    """
    section_lines = []
    section_started = False
    for line in text.splitlines():
        if section_title in line:
            section_started = True
            continue
        if section_started:
            # Если строка пустая или начинается с заглавных букв, можно считать, что секция закончилась
            if line.strip() == "" or re.match(r"^[A-Z]{2,}", line.strip()):
                break
            section_lines.append(line.strip())
    if not section_lines:
        print(f"Секция '{section_title}' не найдена или пуста в файле: {text[:100]}...")
    return section_lines


def parse_file(filepath: Path) -> dict | None:
    """
    Парсит один SRS-файл.
    Извлекает заголовки (:Issued:, :Product:) и секции:
      - I. Regions with Sunspots
      - IA. H-alpha Plages without Spots
      - II. Regions Due to Return
    """
    try:
        print(f"Обработка файла: {filepath.name}")
        with open(filepath, 'r', encoding="utf-8", errors="replace") as f:
            data = f.read()

        # Извлечение заголовков
        issued_match = re.search(r":Issued:\s*(.+)", data)
        product_match = re.search(r":Product:\s*(.+)", data)
        if not issued_match or not product_match:
            print(f"Ошибка парсинга заголовков в файле: {filepath.name}")
            return None

        issued = issued_match.group(1).strip()
        product = product_match.group(1).strip()

        regions_with_sunspots = extract_section(data, "I.  Regions with Sunspots")
        h_alpha_plages = extract_section(data, "IA. H-alpha Plages without Spots")
        regions_due_to_return = extract_section(data, "II. Regions Due to Return")

        return {
            "issued": issued,
            "product": product,
            "regions_with_sunspots": regions_with_sunspots,
            "h_alpha_plages": h_alpha_plages,
            "regions_due_to_return": regions_due_to_return,
        }
    except Exception as e:
        print(f"Ошибка обработки файла {filepath.name}: {e}")
        return None


def save_to_dataframe(parsed_data: list, section_key: str, columns: list) -> pd.DataFrame:
    """
    Преобразует данные секции из всех файлов в DataFrame.
    Каждая строка DataFrame содержит заголовки (issued, product) и поля из строки секции.
    Если число полей в строке меньше ожидаемого, дополняет недостающие поля пустыми строками.
    Если число полей больше ожидаемого — строка пропускается.
    """
    rows = []
    expected_columns = len(columns)
    for entry in parsed_data:
        issued = entry["issued"]
        product = entry["product"]
        for line in entry[section_key]:
            # Пропускаем строки-заголовки (например, начинающиеся с "Nmbr")
            if line.startswith("Nmbr") or re.match(r"^[A-Za-z ]+$", line):
                continue
            split_line = line.split()
            if len(split_line) < expected_columns:
                # Если получено меньше столбцов (например, 7 вместо 8), дополняем пустыми значениями
                print(f"Строка из файла {entry.get('filename', '')} содержит меньше колонок ({len(split_line)}), ожидается {expected_columns}. Дополняем пустыми значениями.")
                split_line += [""] * (expected_columns - len(split_line))
            elif len(split_line) > expected_columns:
                print(f"Проблемная строка в секции '{section_key}' файла {entry.get('filename', '')}: {line}")
                print(f"Ожидаемое количество колонок: {expected_columns}, найдено: {len(split_line)}")
                continue
            rows.append([issued, product] + split_line)
    if not rows:
        print(f"Данные для секции '{section_key}' пусты.")
    return pd.DataFrame(rows, columns=["issued", "product"] + columns)




def main():
    import pandas as pd
    output_directory = Path("../processed_results")
    output_directory.mkdir(parents=True, exist_ok=True)

    parsed_data = []
    for year in range(1996, 2025):
        input_directory = Path(f"../ftp_data/{year}/{year}_SRS")
        if input_directory.exists():
            for filepath in input_directory.glob("*.txt"):
                parsed_entry = parse_file(filepath)
                if parsed_entry:
                    parsed_entry["filename"] = filepath.name
                    parsed_entry["year"] = year
                    parsed_data.append(parsed_entry)
        else:
            print(f"Папка {input_directory} не существует.")

    if not parsed_data:
        print("Парсинг не дал результатов. Проверьте входные файлы.")
        return

    # Создание DataFrame для каждой секции
    sunspots_columns = ["Nmbr", "Location", "Lo", "Area", "Z", "LL", "NN", "Mag_Type"]
    sunspots_df = save_to_dataframe(parsed_data, "regions_with_sunspots", sunspots_columns)

    h_alpha_columns = ["Nmbr", "Location", "Lo"]
    h_alpha_df = save_to_dataframe(parsed_data, "h_alpha_plages", h_alpha_columns)

    return_columns = ["Nmbr", "Lat", "Lo"]
    return_df = save_to_dataframe(parsed_data, "regions_due_to_return", return_columns)

    out_json = output_directory / "combined_srs_all.json"
    data_to_save = {
        "regions_with_sunspots": sunspots_df.to_dict(orient="records"),
        "h_alpha_plages": h_alpha_df.to_dict(orient="records"),
        "regions_due_to_return": return_df.to_dict(orient="records"),
    }
    with open(out_json, "w", encoding="utf-8") as json_file:
        json.dump(data_to_save, json_file, indent=4, ensure_ascii=False)

    sunspots_df.to_parquet(output_directory / "sunspots.parquet", index=False)
    h_alpha_df.to_parquet(output_directory / "h_alpha.parquet", index=False)
    return_df.to_parquet(output_directory / "return.parquet", index=False)

    print(f"Данные успешно обработаны и сохранены в директорию: {output_directory}")


if __name__ == "__main__":
    main()
