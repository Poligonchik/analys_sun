import re
import json
from pathlib import Path

# Папка, где лежат файлы RSGA за 2013 год
input_directory = Path("./ftp_data/2013/2013_RSGA")
# Папка для сохранения результата
output_directory = Path("../processed_results")
output_directory.mkdir(parents=True, exist_ok=True)

# --- Регулярные выражения для поиска ключевых фраз ---
RE_SDF_NUMBER = re.compile(r"SDF Number\s+(\d+)", re.IGNORECASE)
RE_SOLAR_ACTIVITY_LEVEL = re.compile(
    r"Solar activity has been at ([\w\s\-]+) levels",
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
    r"The geomagnetic field has been at ([\w\s\-]+) levels",
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


def extract_section(text: str, section_name: str) -> str:
    """
    Ищем содержимое между 'IA.' ... (до следующего 'I[B-Z].' или конца текста),
    либо 'IIA.' ... (до следующего 'I[B-Z].' или конца текста).

    Используем более простой шаблон, не завязанный на '^' (начало строки).
    """
    # Пример: для IA:
    #   (?s) - включаем DOTALL, чтобы . покрывало в том числе переводы строк
    #   IA\.\s+ - ищем 'IA.' + пробелы
    #   (.*?) - берем все до следующего совпадения
    #   (?=\nI[A-Z]{1,2}\.|$) - до нового раздела вида I[B-Z]. или конца текста
    # Аналогично для "IIA.".
    pattern = rf"(?s){section_name}\.\s+(.*?)(?=\nI[A-Z]{{1,2}}\.|$)"
    match = re.search(pattern, text)
    if not match:
        return ""
    section_text = match.group(1)
    # Нормализуем пробелы
    section_text = re.sub(r"\s+", " ", section_text).strip()
    return section_text


def parse_rsgi_file(filepath: Path) -> dict | None:
    """
    Извлекаем заголовки + 2 секции (IA и IIA),
    затем ищем в них данные по заранее заготовленным регулярным выражениям.
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")

    # Заголовки :Issued: и :Product:
    issued_match = re.search(r":Issued:\s*(.+)", text)
    product_match = re.search(r":Product:\s*(.+)", text)
    if not issued_match or not product_match:
        print(f"[WARN] {filepath.name}: нет заголовка :Issued: или :Product:. Пропускаем.")
        return None

    issued = issued_match.group(1).strip()
    product = product_match.group(1).strip()

    # SDF Number
    sdf_number = None
    m = RE_SDF_NUMBER.search(text)
    if m:
        sdf_number = int(m.group(1))

    # Извлекаем текст секций
    section_ia = extract_section(text, "IA")
    section_iia = extract_section(text, "IIA")

    # Для отладки посмотрим, что реально вырезали
    print("-----")
    print(f"[DEBUG] {filepath.name} - IA section:\n{section_ia}")
    print(f"[DEBUG] {filepath.name} - IIA section:\n{section_iia}")
    print("-----")

    # Если обе секции пусты — пропускаем файл
    if not section_ia and not section_iia:
        print(f"[INFO] {filepath.name}: нет IA и IIA секций, пропускаем.")
        return None

    # Ищем нужные данные
    solar_activity_level = ""
    largest_event_class = ""
    largest_event_time = ""
    largest_event_region = ""
    largest_event_coords = ""
    num_sunspot_regions = ""

    if section_ia:
        # Solar Activity Level
        m = RE_SOLAR_ACTIVITY_LEVEL.search(section_ia)
        if m:
            solar_activity_level = m.group(1).strip()

        # Largest Event
        m = RE_LARGEST_EVENT.search(section_ia)
        if m:
            largest_event_class = m.group(1)
            largest_event_time = m.group(2)
            largest_event_region = m.group(3)
            largest_event_coords = m.group(4)

        # Number of Sunspot Regions
        m = RE_NUM_SUNSPOT_REGIONS.search(section_ia)
        if m:
            num_sunspot_regions = m.group(1)

    geomagnetic_activity_level = ""
    solar_wind_speed_peak = ""
    solar_wind_speed_peak_time = ""
    total_imf_max = ""
    total_imf_max_time = ""
    bz_min = ""
    bz_min_time = ""
    electron_flux_peak = ""

    if section_iia:
        # Geomagnetic Activity Level
        m = RE_GEOMAG_LEVEL.search(section_iia)
        if m:
            geomagnetic_activity_level = m.group(1).strip()

        # Solar Wind Speed
        m = RE_SOLAR_WIND_SPEED.search(section_iia)
        if m:
            solar_wind_speed_peak = m.group(1)
            solar_wind_speed_peak_time = m.group(2)

        # Total IMF
        m = RE_TOTAL_IMF.search(section_iia)
        if m:
            total_imf_max = m.group(1)
            total_imf_max_time = m.group(2)

        # Bz Min
        m = RE_BZ_MIN.search(section_iia)
        if m:
            bz_min = m.group(1)
            bz_min_time = m.group(2)

        # Electron Flux
        m = RE_ELECTRON_FLUX.search(section_iia)
        if m:
            electron_flux_peak = m.group(1)

    # Формируем результат
    return {
        "issued": issued,
        "product": product,
        "sdf_number": sdf_number,

        "solar_activity_level": solar_activity_level,
        "largest_event_class": largest_event_class,
        "largest_event_time": largest_event_time,
        "largest_event_region": largest_event_region,
        "largest_event_coords": largest_event_coords,
        "num_sunspot_regions": num_sunspot_regions,

        "geomagnetic_activity_level": geomagnetic_activity_level,
        "solar_wind_speed_peak": solar_wind_speed_peak,
        "solar_wind_speed_peak_time": solar_wind_speed_peak_time,
        "total_imf_max": total_imf_max,
        "total_imf_max_time": total_imf_max_time,
        "bz_min": bz_min,
        "bz_min_time": bz_min_time,
        "electron_flux_peak": electron_flux_peak
    }


def main():
    input_directory = Path("./ftp_data/2013/2013_RSGA")
    output_directory = Path("../processed_results")
    output_directory.mkdir(parents=True, exist_ok=True)

    all_data = []
    for filepath in input_directory.glob("*.txt"):
        print(f"Обработка: {filepath.name}")
        parsed = parse_rsgi_file(filepath)
        if parsed:
            all_data.append(parsed)

    if not all_data:
        print("Не удалось извлечь ни одной записи.")
        return

    # Сохраняем в JSON
    out_json = output_directory / "rsga_2013_parsed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"Сохранили {len(all_data)} записей в {out_json}")


if __name__ == "__main__":
    main()
