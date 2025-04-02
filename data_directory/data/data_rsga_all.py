import re
import json
from pathlib import Path

RE_SDF_NUMBER = re.compile(r"SDF Number\s+(\d+)", re.IGNORECASE)
RE_SOLAR_ACTIVITY_LEVEL = re.compile(
    r"Solar activity (?:has been at|was)[\s:]+(?:(?:at|been at)\s+)?([a-z]+(?:\s+(?:to\s+)?[a-z]+){0,2})(?=\s+levels\b|$)",
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
    r"The geomagnetic field (?:has been at|was)[\s:]+(?:(?:at|been at)\s+)?([a-z]+(?:\s+(?:to\s+)?[a-z]+){0,2})(?=\s+levels\b|$)",
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
    pattern = rf"(?s){section_name}\.\s+(.*?)(?=\nI[A-Z]{{1,2}}\.|$)"
    match = re.search(pattern, text)
    if not match:
        return ""
    section_text = match.group(1)
    section_text = re.sub(r"\s+", " ", section_text).strip()
    return section_text

def parse_rsgi_file(filepath: Path) -> dict | None:
    """
    Парсит файл RSGA, извлекая заголовки и секции IA и IIA.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")

        issued_match = re.search(r":Issued:\s*(.+)", text)
        product_match = re.search(r":Product:\s*(.+)", text)
        if not issued_match or not product_match:
            print(f"[WARN] {filepath.name}: нет заголовка :Issued: или :Product:. Пропускаем.")
            return None

        issued = issued_match.group(1).strip()
        product = product_match.group(1).strip()

        sdf_number = None
        m = RE_SDF_NUMBER.search(text)
        if m:
            sdf_number = int(m.group(1))

        section_ia = extract_section(text, "IA")
        section_iia = extract_section(text, "IIA")

        # Отладочный вывод секций
        print("-----")
        print(f"[DEBUG] {filepath.name} - IA section:\n{section_ia}")
        print(f"[DEBUG] {filepath.name} - IIA section:\n{section_iia}")
        print("-----")

        if not section_ia and not section_iia:
            print(f"[INFO] {filepath.name}: нет IA и IIA секций, пропускаем.")
            return None

        # Извлекаем данные из секции IA
        solar_activity_level = ""
        largest_event_class = ""
        largest_event_time = ""
        largest_event_region = ""
        largest_event_coords = ""
        num_sunspot_regions = ""

        if section_ia:
            m = RE_SOLAR_ACTIVITY_LEVEL.search(section_ia)
            if m:
                solar_activity_level = m.group(1).strip()

            m = RE_LARGEST_EVENT.search(section_ia)
            if m:
                largest_event_class = m.group(1)
                largest_event_time = m.group(2)
                largest_event_region = m.group(3)
                largest_event_coords = m.group(4)

            m = RE_NUM_SUNSPOT_REGIONS.search(section_ia)
            if m:
                num_sunspot_regions = m.group(1)

        # Извлекаем данные из секции IIA
        geomagnetic_activity_level = ""
        solar_wind_speed_peak = ""
        solar_wind_speed_peak_time = ""
        total_imf_max = ""
        total_imf_max_time = ""
        bz_min = ""
        bz_min_time = ""
        electron_flux_peak = ""

        if section_iia:
            m = RE_GEOMAG_LEVEL.search(section_iia)
            if m:
                geomagnetic_activity_level = m.group(1).strip()

            m = RE_SOLAR_WIND_SPEED.search(section_iia)
            if m:
                solar_wind_speed_peak = m.group(1)
                solar_wind_speed_peak_time = m.group(2)

            m = RE_TOTAL_IMF.search(section_iia)
            if m:
                total_imf_max = m.group(1)
                total_imf_max_time = m.group(2)

            m = RE_BZ_MIN.search(section_iia)
            if m:
                bz_min = m.group(1)
                bz_min_time = m.group(2)

            m = RE_ELECTRON_FLUX.search(section_iia)
            if m:
                electron_flux_peak = m.group(1)

        return {
            "filename": filepath.name,
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
    except Exception as e:
        print(f"Ошибка обработки файла {filepath}: {e}")
        return None

def main():
    output_directory = Path("../processed_results")
    output_directory.mkdir(parents=True, exist_ok=True)

    all_data = []

    for year in range(1988, 1996):
        input_directory = Path(f"./ftp_data/{year}_RSGA")
        if input_directory.exists():
            for filepath in input_directory.glob("*.txt"):
                print(f"Обработка: {filepath.name}")
                parsed = parse_rsgi_file(filepath)
                if parsed:
                    parsed["year"] = year
                    all_data.append(parsed)
        else:
            print(f"Папка {input_directory} не существует.")

    for year in range(1996, 2025):
        input_directory = Path(f"../ftp_data/{year}/{year}_RSGA")
        if input_directory.exists():
            for filepath in input_directory.glob("*.txt"):
                print(f"Обработка: {filepath.name}")
                parsed = parse_rsgi_file(filepath)
                if parsed:
                    parsed["year"] = year
                    all_data.append(parsed)
        else:
            print(f"Папка {input_directory} не существует.")

    if not all_data:
        print("Не удалось извлечь ни одной записи.")
        return

    out_json = output_directory / "rsga_combined_all.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"Сохранили {len(all_data)} записей в {out_json}")

if __name__ == "__main__":
    main()
