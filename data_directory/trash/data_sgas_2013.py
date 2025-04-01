import re
import json
from pathlib import Path

# Папка, где лежат файлы SGAS за 2013 год
input_directory = Path("./ftp_data/2013/2013_SGAS")
# Папка для сохранения результата
output_directory = Path("../processed_results")
output_directory.mkdir(parents=True, exist_ok=True)

# --- Регулярные выражения для поиска ключевых данных ---
# Заголовки
RE_PRODUCT = re.compile(r":Product:\s*(.+)", re.IGNORECASE)
RE_ISSUED = re.compile(r":Issued:\s*(.+)", re.IGNORECASE)
RE_SGAS_NUMBER = re.compile(r"SGAS Number\s+(\d+)", re.IGNORECASE)

# Раздел A. Energetic Events
RE_ENERGETIC_EVENTS_HEADER = re.compile(r"A\.\s+Energetic Events", re.IGNORECASE)
RE_ENERGETIC_EVENT_LINE = re.compile(
    r"(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d+)?\s*([A-Z0-9\.\-]*)\s*([A-Z0-9\-]*)\s*([A-Z0-9\-]*)\s*([A-Z0-9\-]*)",
    re.IGNORECASE
)

# Раздел B. Proton Events
RE_PROTON_EVENTS = re.compile(r"B\.\s+Proton Events:\s*(.+)", re.IGNORECASE)

# Раздел C. Geomagnetic Activity Summary
RE_GEOMAG_SUMMARY = re.compile(r"C\.\s+Geomagnetic Activity Summary:\s*(.+)", re.IGNORECASE)

# Раздел E. Daily Indices
RE_DAILY_INDICES_HEADER = re.compile(r"E\.\s+Daily Indices:", re.IGNORECASE)
RE_DAILY_10CM_SSNN_AP = re.compile(
    r"10\s+cm\s+(\d+)\s+SSN\s+(\d+)\s+Afr/Ap\s+(\d{3}/\d{3})\s+X-ray Background\s+([BFX]\d+\.\d)",
    re.IGNORECASE
)
RE_DAILY_PROTON_FLUENCE = re.compile(
    r"Daily Proton Fluence \(flux accumulation over 24 hrs\)\s+GT\s+1 MeV\s+([\d\.e\+\-]+)\s+GT\s+10 MeV\s+([\d\.e\+\-]+)",
    re.IGNORECASE
)
RE_DAILY_ELECTRON_FLUENCE = re.compile(
    r"Daily Electron Fluence\s+GT\s+2 MeV\s+([\d\.e\+\-]+)",
    re.IGNORECASE
)
RE_K_INDICES = re.compile(
    r"3 Hour K-indices:\s+Boulder\s+([\d\s]+)\s+Planetary\s+([\d\s]+)",
    re.IGNORECASE
)


def extract_section(text: str, section_name: str) -> str:
    """
    Извлекаем содержимое указанного раздела из полного текста файла.
    """
    pattern = rf"(?s){section_name}\.\s+(.*?)(?=\n[A-Z]\.|$)"
    match = re.search(pattern, text)
    if match:
        section_text = match.group(1).strip()
        # Нормализуем пробелы
        section_text = re.sub(r"\s+", " ", section_text)
        return section_text
    else:
        return ""


def parse_energetic_events(section_text: str) -> list:
    """
    Парсим раздел A. Energetic Events и извлекаем события.
    """
    events = []
    lines = section_text.split(' ')
    # Предполагается, что события разделены пробелами, что не идеально.
    # Лучше парсить по строкам, но так как мы нормализовали пробелы, будем искать с помощью регулярки.
    matches = RE_ENERGETIC_EVENT_LINE.findall(section_text)
    for match in matches:
        begin, max_time, end, rgn, loc, xray, op_245MHz, sweep = match
        event = {
            "begin": begin,
            "max": max_time,
            "end": end,
            "region": rgn if rgn else "",
            "location": loc if loc else "",
            "xray_class": xray if xray else "",
            "op_245MHz": op_245MHz if op_245MHz else "",
            "op_10cm": "",  # В примерах отсутствует
            "sweep": sweep if sweep else ""
        }
        events.append(event)
    return events


def parse_proton_events(section_text: str) -> str:
    """
    Парсим раздел B. Proton Events.
    """
    proton_info = section_text.strip()
    return proton_info


def parse_geomag_summary(section_text: str) -> str:
    """
    Парсим раздел C. Geomagnetic Activity Summary.
    """
    summary = section_text.strip()
    return summary


def parse_daily_indices(section_text: str) -> dict:
    """
    Парсим раздел E. Daily Indices.
    """
    indices = {}

    # 10 cm SSN Afr/Ap X-ray Background
    match = RE_DAILY_10CM_SSNN_AP.search(section_text)
    if match:
        indices["10cm"] = int(match.group(1))
        indices["SSN"] = int(match.group(2))
        indices["Afr_Ap"] = match.group(3)
        indices["Xray_Background"] = match.group(4)

    # Daily Proton Fluence
    match = RE_DAILY_PROTON_FLUENCE.search(section_text)
    if match:
        indices["Daily_Proton_Fluence"] = {
            "GT_1_MeV": float(match.group(1)),
            "GT_10_MeV": float(match.group(2))
        }

    # Daily Electron Fluence
    match = RE_DAILY_ELECTRON_FLUENCE.search(section_text)
    if match:
        indices["Daily_Electron_Fluence"] = {
            "GT_2_MeV": float(match.group(1))
        }

    # 3 Hour K-indices
    match = RE_K_INDICES.search(section_text)
    if match:
        boulder_values = [int(x) for x in match.group(1).split()]
        planetary_values = [int(x) for x in match.group(2).split()]
        indices["K_Indices"] = {
            "Boulder": boulder_values,
            "Planetary": planetary_values
        }

    return indices


def parse_sgas_file(filepath: Path) -> dict | None:
    """
    Парсим один SGAS-файл и извлекаем необходимые данные.
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")

    # Извлекаем заголовки
    product_match = RE_PRODUCT.search(text)
    issued_match = RE_ISSUED.search(text)
    sgas_number_match = RE_SGAS_NUMBER.search(text)

    if not product_match or not issued_match or not sgas_number_match:
        print(f"[WARN] {filepath.name}: Отсутствуют заголовочные данные. Пропускаем.")
        return None

    product = product_match.group(1).strip()
    issued = issued_match.group(1).strip()
    sgas_number = int(sgas_number_match.group(1))

    # Извлекаем разделы
    section_a = extract_section(text, "A")
    section_b = extract_section(text, "B")
    section_c = extract_section(text, "C")
    section_e = extract_section(text, "E")

    # Парсим раздел A. Energetic Events
    energetic_events = []
    if section_a and "None" not in section_a:
        energetic_events = parse_energetic_events(section_a)
    else:
        energetic_events = "None"

    # Парсим раздел B. Proton Events
    proton_events = "None"
    if section_b:
        proton_events = parse_proton_events(section_b)

    # Парсим раздел C. Geomagnetic Activity Summary
    geomag_summary = ""
    if section_c:
        geomag_summary = parse_geomag_summary(section_c)

    # Парсим раздел E. Daily Indices
    daily_indices = {}
    if section_e:
        daily_indices = parse_daily_indices(section_e)

    # Формируем результат
    result = {
        "issued": issued,
        "product": product,
        "sgas_number": sgas_number,
        "energetic_events": energetic_events,
        "proton_events": proton_events,
        "geomagnetic_activity_summary": geomag_summary,
        "daily_indices": daily_indices
    }

    return result


def main():
    all_data = []

    for filepath in input_directory.glob("*.txt"):
        print(f"Обработка файла: {filepath.name}")
        parsed_data = parse_sgas_file(filepath)
        if parsed_data:
            all_data.append(parsed_data)
        else:
            print(f"[INFO] {filepath.name}: Данные не извлечены.")

    # Сохраняем результаты в JSON
    if all_data:
        output_file = output_directory / "sgas_2013_parsed.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"Сохранено {len(all_data)} записей в {output_file}")
    else:
        print("Не удалось извлечь данные из ни одного файла.")


if __name__ == "__main__":
    main()
