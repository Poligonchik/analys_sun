import re
import json
from pathlib import Path

# Корневая папка с файлами (структура: ../ftp_data/[год]/[год]_SGAS/*.txt)
input_directory = Path("../ftp_data")
# Папка для сохранения результата
output_directory = Path("../processed_results")
output_directory.mkdir(parents=True, exist_ok=True)

# --- Регулярные выражения для поиска ключевых данных ---
RE_PRODUCT = re.compile(r":product:\s*(.+)", re.IGNORECASE)
RE_ISSUED = re.compile(r":issued:\s*(.+)", re.IGNORECASE)
RE_SGAS_NUMBER = re.compile(r"sgas\s*number\s*(\d+)", re.IGNORECASE)

# Раздел A. Energetic Events
RE_ENERGETIC_EVENTS_HEADER = re.compile(r"a\.\s*energetic events", re.IGNORECASE)
RE_ENERGETIC_EVENT_LINE = re.compile(
    r"(\d{3,4}|[a-z0-9]+)\s+(\d{3,4}|[a-z0-9/]+)\s+(\d{3,4}|[a-z0-9/]+)\s*(\d+)?\s*([a-z0-9\.\-]+)?\s*([a-z0-9\.\-]+)?\s*([a-z0-9\.\-]+)?\s*([a-z0-9\.\-]+)?",
    re.IGNORECASE
)

RE_PROTON_EVENTS = re.compile(r"b\.\s*proton events:?\s*(.+)", re.IGNORECASE)
RE_GEOMAG_SUMMARY = re.compile(r"c\.\s*geomagnetic activity summary:?\s*(.+)", re.IGNORECASE)

# Раздел E. Daily Indices
RE_DAILY_INDICES_HEADER = re.compile(r"e\.\s*daily indices:?", re.IGNORECASE)
# Основной шаблон, если формат "10 cm 147 ssn 094 afr/ap 008/007 x-ray background b6.8"
RE_DAILY_10CM_SSNN_AP = re.compile(
    r"10\s*cm\s*(\d+)\s+ssn\s*(\d+)\s+afr/ap\s*(\d{3}/\d{3}).+x-ray background\s*([bfx]\d+\.\d+)",
    re.IGNORECASE
)
# Альтернативный поиск для 10 cm (например, если там просто "10 cm 22")
RE_ALT_TEN_CM = re.compile(r"10\s*cm\s+(\d+)", re.IGNORECASE)

RE_DAILY_PROTON_FLUENCE = re.compile(
    r"daily proton fluence.*gt\s*1 mev\s*([\d\.e\+\-]+).+gt\s*10 mev\s*([\d\.e\+\-]+)",
    re.IGNORECASE | re.DOTALL
)
RE_DAILY_ELECTRON_FLUENCE = re.compile(
    r"daily electron fluence.*gt\s*2 mev\s*([\d\.e\+\-]+)",
    re.IGNORECASE | re.DOTALL
)
RE_K_INDICES = re.compile(
    r"3 hour k[- ]?indices:?\s+boulder\s+([\d\s]+)\s+planetary\s+([\d\s]+)",
    re.IGNORECASE
)

def extract_section(text: str, section_name: str) -> str:
    """
    Извлекает содержимое указанного раздела (например, "a", "b", "c", "e")
    из полного текста файла.
    """
    pattern = rf"(?s){section_name}\.\s+(.*?)(?=\n[a-z]\.|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        section_text = match.group(1).strip()
        section_text = re.sub(r"\s+", " ", section_text).lower()
        return section_text
    else:
        return ""

def parse_energetic_events(section_text: str):
    events = []
    matches = RE_ENERGETIC_EVENT_LINE.findall(section_text)
    if matches:
        for match in matches:
            begin, max_time, end, rgn, loc, xray, op_245MHz, sweep = match
            event = {
                "begin": begin.lower(),
                "max": max_time.lower(),
                "end": end.lower(),
                "region": rgn.lower() if rgn else "",
                "location": loc.lower() if loc else "",
                "xray_class": xray.lower() if xray else "",
                "op_245mhz": op_245MHz.lower() if op_245MHz else "",
                "op_10cm": "",  # При необходимости можно добавить логику
                "sweep": sweep.lower() if sweep else ""
            }
            events.append(event)
    else:
        events = "none"
    return events

def parse_proton_events(section_text: str) -> str:
    return section_text.strip().lower() if section_text else "none"

def parse_geomag_summary(section_text: str) -> str:
    """
    Преобразует полное описание геомагнитной активности в краткую метку (1–2 слова)
    для последующего анализа.
    """
    s = section_text.strip().lower() if section_text else ""
    if "quiet" in s and "unsettled" in s:
        return "quiet unsettled"
    elif "quiet" in s and "active" in s:
        return "quiet active"
    elif "quiet" in s:
        return "quiet"
    elif "active" in s:
        return "active"
    elif "unsettled" in s:
        return "unsettled"
    else:
        return s

def parse_daily_indices(section_text: str) -> dict:
    indices = {}
    match = RE_DAILY_10CM_SSNN_AP.search(section_text)
    if match:
        try:
            indices["10cm"] = int(match.group(1))
        except:
            indices["10cm"] = None
        try:
            indices["ssn"] = int(match.group(2))
        except:
            indices["ssn"] = None
        indices["afr_ap"] = match.group(3)
        indices["xray_background"] = match.group(4)
    else:
        alt_match = RE_ALT_TEN_CM.search(section_text)
        indices["10cm"] = int(alt_match.group(1)) if alt_match else None

        ssn_match = re.search(r"ssn\s+(\d+)", section_text, re.IGNORECASE)
        indices["ssn"] = int(ssn_match.group(1)) if ssn_match else None

        afr_ap_match = re.search(r"afr/ap\s+(\d{3}/\d{3})", section_text, re.IGNORECASE)
        indices["afr_ap"] = afr_ap_match.group(1) if afr_ap_match else None

        xray_match = re.search(r"x-ray background\s+([bfx]\d+\.\d+)", section_text, re.IGNORECASE)
        indices["xray_background"] = xray_match.group(1) if xray_match else None

    match = RE_DAILY_PROTON_FLUENCE.search(section_text)
    if match:
        try:
            indices["daily_proton_fluence"] = {
                "gt_1_mev": float(match.group(1)),
                "gt_10_mev": float(match.group(2))
            }
        except:
            indices["daily_proton_fluence"] = None
    else:
        indices["daily_proton_fluence"] = None

    match = RE_DAILY_ELECTRON_FLUENCE.search(section_text)
    if match:
        try:
            indices["daily_electron_fluence"] = {
                "gt_2_mev": float(match.group(1))
            }
        except:
            indices["daily_electron_fluence"] = None
    else:
        indices["daily_electron_fluence"] = None

    match = RE_K_INDICES.search(section_text)
    if match:
        try:
            boulder_values = [int(x) for x in match.group(1).split()]
            planetary_values = [int(x) for x in match.group(2).split()]
            indices["k_indices"] = {
                "boulder": boulder_values,
                "planetary": planetary_values
            }
        except:
            indices["k_indices"] = None
    else:
        indices["k_indices"] = None

    # Удаляем ключи, для которых данные не найдены (None)
    indices = {k: v for k, v in indices.items() if v not in [None, ""]}
    return indices

def parse_sgas_file(filepath: Path) -> dict | None:
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[ERROR] Чтение {filepath.name}: {e}")
        return None

    text_lower = text.lower()

    product_match = RE_PRODUCT.search(text_lower)
    issued_match = RE_ISSUED.search(text_lower)
    sgas_number_match = RE_SGAS_NUMBER.search(text_lower)

    if not product_match or not issued_match or not sgas_number_match:
        print(f"[WARN] {filepath.name}: Отсутствуют заголовочные данные, пропуск файла.")
        return None

    product = product_match.group(1).strip()
    issued = issued_match.group(1).strip()
    sgas_number = int(sgas_number_match.group(1))

    section_a = extract_section(text_lower, "a")
    section_b = extract_section(text_lower, "b")
    section_c = extract_section(text_lower, "c")
    section_e = extract_section(text_lower, "e")

    energetic_events = parse_energetic_events(section_a) if section_a and "none" not in section_a else "none"
    proton_events = parse_proton_events(section_b)
    geomag_summary = parse_geomag_summary(section_c)
    daily_indices = parse_daily_indices(section_e)

    # Если daily_indices пуст, не включаем его в итоговый результат
    result = {
        "issued": issued,
        "product": product,
        "sgas_number": sgas_number,
        "energetic_events": energetic_events,
        "proton_events": proton_events,
        "geomagnetic_activity_summary": geomag_summary,
    }
    if daily_indices:
        result["daily_indices"] = daily_indices

    return result

def main():
    all_data = []
    # Рекурсивный поиск файлов SGAS во всех подпапках, где в имени папки встречается SGAS
    pattern = input_directory.glob("**/*SGAS/*.txt")
    for filepath in pattern:
        print(f"Обработка файла: {filepath.name}")
        parsed_data = parse_sgas_file(filepath)
        if parsed_data:
            all_data.append(parsed_data)
        else:
            print(f"[INFO] {filepath.name}: данные не извлечены.")

    if all_data:
        output_file = output_directory / "sgas_all.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"Сохранено {len(all_data)} записей в {output_file}")
    else:
        print("Не удалось извлечь данные ни из одного файла.")

if __name__ == "__main__":
    main()
