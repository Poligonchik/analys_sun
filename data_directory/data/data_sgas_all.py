import re
import json
from pathlib import Path

# Корневая папка с файлами (структура: ../ftp_data/[год]/[год]_SGAS/*.txt)
input_directory = Path("../ftp_data")
# Папка для сохранения результата
output_directory = Path("../processed_results")
output_directory.mkdir(parents=True, exist_ok=True)

# --- Регулярные выражения для поиска заголовочных данных ---
RE_PRODUCT = re.compile(r":product:\s*(.+)", re.IGNORECASE)
RE_ISSUED = re.compile(r":issued:\s*(.+)", re.IGNORECASE)
RE_SGAS_NUMBER = re.compile(r"sgas\s*number\s*(\d+)", re.IGNORECASE)

RE_PROTON_EVENTS = re.compile(r"b\.\s*proton events:?\s*(.+)", re.IGNORECASE)
RE_GEOMAG_SUMMARY = re.compile(r"c\.\s*geomagnetic activity summary:?\s*(.+)", re.IGNORECASE)

RE_DAILY_10CM_SSNN_AP = re.compile(
    r"10\s*cm\s*(\d+)\s+ssn\s*(\d+)\s+afr/ap\s*(\d{3}/\d{3}).+x-ray background\s*([bfx]\d+\.\d+)",
    re.IGNORECASE
)
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
    Извлекает содержимое указанного раздела из полного текста файла.

    Для раздела A (Energetic Events) извлекаем текст между подстроками
    "a. energetic events" и "b. proton events".

    Inline-флаги (?si) позволяют работать в режиме DOTALL (точка соответствует переносу строк)
    и игнорировать регистр.
    Важно: мы заменяем только пробелы и табуляции, чтобы сохранить переносы строк.
    """
    if section_name.lower() == "a":
        pattern = r"(?si)a\.\s*energetic events\s+(.*?)(?=b\.\s*proton events)"
    else:
        pattern = rf"(?si){section_name}\.\s+(.*?)(?=\n\s*[a-z]\.|$)"
    match = re.search(pattern, text)
    if match:
        section_text = match.group(1).strip()
        # Заменяем последовательности пробелов и табуляций на один пробел, сохраняя переносы строк
        section_text = re.sub(r"[ \t]+", " ", section_text)
        return section_text
    else:
        return ""


def parse_energetic_events(section_text: str):
    """
    Разбирает содержимое раздела A (Energetic Events) из SGAS файла.

    Ожидаемый формат (после заголовка):
      Begin  Max  End  Rgn   Loc   Xray  Op 245MHz 10cm   Sweep
       0004 0021 0041  1667 N22E19 C8.7  1f 950           II/IV
       0112 0112 0112                       150
       0127 0127 0127                       100
       0133 0133 0134                       150

    Функция:
      - Разбивает текст на строки.
      - Пропускает первую строку, если она содержит слово "begin" (заголовок).
      - Для остальных строк, если первый токен (после split) начинается с цифры,
        разбивает строку на токены. Если токенов меньше 9, дополняет пустыми строками.
    Возвращает список словарей с ключами:
      "begin", "max", "end", "region", "location", "xray_class", "op_245mhz", "op_10cm", "sweep".
    """
    events = []
    lines = section_text.splitlines()
    if lines and ("begin" in lines[0].lower() and "max" in lines[0].lower()):
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if not tokens or not tokens[0][0].isdigit():
            continue
        while len(tokens) < 9:
            tokens.append("")
        tokens = tokens[:9]
        event = {
            "begin": tokens[0],
            "max": tokens[1],
            "end": tokens[2],
            "region": tokens[3],
            "location": tokens[4],
            "xray_class": tokens[5],
            "op_245mhz": tokens[6],
            "op_10cm": tokens[7],
            "sweep": tokens[8]
        }
        events.append(event)
    return events


def parse_proton_events(section_text: str) -> str:
    return section_text.strip().lower() if section_text else "none"


def parse_geomag_summary(section_text: str) -> str:
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
            indices["daily_electron_fluence"] = {"gt_2_mev": float(match.group(1))}
        except:
            indices["daily_electron_fluence"] = None
    else:
        indices["daily_electron_fluence"] = None
    match = RE_K_INDICES.search(section_text)
    if match:
        try:
            boulder_values = [int(x) for x in match.group(1).split()]
            planetary_values = [int(x) for x in match.group(2).split()]
            indices["k_indices"] = {"boulder": boulder_values, "planetary": planetary_values}
        except:
            indices["k_indices"] = None
    else:
        indices["k_indices"] = None
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
    energetic_events = parse_energetic_events(section_a) if section_a.strip() else []
    proton_events = parse_proton_events(section_b)
    geomag_summary = parse_geomag_summary(section_c)
    daily_indices = parse_daily_indices(section_e)
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
