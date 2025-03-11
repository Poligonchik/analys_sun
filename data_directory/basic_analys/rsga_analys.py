import pandas as pd
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import re

# --- Регулярные выражения для RSGA (без изменений) ---
RE_SDF_NUMBER = re.compile(r"SDF Number\s+(\d+)", re.IGNORECASE)
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

def analyze_rsga_relationships():
    """
    Функция анализирует взаимосвязи между различными характеристиками RSGA-данных:
      1. Корреляционная матрица для числовых полей.
      2. Pairplot для числовых полей.
      3. Scatter plot для зависимости solar_wind_speed_peak от total_imf_max.
      4. Частотное распределение (bar plot) для нормализованных категориальных полей
         solar_activity_level и geomagnetic_activity_level.
    """
    output_directory = Path("../processed_results")
    json_path = output_directory / "rsga_combined_all.json"
    if not json_path.exists():
        print("Файл rsga_combined_all.json не найден.")
        return

    df = load_rsga_data(json_path)

    # Приводим некоторые поля к числовому типу
    df["sdf_number"] = pd.to_numeric(df["sdf_number"], errors="coerce")
    df["solar_wind_speed_peak"] = pd.to_numeric(df["solar_wind_speed_peak"], errors="coerce")
    df["total_imf_max"] = pd.to_numeric(df["total_imf_max"], errors="coerce")
    df["electron_flux_peak"] = pd.to_numeric(df["electron_flux_peak"], errors="coerce")

    # Определяем числовые поля для анализа
    numeric_cols = ["sdf_number", "solar_wind_speed_peak", "total_imf_max", "electron_flux_peak"]
    corr_matrix = df[numeric_cols].corr()

    # 1. Корреляционная матрица
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Корреляционная матрица для числовых характеристик")
    plt.tight_layout()
    plt.show()

    # 2. Pairplot для числовых характеристик
    sns.pairplot(df[numeric_cols].dropna())
    plt.suptitle("Pairplot для числовых характеристик", y=1.02)
    plt.tight_layout()
    plt.show()

    # 3. Scatter plot: зависимость между Solar Wind Speed Peak и Total IMF Max
    subset = df.dropna(subset=["solar_wind_speed_peak", "total_imf_max"])
    if not subset.empty:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=subset, x="solar_wind_speed_peak", y="total_imf_max", hue="sdf_number", palette="viridis")
        plt.title("Зависимость Total IMF Max от Solar Wind Speed Peak")
        plt.xlabel("Solar Wind Speed Peak (km/s)")
        plt.ylabel("Total IMF Max (nT)")
        plt.tight_layout()
        plt.show()
    else:
        print("Нет данных для построения зависимости Solar Wind Speed vs Total IMF.")

    # 4. Анализ распределения категориальных полей
    # Для solar_activity_level: приведем к нижнему регистру и уберем лишние слова.
    if "solar_activity_level" in df.columns:
        df["solar_activity_level_norm"] = df["solar_activity_level"].str.lower().str.strip()
        plt.figure(figsize=(10, 5))
        order = df["solar_activity_level_norm"].value_counts().index
        sns.countplot(data=df, x="solar_activity_level_norm", order=order)
        plt.title("Распределение значений solar_activity_level")
        plt.xlabel("Solar Activity Level")
        plt.ylabel("Количество записей")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Поле solar_activity_level отсутствует в данных.")

    # Для geomagnetic_activity_level:
    if "geomagnetic_activity_level" in df.columns:
        df["geomagnetic_activity_level_norm"] = df["geomagnetic_activity_level"].str.lower().str.strip()
        plt.figure(figsize=(10, 5))
        order = df["geomagnetic_activity_level_norm"].value_counts().index
        sns.countplot(data=df, x="geomagnetic_activity_level_norm", order=order)
        plt.title("Распределение значений geomagnetic_activity_level")
        plt.xlabel("Geomagnetic Activity Level")
        plt.ylabel("Количество записей")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Поле geomagnetic_activity_level отсутствует в данных.")


def extract_section(text: str, section_name: str) -> str:
    """
    Извлекает содержимое секции (например, IA. или IIA.) из текста.
    Ищет содержимое после 'section_name.' до появления нового раздела (начинающегося с I[A-Z]{1,2}.) или до конца текста.
    """
    pattern = rf"(?s){section_name}\.\s+(.*?)(?=\nI[A-Z]{{1,2}}\.|$)"
    match = re.search(pattern, text)
    if not match:
        return ""
    section_text = match.group(1)
    section_text = re.sub(r"\s+", " ", section_text).strip()
    return section_text


def parse_rsgi_file(filepath: Path) -> dict | None:
    """
    Парсит файл RSGA, извлекая заголовки и секции IA и IIA, затем данные по ключевым регуляркам.
    """
    try:
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


def load_rsga_data(json_path: Path) -> pd.DataFrame:
    """
    Загружает объединённые данные RSGA из JSON-файла и возвращает DataFrame.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Если JSON представлен как словарь с несколькими ключами, объединяем записи:
    if isinstance(data, dict):
        all_records = []
        for key in data:
            all_records.extend(data[key])
        df = pd.DataFrame(all_records)
    else:
        df = pd.DataFrame(data)
    return df


def detailed_field_analysis():
    """
    Проводит детальный анализ основных полей RSGA:
    - Выводит общее число записей, число непустых значений (учитывая NaN и пустые строки)
    - Выводит таблицы распределения (value_counts) для выбранных полей.
    """
    output_directory = Path("../processed_results")
    json_path = output_directory / "rsga_combined_all.json"
    if not json_path.exists():
        print("Файл rsga_combined_all.json не найден.")
        return

    df = load_rsga_data(json_path)
    # Определяем список анализируемых столбцов
    fields = [
        "sdf_number", "solar_activity_level", "largest_event_class",
        "largest_event_time", "largest_event_region", "largest_event_coords",
        "num_sunspot_regions", "geomagnetic_activity_level", "solar_wind_speed_peak",
        "solar_wind_speed_peak_time", "total_imf_max", "total_imf_max_time",
        "bz_min", "bz_min_time", "electron_flux_peak"
    ]

    print("=== Детальный анализ полей RSGA ===")
    for field in fields:
        print(f"--- Анализ поля: {field} ---")
        total = len(df)
        # Для строковых полей считаем пустые строки как пустые
        non_empty = df[field].astype(str).str.strip() != ""
        count_nonempty = non_empty.sum()
        count_missing = total - count_nonempty
        print(f"Всего записей: {total}, Непустых: {count_nonempty}, Пустых: {count_missing}")
        # Выводим value_counts (только для нечисловых или для небольшого числа уникальных значений)
        try:
            # Если можно привести к числовому типу – попробуем
            df[field] = pd.to_numeric(df[field], errors="ignore")
        except Exception:
            pass
        vc = df[field].value_counts(dropna=False)
        print(vc)
        print("\n")


def basic_analysis_rsga():
    output_directory = Path("../processed_results")
    json_path = output_directory / "rsga_combined_all.json"
    if not json_path.exists():
        print("Файл rsga_combined_all.json не найден.")
        return

    df = load_rsga_data(json_path)
    print("=== Общая информация по RSGA ===")
    print(df.info())
    print("\n=== Первые 10 записей ===")
    print(df.head(10))
    print("\n=== Описательная статистика для числовых полей (sdf_number, year) ===")
    df["sdf_number"] = pd.to_numeric(df["sdf_number"], errors="coerce")
    print(df[["sdf_number", "year"]].describe())

    counts_by_year = df["year"].value_counts().sort_index()
    print("\nКоличество записей по годам:")
    print(counts_by_year)

    plt.figure(figsize=(10, 5))
    counts_by_year.plot(kind="bar")
    plt.title("Количество RSGA записей по годам")
    plt.xlabel("Год")
    plt.ylabel("Количество записей")
    plt.tight_layout()
    plt.show()


def main():
    print("==== Базовый анализ RSGA данных ====")
    basic_analysis_rsga()
    print("\n==== Детальный анализ полей RSGA ====")
    detailed_field_analysis()
    print("==== Анализ взаимосвязей в RSGA данных ====")
    analyze_rsga_relationships()

if __name__ == "__main__":
    main()
