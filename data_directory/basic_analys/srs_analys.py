import pandas as pd
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import re

def basic_analysis_srs():
    # Путь к директории, где лежат файлы с результатами
    output_directory = Path("../processed_results")

    # ===== 1. Загрузка данных из Parquet =====
    sunspots_path = output_directory / "sunspots.parquet"
    h_alpha_path = output_directory / "h_alpha.parquet"
    return_path = output_directory / "return.parquet"

    if sunspots_path.exists():
        sunspots_df = pd.read_parquet(sunspots_path)
        print("=== Sunspots DataFrame ===")
        print(sunspots_df.head(10))       # первые 10 строк
        print(sunspots_df.info())         # информация о столбцах
        print(sunspots_df.describe())     # описательные статистики (числовые)
        print("Количество пропусков по столбцам:\n", sunspots_df.isna().sum())
        print("\n\n")
    else:
        print("Файл sunspots.parquet не найден.")

    if h_alpha_path.exists():
        h_alpha_df = pd.read_parquet(h_alpha_path)
        print("=== H-alpha Plages DataFrame ===")
        print(h_alpha_df.head(10))
        print(h_alpha_df.info())
        print(h_alpha_df.describe())
        print("Количество пропусков по столбцам:\n", h_alpha_df.isna().sum())
        print("\n\n")
    else:
        print("Файл h_alpha.parquet не найден.")

    if return_path.exists():
        return_df = pd.read_parquet(return_path)
        print("=== Regions Due to Return DataFrame ===")
        print(return_df.head(10))
        print(return_df.info())
        print(return_df.describe())
        print("Количество пропусков по столбцам:\n", return_df.isna().sum())
        print("\n\n")
    else:
        print("Файл return.parquet не найден.")

    # ===== 2. Загрузка данных из JSON (если нужно) =====
    json_path = output_directory / "combined_srs_all.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("=== Keys in combined_srs_all.json ===")
        print(list(data.keys()))
        # Например, можно посмотреть, сколько записей в каждом разделе:
        for key in data:
            print(f"{key}: {len(data[key])} записей")
        print("\nПример первых 5 записей для regions_with_sunspots:")
        print(data["regions_with_sunspots"][:5])
    else:
        print("Файл combined_srs_all.json не найден.")



def parse_location(loc_str: str):
    """
    Преобразует строку вида "N07W67" или "S23E45" в числовые широту и долготу.
    """
    pattern = r"^([NS])(\d{1,2})([EW])(\d{1,3})$"
    match = re.match(pattern, loc_str.strip())
    if not match:
        return None, None
    lat_sign = 1 if match.group(1) == "N" else -1
    lat_val = int(match.group(2))
    lon_sign = 1 if match.group(3) == "E" else -1
    lon_val = int(match.group(4))
    return lat_sign * lat_val, lon_sign * lon_val

def add_lat_lon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame столбцы 'lat' и 'lon', распарсив столбец 'Location'.
    """
    lat_vals, lon_vals = [], []
    for loc in df["Location"]:
        lat, lon = parse_location(str(loc))
        lat_vals.append(lat)
        lon_vals.append(lon)
    df["lat"] = lat_vals
    df["lon"] = lon_vals
    return df

def visualize_srs_data():
    output_directory = Path("../processed_results")

    sunspots_path = output_directory / "sunspots.parquet"
    h_alpha_path = output_directory / "h_alpha.parquet"
    return_path = output_directory / "return.parquet"

    if not sunspots_path.exists():
        print("Файл sunspots.parquet не найден.")
        return

    sunspots_df = pd.read_parquet(sunspots_path)
    print("=== Sunspots DataFrame: первые 5 строк ===")
    print(sunspots_df.head(5), "\n")

    # Приводим Area и NN к числовому типу
    sunspots_df["Area"] = pd.to_numeric(sunspots_df["Area"], errors="coerce")
    sunspots_df["NN"] = pd.to_numeric(sunspots_df["NN"], errors="coerce")
    print("Количество пропусков в столбцах Area и NN:\n", sunspots_df[["Area", "NN"]].isna().sum(), "\n")

    # Добавляем столбцы lat и lon, распарсив Location
    sunspots_df = add_lat_lon_columns(sunspots_df)

    # 1) Гистограмма распределения площади (Area)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=sunspots_df, x="Area", bins=30, kde=True)
    plt.title("Распределение площади пятен (Area)")
    plt.xlabel("Area")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show()

    # 2) Гистограмма распределения числа пятен (NN)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=sunspots_df, x="NN", bins=30, kde=True)
    plt.title("Распределение числа пятен (NN)")
    plt.xlabel("NN")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show()

    # 3) Диаграмма рассеяния: зависимость Area от NN
    plt.figure(figsize=(8, 5))
    subset_df = sunspots_df.dropna(subset=["Area", "NN"])
    sns.scatterplot(data=subset_df, x="Area", y="NN")
    plt.title("Зависимость Area от NN")
    plt.xlabel("Area")
    plt.ylabel("NN")
    plt.tight_layout()
    plt.show()

    # 4) Countplot для магнитных типов (Mag_Type)
    plt.figure(figsize=(8, 5))
    mag_subset = sunspots_df.dropna(subset=["Mag_Type"])
    sns.countplot(data=mag_subset, x="Mag_Type", order=mag_subset["Mag_Type"].value_counts().index)
    plt.title("Распределение магнитных типов (Mag_Type)")
    plt.xlabel("Mag_Type")
    plt.ylabel("Количество")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5) Тепловая карта корреляций между Area и NN
    num_cols = ["Area", "NN"]
    corr_df = sunspots_df[num_cols].dropna()
    corr_matrix = corr_df.corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr_matrix, annot=True, cmap="magma")
    plt.title("Корреляция между Area и NN")
    plt.tight_layout()
    plt.show()

    # Дополнительные графики:

    # 6) Pairplot для числовых столбцов: Area, NN, Lo, lat, lon
    # Приведём Lo к числовому типу
    sunspots_df["Lo"] = pd.to_numeric(sunspots_df["Lo"], errors="coerce")
    numeric_cols = ["Area", "NN", "Lo", "lat", "lon"]
    sns.pairplot(sunspots_df[numeric_cols].dropna())
    plt.suptitle("Pairplot для числовых показателей", y=1.02)
    plt.tight_layout()
    plt.show()

    # 7) Scatter plot: зависимость широты (lat) от площади (Area)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sunspots_df, x="lat", y="Area")
    plt.title("Зависимость Area от широты (lat)")
    plt.xlabel("Широта (lat)")
    plt.ylabel("Area")
    plt.tight_layout()
    plt.show()

    # 8) Scatter plot: зависимость широты (lat) от числа пятен (NN)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sunspots_df, x="lat", y="NN")
    plt.title("Зависимость NN от широты (lat)")
    plt.xlabel("Широта (lat)")
    plt.ylabel("NN")
    plt.tight_layout()
    plt.show()

    # 9) Boxplot: сравнение распределения площади (Area) по магнитным типам (Mag_Type)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=mag_subset, x="Mag_Type", y="Area", order=mag_subset["Mag_Type"].value_counts().index)
    plt.title("Boxplot площади (Area) по магнитным типам (Mag_Type)")
    plt.xlabel("Mag_Type")
    plt.ylabel("Area")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print("Визуализация завершена!")

def main():
    print("==== Базовый анализ SRS данных ====")
    basic_analysis_srs()
    print("\n==== Визуализация SRS данных ====")
    visualize_srs_data()

if __name__ == "__main__":
    main()