import pandas as pd
import json
import matplotlib.pyplot as plt

# Укажите путь к файлу JSON
json_path = "./processed_results/combined_srs.json"

# Загрузка данных из JSON
try:
    with open(json_path, "r") as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Файл {json_path} не найден. Проверьте путь.")
    exit()

# Преобразуем секции в DataFrame
sunspots_df = pd.DataFrame(data["regions_with_sunspots"])
h_alpha_df = pd.DataFrame(data["h_alpha_plages"])
return_df = pd.DataFrame(data["regions_due_to_return"])

# Пример анализа данных
# 1. Общее количество записей в каждой секции
print("Количество записей в секции 'regions_with_sunspots':", len(sunspots_df))
print("Количество записей в секции 'h_alpha_plages':", len(h_alpha_df))
print("Количество записей в секции 'regions_due_to_return':", len(return_df))

# 2. Топ-5 регионов с наибольшей площадью (Area)
if not sunspots_df.empty:
    sunspots_df["Area"] = pd.to_numeric(sunspots_df["Area"], errors="coerce")  # Преобразуем Area в числовой формат
    top_areas = sunspots_df.nlargest(5, "Area")
    print("\nТоп-5 регионов с наибольшей площадью (Area):")
    print(top_areas[["Nmbr", "Location", "Area"]])

# 3. Частота появления магнитных типов (Mag_Type)
if not sunspots_df.empty:
    mag_type_counts = sunspots_df["Mag_Type"].value_counts()
    print("\nЧастота появления магнитных типов (Mag_Type):")
    print(mag_type_counts)

# 4. Уникальные значения в секции 'h_alpha_plages'
if not h_alpha_df.empty:
    unique_locations = h_alpha_df["Location"].unique()
    print("\nУникальные местоположения в секции 'h_alpha_plages':")
    print(unique_locations)

# 5. Минимальные и максимальные долготы (Lo) в секции 'regions_due_to_return'
if not return_df.empty:
    return_df["Lo"] = pd.to_numeric(return_df["Lo"], errors="coerce")
    min_longitude = return_df["Lo"].min()
    max_longitude = return_df["Lo"].max()
    print("\nМинимальная долгота в 'regions_due_to_return':", min_longitude)
    print("Максимальная долгота в 'regions_due_to_return':", max_longitude)

if not sunspots_df.empty:
    mag_type_counts.plot(kind="bar", title="Частота появления магнитных типов")
    plt.xlabel("Mag_Type")
    plt.ylabel("Количество")
    plt.show()
