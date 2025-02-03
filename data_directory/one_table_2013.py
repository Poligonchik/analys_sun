import pandas as pd
import os

# Путь к директории с файлами
directory = "./processed_data"

# Словарь для хранения данных по типам
data_by_type = {
    "DGD": [],
    "DSD": []
}

# Функция для фильтрации ненужных строк
def filter_data(data):
    # Удаляем строки, содержащие указанные слова
    filtered_data = data[~data.iloc[:, 0].str.contains("Prepared|Space|Joint|NOAA|Air Force", na=False)]
    return filtered_data

# Функция для очистки дат
def clean_dates(data):
    if ":Issued:" in data.columns:
        data[":Issued:"] = pd.to_datetime(data[":Issued:"], errors='coerce')
    return data

# Удаление пустых строк и столбцов
def remove_empty(data):
    # Удаляем строки, содержащие только NaN или пробелы
    data = data.replace(r'^\s*$', pd.NA, regex=True)  # Заменяем пробелы на NaN
    data = data.dropna(how="all", axis=0)  # Удаляем пустые строки
    data = data.dropna(how="all", axis=1)  # Удаляем пустые столбцы
    return data

# Классификация и загрузка данных
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        try:
            data = pd.read_csv(filepath)

            # Убираем пустые строки и столбцы сразу после загрузки
            data = remove_empty(data)

            if "DGD" in filename:
                data_by_type["DGD"].append(data)
            elif "DSD" in filename:
                data_by_type["DSD"].append(data)
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")

# Объединение файлов в каждой категории
for data_type, data_list in data_by_type.items():
    if data_list:
        try:
            # Объединяем данные в одну таблицу
            combined_data = pd.concat(data_list, ignore_index=True)

            # Удаляем пустые строки и столбцы
            combined_data = remove_empty(combined_data)

            # Удаляем ненужные строки
            combined_data = filter_data(combined_data)

            # Обрабатываем даты
            combined_data = clean_dates(combined_data)

            # Сохраняем объединенный файл
            output_path = os.path.join(directory, f"combined_{data_type}.csv")
            combined_data.to_csv(output_path, index=False)
            print(f"Сохранен файл: {output_path}")

            # Вывод первых строк для проверки
            print(f"Первые строки {data_type}:")
            print(combined_data.head())
        except Exception as e:
            print(f"Ошибка при объединении данных типа {data_type}: {e}")
