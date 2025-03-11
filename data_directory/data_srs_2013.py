import pandas as pd
import re
from pathlib import Path
import json

# Путь к директориям
input_directory = Path("./ftp_data/2013/2013_SRS")  # Укажите путь к папке с файлами
output_directory = Path("./processed_results")
output_directory.mkdir(parents=True, exist_ok=True)  # Создаем директорию, если она не существует


# Функция для парсинга
def parse_section(data, section_title, filename):
    section = []
    section_started = False
    for line in data:
        if section_title in line:  # Начало секции
            section_started = True
            continue
        if section_started:
            # Проверяем окончание секции
            if line.strip() == "" or re.match(r"^[A-Z]{2,}", line):
                break
            section.append(line.strip())
    if not section:
        print(f"Секция '{section_title}' не найдена или пуста в файле: {filename}")
    return section



# Парсинг одного файла
def parse_file(filepath):
    try:
        print(f"Обработка файла: {filepath}")
        with open(filepath, 'r') as f:
            data = f.readlines()

        # Извлечение заголовков
        issued_match = re.search(r":Issued:\s*(.+)", "".join(data))
        product_match = re.search(r":Product:\s*(.+)", "".join(data))

        if not issued_match or not product_match:
            print(f"Ошибка парсинга заголовков в файле: {filepath}")
            return None

        issued = issued_match.group(1)
        product = product_match.group(1)

        # парсинг частей нужных с реальными данными (дальше предсказания)
        regions_with_sunspots = parse_section(data, "I.  Regions with Sunspots", filepath)
        h_alpha_plages = parse_section(data, "IA. H-alpha Plages without Spots", filepath)
        regions_due_to_return = parse_section(data, "II. Regions Due to Return", filepath)
        # остальные не нужны

        return {
            "issued": issued,
            "product": product,
            "regions_with_sunspots": regions_with_sunspots,
            "h_alpha_plages": h_alpha_plages,
            "regions_due_to_return": regions_due_to_return,
        }
    except Exception as e:
        print(f"Ошибка обработки файла {filepath}: {e}")
        return None


def save_to_dataframe(parsed_data, section_key, columns):
    rows = []
    for entry in parsed_data:
        issued = entry["issued"]
        product = entry["product"]
        for line in entry[section_key]:
            # Пропускаем строки, которые являются заголовками
            if line.startswith("Nmbr") or re.match(r"^[A-Za-z ]+$", line):
                continue
            split_line = line.split()
            # Проверяем количество колонок
            if len(split_line) != len(columns):
                print(f"Проблемная строка в секции '{section_key}': {line}")
                print(f"Ожидаемое количество колонок: {len(columns)}, найдено: {len(split_line)}")
                continue
            rows.append([issued, product] + split_line)
    if not rows:
        print(f"Данные для секции '{section_key}' пусты.")
    return pd.DataFrame(rows, columns=["issued", "product"] + columns)




# Обработка файлов
parsed_data = []
for filepath in input_directory.glob("*.txt"):  # Берем только файлы с расширением .txt из директории
    parsed_entry = parse_file(filepath)
    if parsed_entry:
        parsed_data.append(parsed_entry)

# Проверка, есть ли данные после парсинга
if not parsed_data:
    print("Парсинг не удалось выполнить. Проверьте входные файлы.")
    exit()

# Создание таблиц
sunspots_columns = ["Nmbr", "Location", "Lo", "Area", "Z", "LL", "NN", "Mag_Type"]
sunspots_df = save_to_dataframe(parsed_data, "regions_with_sunspots", sunspots_columns)

h_alpha_columns = ["Nmbr", "Location", "Lo"]
h_alpha_df = save_to_dataframe(parsed_data, "h_alpha_plages", h_alpha_columns)

return_columns = ["Nmbr", "Lat", "Lo"]
return_df = save_to_dataframe(parsed_data, "regions_due_to_return", return_columns)

# Сохранение данных в JSON
output_json = output_directory / "combined_srs.json"
data_to_save = {
    "regions_with_sunspots": sunspots_df.to_dict(orient="records"),
    "h_alpha_plages": h_alpha_df.to_dict(orient="records"),
    "regions_due_to_return": return_df.to_dict(orient="records"),
}
with open(output_json, "w") as json_file:
    json.dump(data_to_save, json_file, indent=4)

# Сохранение данных в Parquet
sunspots_df.to_parquet(output_directory / "sunspots.parquet", index=False)
h_alpha_df.to_parquet(output_directory / "h_alpha.parquet", index=False)
return_df.to_parquet(output_directory / "return.parquet", index=False)

print(f"Данные успешно обработаны и сохранены в директорию: {output_directory}")
