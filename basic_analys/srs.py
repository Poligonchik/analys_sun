import json
import pandas as pd
import numpy as np
import json
import numpy as np

# Рекурсивная функция для разворачивания вложенного словаря в плоский
def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            # Обрабатываем список: для каждого элемента, если он словарь, разворачиваем его,
            # иначе записываем как отдельное значение с индексом
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(flatten_dict(item, f"{new_key}[{i}]", sep=sep))
                else:
                    items[f"{new_key}[{i}]"] = item
        else:
            items[new_key] = v
    return items

# Функция для проверки, является ли значение пустым
def is_missing(val):
    if pd.isna(val):
        return True
    if isinstance(val, str):
        if val.strip() in ["*", "NaN", ""]:
            return True
    return False

# Загрузка данных из файла dsd.json
with open("../unified_json/dsd.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Предполагается, что данные представляют собой список словарей
flattened_data = [flatten_dict(record) for record in data]

# Создаем DataFrame из плоских записей
df_flat = pd.DataFrame(flattened_data)

# Подсчет пустых значений по каждому столбцу
missing_counts = df_flat.applymap(is_missing).sum()

print("Количество пустых значений по всем столбцам (с учетом вложенных структур):")
print(missing_counts)

input_file = "../unified_json/dsd.json"

# Загрузка данных из файла dsd.json
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)


# Функция для проверки пустых значений (None, пустая строка, "NaN" или "*")
def is_missing(val):
    if val is None:
        return True
    if isinstance(val, str) and val.strip() in ["", "NaN", "*"]:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False


# Обработка каждого словаря в данных
for record in data:
    # Удаляем ключи optical_flares и solar_field, если они существуют
    record.pop("optical_flares", None)
    record.pop("solar_field", None)

    # Обработка x_ray_flux: если значение отсутствует, заменяем на "A0.0"
    if "x_ray_flux" in record:
        if is_missing(record["x_ray_flux"]):
            record["x_ray_flux"] = "A0.0"
    else:
        record["x_ray_flux"] = "A0.0"

    # Обработка background: если значение отсутствует, заменяем на 0.0
    if "background" in record:
        if is_missing(record["background"]):
            record["background"] = 0.0
    else:
        record["background"] = 0.0

# Сохраняем изменённые данные обратно в файл dsd.json
with open(input_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Данные успешно обновлены и сохранены в", input_file)
