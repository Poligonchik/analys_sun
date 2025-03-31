import json
import pandas as pd
import numpy as np

# Тут я обнаружила ошибку, поэтому исправляю ее


def is_missing(val):
    if val is None:
        return True
    if isinstance(val, str) and val.strip() in ["", "NaN", "*"]:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return False

input_file = "../result_json/dsd.json"

# Загрузка данных из файла dsd.json
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

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
    # Затем удаляем ключ background
    record.pop("background", None)

# Сохраняем изменённые данные обратно в файл dsd.json
with open(input_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Данные успешно обновлены и сохранены в", input_file)
