import json
import pandas as pd

# Путь к файлу edited_events.json
file_path = "data/edited_events.json"  # Убедитесь, что файл находится в этой директории или измените путь

# Открываем и загружаем данные
with open(file_path, "r") as file:
    data = json.load(file)

# Проверяем первые несколько записей
print("Количество записей:", len(data))
print("Пример первой записи:")
print(data[0])


# Преобразуем данные в DataFrame
df = pd.DataFrame(data)

# Преобразуем даты в datetime формат
df['begin_datetime'] = pd.to_datetime(df['begin_datetime'])
df['end_datetime'] = pd.to_datetime(df['end_datetime'])

# Найдем первую и последнюю дату
first_date = df['begin_datetime'].min()
last_date = df['begin_datetime'].max()

print(f"Данные охватывают период с {first_date} по {last_date}")
