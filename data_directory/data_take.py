import os
import requests

# URL для загрузки данных
base_url = "https://services.swpc.noaa.gov/json/"

# Список JSON-файлов, которые нужно скачать
files_to_download = [
    "edited_events.json",  # Обновляется регулярно, проверьте наличие старых данных
    # Добавьте другие файлы, если нужно
]

# Директория для сохранения данных
save_dir = "./data/"
os.makedirs(save_dir, exist_ok=True)

# Загрузка файлов
for file_name in files_to_download:
    url = f"{base_url}{file_name}"
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Скачан файл: {file_name}")
    else:
        print(f"Не удалось скачать файл: {file_name} (Статус: {response.status_code})")
