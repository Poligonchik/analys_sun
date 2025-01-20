import os
import tarfile
from ftplib import FTP, error_perm
import pandas as pd


def connect_ftp(server: str) -> FTP:
    """
    Подключение к FTP-серверу.
    """
    ftp = FTP(server)
    ftp.login()
    print(f"Успешно подключились к FTP: {server}")
    return ftp


def is_directory(ftp: FTP, item: str) -> bool:
    """
    Проверяет, является ли 'item' папкой.
    """
    original_cwd = ftp.pwd()
    try:
        ftp.cwd(item)
        ftp.cwd(original_cwd)
        return True
    except error_perm:
        return False


def download_needed_files(ftp: FTP, remote_path: str, local_path: str):
    """
    Скачивает только нужные файлы с проверкой на размер.
    """
    os.makedirs(local_path, exist_ok=True)
    ftp.cwd(remote_path)
    items = ftp.nlst()
    print(f"Список файлов в {remote_path}: {items}")

    for item in items:
        if item in (".", ".."):
            continue

        if is_directory(ftp, item):
            new_remote_path = f"{remote_path}/{item}"
            new_local_path = os.path.join(local_path, item)
            download_needed_files(ftp, new_remote_path, new_local_path)
        else:
            if (
                item.endswith("_DSD.txt")
                or item.endswith("_DGD.txt")
                or item.endswith("_events.tar.gz")
                or item.endswith("_SGAS.tar.gz")
                or item.endswith("_SRS.tar.gz")
            ):
                local_file_path = os.path.join(local_path, item)
                if os.path.exists(local_file_path):
                    print(f"Файл уже существует: {local_file_path}, пропуск...")
                    continue
                try:
                    with open(local_file_path, "wb") as f:
                        ftp.retrbinary(f"RETR {item}", f.write)
                        if os.path.getsize(local_file_path) == 0:
                            print(f"Пропуск пустого файла: {local_file_path}")
                            os.remove(local_file_path)
                        else:
                            print(f"Скачан файл: {local_file_path}")
                except error_perm as e:
                    print(f"Ошибка при скачивании {item}: {e}")
            else:
                print(f"Пропущен файл: {item} (не нужен)")


def extract_tar_gz(file_path: str, extract_to: str):
    """
    Извлечение файлов из архива .tar.gz.
    """
    if os.path.getsize(file_path) == 0:
        print(f"Пропуск пустого архива: {file_path}")
        return

    os.makedirs(extract_to, exist_ok=True)
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
            print(f"Извлечено: {file_path}")
    except tarfile.ReadError as e:
        print(f"Ошибка при извлечении {file_path}: {e}")


def parse_txt_to_csv(file_path: str, output_path: str):
    """
    Преобразует текстовый файл в CSV, дополняя строки пустыми значениями.
    """
    with open(file_path, "r") as file:
        data = [line.strip().split() for line in file if line.strip()]  # Убираем пустые строки
        max_columns = max(len(row) for row in data)  # Находим максимальное число колонок
        data = [row + [""] * (max_columns - len(row)) for row in data]  # Дополняем недостающие значения

        print(f"Обработка файла {file_path}, первые строки данных:\n{data[:5]}")
        df = pd.DataFrame(data[1:], columns=data[0])  # Используем первую строку как заголовок
        output_file = os.path.join(output_path, os.path.basename(file_path).replace(".txt", ".csv"))
        df.to_csv(output_file, index=False)
        print(f"Файл сохранен: {output_file}")


def process_all_files(local_path: str, output_path: str):
    """
    Обработка всех файлов в указанной локальной директории.
    """
    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".tar.gz"):
                extract_tar_gz(file_path, os.path.join(output_path, os.path.splitext(file)[0]))

            elif file.endswith(".txt"):
                parse_txt_to_csv(file_path, output_path)


if __name__ == "__main__":
    ftp_server = "ftp.swpc.noaa.gov"
    ftp_path = "/pub/warehouse/"
    local_save_path = "./ftp_data"
    output_path = "./processed_data"

    # Подключение к FTP-серверу и загрузка данных
    ftp = connect_ftp(ftp_server)
    print(f"Загрузка данных из {ftp_path}...")
    download_needed_files(ftp, ftp_path, local_save_path)
    ftp.quit()

    # Обработка загруженных файлов
    print("Обработка файлов...")
    process_all_files(local_save_path, output_path)

    print("Все данные обработаны и сохранены.")
