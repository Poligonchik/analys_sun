import os
from ftplib import FTP, error_perm


def connect_ftp(server: str) -> FTP:
    """
    Подключение к FTP-серверу.
    """
    ftp = FTP(server)
    ftp.login()
    print(f"Успешно подключились к FTP: {server}")
    ftp.cwd('/pub/warehouse/')  # Переход в директорию с файлами
    print("Текущая директория на FTP-сервере:", ftp.pwd())
    return ftp


def download_specific_files(ftp: FTP, file_list: list, local_path: str):
    """
    Скачивает указанные файлы в локальную директорию.
    """
    os.makedirs(local_path, exist_ok=True)

    # Получаем список всех файлов в текущей директории
    items = ftp.nlst()
    print(f"Файлы в текущей директории: {items}")

    for file_name in file_list:
        if file_name in items:
            local_file_path = os.path.join(local_path, file_name)
            if os.path.exists(local_file_path):
                print(f"Файл уже существует: {local_file_path}, пропуск...")
                continue

            try:
                print(f"Скачивание файла: {file_name}")
                with open(local_file_path, "wb") as f:
                    ftp.retrbinary(f"RETR {file_name}", f.write)
                print(f"Файл успешно скачан: {local_file_path}")
            except error_perm as e:
                print(f"Ошибка при скачивании {file_name}: {e}")
        else:
            print(f"Файл не найден на сервере: {file_name}")


if __name__ == "__main__":
    ftp_server = "ftp.swpc.noaa.gov"
    local_save_path = "../ftp_data"

    # Список файлов для скачивания
    files_to_download = [
        "1988_RSGA.tar.gz",
        "1989_RSGA.tar.gz",
        "1990_RSGA.tar.gz",
        "1991_RSGA.tar.gz",
        "1992_RSGA.tar.gz",
        "1993_RSGA.tar.gz",
        "1994_RSGA.tar.gz",
        "1995_RSGA.tar.gz"
    ]

    # Подключение к FTP-серверу и загрузка данных
    ftp = connect_ftp(ftp_server)
    download_specific_files(ftp, files_to_download, local_save_path)
    ftp.quit()

    print("Все доступные файлы загружены в директорию:", local_save_path)
