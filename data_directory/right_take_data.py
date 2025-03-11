import os
from ftplib import FTP, error_perm


def connect_ftp(server: str) -> FTP:
    """
    Подключение к FTP-серверу.
    """
    ftp = FTP(server)
    ftp.login()
    print(f"Подключились к : {server}")
    return ftp


def is_directory(ftp: FTP, directory: str) -> bool:
    """
    Проверяет, является ли 'item' папкой.
    """
    try:
        ftp.cwd(directory)  # пробую зайти в файл
        ftp.cwd("..")  # если получилось, то это папка, возвращаемся
        return True
    except error_perm:
        return False


def download_needed_files(ftp: FTP, remote_path: str, local_path: str):
    """
    Скачивает только нужные файлы, игнорируя ненужные папки.
    """
    os.makedirs(local_path, exist_ok=True)
    ftp.cwd(remote_path)
    items = ftp.nlst()
    print(f"Список файлов в {remote_path}: {items}")

    for item in items:
        if item in (".", ".."):
            continue
        # Проверка на папку
        full_item_path = f"{remote_path}/{item}"
        if is_directory(ftp, full_item_path):
            if "_plots" in item:    # plots папки я не беру, а то долго
                continue
            print(f"Заходим в папку: {item}")
            new_local_path = os.path.join(local_path, item)
            download_needed_files(ftp, full_item_path, new_local_path)
        else:
            if (
                item.endswith("_DSD.txt")
                or item.endswith("_DGD.txt")
                or item.endswith("_events.tar.gz")
                or item.endswith("_SGAS.tar.gz")
                or item.endswith("_SRS.tar.gz")
                or item.endswith("_RSGA.txt")
                or item.endswith("_RSGA.tar.gz")
            ):
                local_file_path = os.path.join(local_path, item)
                if os.path.exists(local_file_path):
                    print(f"Файл уже существует: {local_file_path}, пропускаем")
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
                print(f"Пропущенный файл: {item}")


if __name__ == "__main__":
    ftp_server = "ftp.swpc.noaa.gov"
    ftp_path = "/pub/warehouse/"
    local_save_path = "./ftp_data"

    ftp = connect_ftp(ftp_server)
    download_needed_files(ftp, ftp_path, local_save_path)
    ftp.quit()

    print("Процесс завершен, все файлы загружены в ./ftp_data")
