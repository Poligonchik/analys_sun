import os
import time
import threading
from ftplib import FTP, error_perm, error_temp

# Настройки FTP-сервера
FTP_SERVER = "ftp.swpc.noaa.gov"
FTP_PATH = "/pub/warehouse/"    # Удалённая папка, которую нужно скачать
LOCAL_SAVE_PATH = "./ftp_data"  # Локальная папка для сохранения файлов

# Параметры keep-alive
KEEPALIVE_INTERVAL = 300  # Интервал отправки NOOP в секундах (например, 5 минут)

# Расширения файлов, которые нужно скачивать
ALLOWED_EXTENSIONS = (".json", ".txt")


def connect_ftp(server: str) -> FTP:
    """
    Подключается к FTP-серверу и возвращает объект FTP.
    """
    ftp = FTP(server, timeout=600)  # Устанавливаем таймаут соединения
    ftp.login()  # анонимный вход
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
    except:
        return False


def ftp_file_size(ftp: FTP, filename: str) -> int:
    """
    Возвращает размер файла в байтах или -1, если недоступен.
    """
    try:
        return ftp.size(filename)
    except:
        return -1


def ensure_directory(local_path: str):
    """
    Гарантирует, что по пути local_path существует директория.
    Если там существует одноимённый файл — удаляет его и создаёт папку.
    """
    if os.path.exists(local_path):
        if os.path.isfile(local_path):
            os.remove(local_path)
            os.makedirs(local_path, exist_ok=True)
    else:
        os.makedirs(local_path, exist_ok=True)


def list_ftp_contents(ftp: FTP, remote_path: str, indent: int = 0):
    """
    Рекурсивно выводит структуру файлов и папок, указывая размер для файлов.
    """
    ftp.cwd(remote_path)
    items = ftp.nlst()

    for item in items:
        if item in (".", ".."):
            continue

        display_name = "  " * indent + f"- {item}"

        if is_directory(ftp, item):
            print(display_name + "/")
            new_remote = f"{remote_path}/{item}"
            list_ftp_contents(ftp, new_remote, indent + 1)
        else:
            size = ftp_file_size(ftp, item)
            print(f"{display_name} (файл, {size} байт)")

    ftp.cwd("..")


def download_selected_files(ftp: FTP, remote_path: str, local_path: str,
                            allowed_extensions=ALLOWED_EXTENSIONS):
    """
    Рекурсивно обходит FTP-каталог и скачивает только те файлы,
    которые соответствуют списку 'allowed_extensions'.
    """
    # Создаём локальную папку
    ensure_directory(local_path)

    ftp.cwd(remote_path)
    items = ftp.nlst()

    for item in items:
        if item in (".", ".."):
            continue

        if is_directory(ftp, item):
            # Папка: рекурсивно заходим внутрь
            new_remote = f"{remote_path}/{item}"
            new_local = os.path.join(local_path, item)
            download_selected_files(ftp, new_remote, new_local, allowed_extensions)
        else:
            # Файл: проверяем расширение
            _, ext = os.path.splitext(item)
            if ext.lower() in allowed_extensions:
                local_file_path = os.path.join(local_path, item)
                if os.path.exists(local_file_path):
                    print(f"Файл уже существует и будет пропущен: {remote_path}/{item}")
                    continue  # Пропускаем уже скачанные файлы

                print(f"Скачиваем файл: {remote_path}/{item}")
                try:
                    with open(local_file_path, "wb") as f:
                        ftp.retrbinary(f"RETR {item}", f.write)
                except error_temp as e:
                    print(f"Ошибка при скачивании {item}: {e}")
                    # Можно реализовать повторную попытку здесь
                except Exception as e:
                    print(f"Не удалось скачать {item}: {e}")

    ftp.cwd("..")


def keep_alive(ftp: FTP, interval: int):
    """
    Функция, которая периодически отправляет команду NOOP, чтобы поддерживать соединение.
    Запускается в отдельном потоке.
    """
    while True:
        time.sleep(interval)
        try:
            ftp.voidcmd("NOOP")
            print("Отправлена команда NOOP для поддержания соединения.")
        except Exception as e:
            print(f"Ошибка при отправке NOOP: {e}")
            break  # Выходим из цикла, если не удалось отправить NOOP


if __name__ == "__main__":
    try:
        ftp = connect_ftp(FTP_SERVER)

        # Запуск keep-alive в отдельном потоке
        keep_alive_thread = threading.Thread(target=keep_alive, args=(ftp, KEEPALIVE_INTERVAL), daemon=True)
        keep_alive_thread.start()

        # Пример: скачиваем только .json и .txt файлы
        print(f"\nНачнём скачивание только .json и .txt файлов из {FTP_PATH}...\n")
        download_selected_files(ftp, FTP_PATH, LOCAL_SAVE_PATH, ALLOWED_EXTENSIONS)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        try:
            ftp.quit()
            print("\nСоединение с FTP закрыто.")
        except:
            pass
