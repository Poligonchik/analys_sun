import pathlib


def count_files(root_dir):
    """
    Рекурсивно проходит по всем файлам в root_dir и считает количество файлов
    с заданными суффиксами, учитывая варианты с и без подчёркивания.

    :param root_dir: Корневая директория для поиска файлов.
    :return: Кортеж из общего количества файлов и словаря с подсчётом по суффиксам.
    """
    # Определение суффиксов для поиска (без подчёркивания)
    base_suffixes = [
        "DSD.txt",
        "DGD.txt",
        "events.txt",
        "SGAS.txt",
        "SRS.txt",
        "RSGA.txt"
    ]

    # Создание списка суффиксов с и без подчёркивания
    suffixes = []
    for suffix in base_suffixes:
        suffixes.append("_" + suffix)
        suffixes.append(suffix)

    # Инициализация счётчиков
    total_count = 0
    suffix_counts = {suffix: 0 for suffix in base_suffixes}

    # Создание объекта Path для корневой директории
    root_path = pathlib.Path(root_dir)

    if not root_path.exists():
        print(f"Директория {root_dir} не существует.")
        return total_count, suffix_counts

    # Рекурсивный проход по всем файлам
    for file_path in root_path.rglob('*'):
        if file_path.is_file():
            matched = False
            for base_suffix in base_suffixes:
                if file_path.name.endswith("_" + base_suffix) or file_path.name.endswith(base_suffix):
                    total_count += 1
                    suffix_counts[base_suffix] += 1
                    matched = True
                    print(f"Найден файл: {file_path.name} (суффикс: {base_suffix})")
                    break  # Предотвращает двойной счёт, если файл соответствует нескольким суффиксам
            if not matched:
                print(f"Файл не соответствует суффиксам: {file_path.name}")

    return total_count, suffix_counts


def main():
    # Задание корневой директории
    root_directory = './ftp_data'

    # Получение подсчёта файлов
    total, counts = count_files(root_directory)

    # Вывод результатов
    print(f"\nОбщее количество файлов с заданными суффиксами: {total}")
    print("Количество файлов по каждому суффиксу:")
    for suffix, count in counts.items():
        print(f"  {suffix}: {count}")


if __name__ == "__main__":
    main()
