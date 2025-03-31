import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path


def load_dsd_data(filepath: Path) -> pd.DataFrame:
    """
    Загружает данные из файла DSD (JSON), приводит столбец 'date' к datetime
    и поля 'flares.M' и 'flares.X' к числовому типу.
    """
    with filepath.open('r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%Y %m %d', errors='coerce')
    for col in ['flares.M', 'flares.X']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_events_data(filepath: Path) -> pd.DataFrame:
    """
    Загружает данные из файла Events (JSON) и приводит столбец 'date' к datetime.
    Преобразует временные поля ('begin', 'max', 'end') к числовому типу и вычисляет длительность события.
    Затем агрегирует данные по дате: считает количество событий и среднюю длительность.
    """
    with filepath.open('r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%Y %m %d', errors='coerce')

    # Преобразуем временные параметры к числовому виду и вычисляем продолжительность, если возможно.
    for col in ['begin', 'max', 'end']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Вычисляем длительность события как разницу между end и begin (если имеются оба значения)
    df['duration'] = df[['begin', 'end']].apply(lambda row: row['end'] - row['begin']
    if pd.notna(row['begin']) and pd.notna(row['end'])
    else np.nan, axis=1)
    # Группируем по дате: количество событий и средняя длительность
    agg_df = df.groupby('date').agg(
        count_events=('date', 'count'),
        mean_duration=('duration', 'mean')
    ).reset_index()
    return agg_df


def load_srs_data(filepath: Path) -> pd.DataFrame:
    """
    Загружает данные из файла SRS (JSON), приводит столбец 'date' к datetime,
    а числовые поля (например, 'Nmbr', 'Lo', 'Area', 'LL', 'NN') – к числовому типу.
    Затем агрегирует данные по дате (среднее значение для числовых признаков).
    """
    with filepath.open('r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], format='%Y %m %d', errors='coerce')
    for col in ['Nmbr', 'Lo', 'Area', 'LL', 'NN']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    agg_df = df.groupby('date').agg({
        'Nmbr': 'mean',
        'Lo': 'mean',
        'Area': 'mean',
        'LL': 'mean',
        'NN': 'mean'
    }).reset_index()
    # Переименуем столбцы для ясности
    agg_df.rename(columns={
        'Nmbr': 'srs_Nmbr_mean',
        'Lo': 'srs_Lo_mean',
        'Area': 'srs_Area_mean',
        'LL': 'srs_LL_mean',
        'NN': 'srs_NN_mean'
    }, inplace=True)
    return agg_df


def merge_datasets(dsd_df: pd.DataFrame, events_df: pd.DataFrame, srs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет данные DSD, агрегированные данные Events и SRS по полю 'date'.
    Использует левое объединение, сохраняя все записи из DSD.
    """
    merged_df = pd.merge(dsd_df, events_df, on='date', how='left')
    merged_df = pd.merge(merged_df, srs_df, on='date', how='left')
    return merged_df


def compute_correlations(df: pd.DataFrame, target_columns: list) -> dict:
    """
    Вычисляет коэффициенты корреляции Пирсона для каждого целевого столбца (например, 'flares.M' и 'flares.X')
    со всеми другими числовыми признаками в DataFrame.

    Возвращает словарь: { target: { feature: corr_value, ... }, ... }
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = {}
    for target in target_columns:
        corr_dict = {}
        for col in numeric_cols:
            if col == target:
                continue
            subset = df[[target, col]].dropna()
            if subset.empty:
                corr = np.nan
            else:
                corr, _ = pearsonr(subset[target], subset[col])
            corr_dict[col] = corr
        correlations[target] = corr_dict
    return correlations


def plot_scatter_plots(df: pd.DataFrame, target_columns: list, threshold: float = 0.2):
    """
    Для каждого числового признака вычисляет коэффициент корреляции с целевыми столбцами.
    Если абсолютное значение корреляции больше threshold, строит scatter plot.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for target in target_columns:
        for col in numeric_cols:
            if col == target:
                continue
            subset = df[[target, col]].dropna()
            if subset.empty:
                continue
            corr, _ = pearsonr(subset[target], subset[col])
            if abs(corr) >= threshold:
                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=subset, x=col, y=target)
                plt.title(f"{target} vs {col}\nPearson r = {corr:.2f}")
                plt.xlabel(col)
                plt.ylabel(target)
                plt.tight_layout()
                plt.show()


def main():
    # Пути к файлам
    dsd_path = Path("../result_json/dsd.json")
    events_path = Path("../result_json/events.json")
    srs_path = Path("../result_json/srs.json")

    # Загрузка данных
    if not dsd_path.exists():
        print(f"Файл {dsd_path} не найден!")
        return
    dsd_df = load_dsd_data(dsd_path)
    print("DSD данные (первые 5 строк):")
    print(dsd_df.head())

    if not events_path.exists():
        print(f"Файл {events_path} не найден!")
        return
    events_df = load_events_data(events_path)
    print("\nАгрегированные данные Events (первые 5 строк):")
    print(events_df.head())

    if not srs_path.exists():
        print(f"Файл {srs_path} не найден!")
        return
    srs_df = load_srs_data(srs_path)
    print("\nАгрегированные данные SRS (первые 5 строк):")
    print(srs_df.head())

    # Объединяем данные по дате
    merged_df = merge_datasets(dsd_df, events_df, srs_df)
    print("\nОбъединённые данные (первые 5 строк):")
    print(merged_df.head())

    # Определяем целевые столбцы для анализа – вспышки типа M и X
    target_columns = ['flares.M', 'flares.X']

    # Вычисляем корреляции целевых столбцов с другими числовыми признаками
    correlations = compute_correlations(merged_df, target_columns)
    print("\nКорреляции для вспышек типа M и X с другими признаками:")
    for target, corr_dict in correlations.items():
        print(f"\n{target}:")
        for feature, corr in corr_dict.items():
            print(f"  {feature}: {corr:.2f}")

    # Строим scatter plots для пар с корреляцией по модулю >= threshold (0.2)
    plot_scatter_plots(merged_df, target_columns, threshold=0.2)


if __name__ == "__main__":
    main()
