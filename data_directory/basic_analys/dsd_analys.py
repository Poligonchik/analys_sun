import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import pearsonr


def load_data(json_path: Path) -> pd.DataFrame:
    """
    Загружает данные из JSON-файла и преобразует их в DataFrame.
    Если столбцы 'flares' и 'optical_flares' содержат вложенные словари,
    разворачивает их в отдельные столбцы.
    """
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Преобразуем столбец даты в datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Разворачиваем вложенные словари
    if 'flares' in df.columns:
        flares_df = pd.json_normalize(df['flares'])
        flares_df.columns = [f'flares_{col}' for col in flares_df.columns]
        df = pd.concat([df.drop(columns=['flares']), flares_df], axis=1)
    if 'optical_flares' in df.columns:
        optical_df = pd.json_normalize(df['optical_flares'])
        optical_df.columns = [f'optical_{col}' for col in optical_df.columns]
        df = pd.concat([df.drop(columns=['optical_flares']), optical_df], axis=1)
    return df


def plot_individual_scatter_plots(df: pd.DataFrame, max_nan_ratio=0.8):
    """
    Строит отдельные scatter plots для каждого попарного сочетания числовых признаков,
    у которых доля пропущенных значений не превышает max_nan_ratio.
    На графике отображается коэффициент корреляции.
    """
    total_rows = len(df)
    nan_ratio = df.isnull().sum() / total_rows

    # Оставляем столбцы, где доля пропусков не превышает max_nan_ratio
    valid_cols = [col for col in df.columns if nan_ratio[col] <= max_nan_ratio]
    # Выбираем числовые столбцы
    num_cols = df[valid_cols].select_dtypes(include=['float64', 'int64']).columns.tolist()
    print("Числовые столбцы для анализа:", num_cols)

    # Для каждого сочетания двух числовых столбцов
    for col1, col2 in itertools.combinations(num_cols, 2):
        # Убираем пропуски по этим двум столбцам
        plot_df = df[[col1, col2]].dropna()
        if plot_df.empty:
            continue

        # Вычисляем коэффициент корреляции
        corr_coef, _ = pearsonr(plot_df[col1], plot_df[col2])

        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=plot_df, x=col1, y=col2)
        plt.title(f"{col1} vs {col2}\nPearson r = {corr_coef:.2f}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.show()


def main():
    json_path = Path('../processed_results/DSD_all.json')
    if not json_path.exists():
        print(f"Файл {json_path} не найден!")
        return

    df = load_data(json_path)
    print("Первые 5 строк DataFrame:")
    print(df.head())
    # Выводим общую информацию (при необходимости)
    print("\nИнформация о DataFrame:")
    print(df.info())

    # Строим отдельные графики для попарных зависимостей числовых признаков
    plot_individual_scatter_plots(df, max_nan_ratio=0.8)


if __name__ == "__main__":
    main()
