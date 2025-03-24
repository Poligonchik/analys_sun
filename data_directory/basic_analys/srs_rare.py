import pandas as pd
from pathlib import Path


def print_rarest_values(df: pd.DataFrame, top_n: int = 10):
    """
    Для каждого столбца DataFrame выводит top_n редчайших значений (на основе частот).
    Если в столбце много значений с одинаковой минимальной частотой, будет показано только top_n значений.
    """
    for col in df.columns:
        print(f"\nСтолбец: {col}")
        # Вычисляем частоты значений, включая NaN (если требуется)
        value_counts = df[col].value_counts(dropna=False).sort_values(ascending=True)
        print(value_counts.head(top_n))
        print("-" * 50)


def main_rarest_values():
    # Путь к директории с результатами
    output_directory = Path("../processed_results")
    sunspots_path = output_directory / "sunspots.parquet"

    if sunspots_path.exists():
        sunspots_df = pd.read_parquet(sunspots_path)
        print("=== Самые редкие значения (10 по каждому столбцу) для sunspots ===")
        print_rarest_values(sunspots_df, top_n=10)
    else:
        print("Файл sunspots.parquet не найден.")


if __name__ == "__main__":
    main_rarest_values()
