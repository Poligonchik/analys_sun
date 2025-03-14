import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_transform_data(json_file: str) -> pd.DataFrame:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for row in data:
        rec = {}

        # Основные поля
        rec["issued"] = row.get("issued", None)
        rec["product"] = row.get("product", None)
        rec["sgas_number"] = row.get("sgas_number", None)

        # Раздел daily_indices (если отсутствует или равен None, используем пустой словарь)
        daily_indices = row.get("daily_indices") or {}
        rec["ten_cm"] = daily_indices.get("10cm", None)
        rec["ssn"] = daily_indices.get("ssn", None)
        rec["afr_ap"] = daily_indices.get("afr_ap", None)
        rec["xray_background"] = daily_indices.get("xray_background", None)

        # Daily Proton Fluence
        dp_fluence = daily_indices.get("daily_proton_fluence") or {}
        rec["dp_fluence_gt_1_mev"] = dp_fluence.get("gt_1_mev", None)
        rec["dp_fluence_gt_10_mev"] = dp_fluence.get("gt_10_mev", None)

        # Daily Electron Fluence
        de_fluence = daily_indices.get("daily_electron_fluence") or {}
        rec["de_fluence_gt_2_mev"] = de_fluence.get("gt_2_mev", None)

        # K-indices (Boulder и Planetary)
        k_indices = daily_indices.get("k_indices") or {}
        if "boulder" in k_indices:
            for i, val in enumerate(k_indices["boulder"]):
                rec[f"k_boulder_{i}"] = val
        if "planetary" in k_indices:
            for i, val in enumerate(k_indices["planetary"]):
                rec[f"k_planetary_{i}"] = val

        # Энергетические события (считаем их количество)
        energetic_events = row.get("energetic_events", None)
        rec["energetic_events_count"] = len(energetic_events) if isinstance(energetic_events, list) else 0

        # Proton events (строка)
        rec["proton_events"] = row.get("proton_events", None)

        # Geomagnetic summary (строка)
        rec["geomagnetic_activity_summary"] = row.get("geomagnetic_activity_summary", None)

        # Удаляем ключи из daily_indices, если их значения равны None
        daily_keys = ["ten_cm", "ssn", "afr_ap", "xray_background",
                      "dp_fluence_gt_1_mev", "dp_fluence_gt_10_mev", "de_fluence_gt_2_mev"]
        for k in daily_keys:
            if rec.get(k) is None:
                rec.pop(k, None)

        records.append(rec)

    df = pd.DataFrame(records)

    # Приведение потенциально числовых полей к float
    numeric_candidates = [
        "ten_cm", "ssn", "dp_fluence_gt_1_mev", "dp_fluence_gt_10_mev",
        "de_fluence_gt_2_mev", "energetic_events_count"
    ]
    k_boulder_cols = [col for col in df.columns if col.startswith("k_boulder_")]
    k_planetary_cols = [col for col in df.columns if col.startswith("k_planetary_")]
    numeric_candidates.extend(k_boulder_cols)
    numeric_candidates.extend(k_planetary_cols)

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def analyze_missing_values(df: pd.DataFrame):
    """
    Анализ пропусков: сколько NaN в каждом столбце,
    визуализация пропусков в виде простого barplot.
    """
    missing = df.isna().sum().sort_values(ascending=False)
    print("\n=== Анализ пропущенных значений ===")
    print(missing)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=90)
    plt.title("Количество пропусков (NaN) по столбцам")
    plt.tight_layout()
    plt.show()

def analyze_categorical_columns(df: pd.DataFrame):
    """
    Анализ категориальных столбцов: вывод уникальных значений, частот и построение графиков.
    Здесь в примере мы будем считать категориальными поля 'product', 'issued', 'afr_ap', 'xray_background',
    'proton_events' и 'geomagnetic_activity_summary'.
    В реальности лучше вручную выбрать, какие поля считать категориальными.
    """
    cat_cols = ["product", "issued", "afr_ap", "xray_background", "proton_events", "geomagnetic_activity_summary"]
    cat_cols = [c for c in cat_cols if c in df.columns]  # только те, что реально есть в DataFrame

    print("\n=== Анализ категориальных столбцов ===")
    for col in cat_cols:
        print(f"\n--- Поле: {col} ---")
        unique_vals = df[col].dropna().unique()
        print(f"Уникальных значений: {len(unique_vals)}")
        print("Первые 10 уникальных значений:", unique_vals[:10])

        # value_counts
        counts = df[col].value_counts(dropna=False)
        print("Топ-10 по частоте:")
        print(counts.head(10))

        # Построим barplot только если число уникальных не слишком велико (например, < 30)
        if len(unique_vals) < 30:
            plt.figure(figsize=(8, 4))
            # В pandas 2.0+ нужно counts.sort_values(ascending=False)[:30]
            # но учитываем, что у нас уже counts.head(30) – можно совместить
            subset_counts = counts.head(30)  # ограничим 30 категориями
            sns.barplot(x=subset_counts.index.astype(str), y=subset_counts.values)
            plt.title(f"Распределение категорий: {col}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

def analyze_numeric_columns(df: pd.DataFrame):
    """
    Анализ числовых столбцов: describe, гистограммы, boxplot.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\n=== Анализ числовых столбцов ===")
    print("Числовые столбцы:", numeric_cols.tolist())

    # Описательная статистика
    print("\n--- describe() по числовым столбцам ---")
    print(df[numeric_cols].describe())

    for col in numeric_cols:
        print(f"\n--- Анализ поля: {col} ---")
        # Пропусков:
        na_count = df[col].isna().sum()
        print(f"Пропусков: {na_count}")
        # Уникальных значений
        unique_vals = df[col].dropna().unique()
        print(f"Уникальных значений: {len(unique_vals)}")

        # Гистограмма
        plt.figure(figsize=(7, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Распределение: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

        # Boxplot
        plt.figure(figsize=(3, 5))
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.show()

def analyze_correlations(df: pd.DataFrame):
    """
    Анализ корреляций для числовых столбцов:
    - вывод корреляционной матрицы
    - построение heatmap
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    print("\n=== Корреляционная матрица (числовые столбцы) ===")
    print(corr_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def main():
    # 1. Загрузка и трансформация
    df = load_and_transform_data("../processed_results/sgas_all.json")

    # 2. Анализ пропусков
    analyze_missing_values(df)

    # 3. Анализ категориальных столбцов
    analyze_categorical_columns(df)

    # 4. Анализ числовых столбцов
    analyze_numeric_columns(df)

    # 5. Анализ корреляций
    analyze_correlations(df)

if __name__ == "__main__":
    main()
