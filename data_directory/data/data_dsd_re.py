import json
from pathlib import Path


def shift_flares(record: dict) -> dict:
    """
    Переносит значение 'background' в 'flares.C', а flares сдвигает:
    C -> M, M -> X, X -> S, старое S удаляется.
    """
    if "background" in record and "flares" in record:
        bg_val = record["background"]
        orig = record["flares"] or {}
        record["flares"] = {
            "C": bg_val,
            "M": orig.get("C", 0),
            "X": orig.get("M", 0),
            "S": orig.get("X", 0)
        }
        del record["background"]
    return record


def shift_optical_flares(record: dict) -> dict:
    """
    Заполняет optical_flares на основе сдвинутых flares:
    optical["1"] = flares["X"]
    optical["2"] = flares["S"]
    optical["3"] = flares["S"]
    """
    flares = record.get("flares", {})
    record["optical_flares"] = {
        "1": flares.get("X", None),
        "2": flares.get("S", None),
        "3": flares.get("S", None)
    }
    return record


def transform_and_save(json_path: Path):
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    transformed = []
    for rec in data:
        rec = shift_flares(rec)
        rec = shift_optical_flares(rec)
        transformed.append(rec)

    with json_path.open('w', encoding='utf-8') as f:
        json.dump(transformed, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    json_file = Path("../processed_results/DSD_all.json")
    transform_and_save(json_file)
