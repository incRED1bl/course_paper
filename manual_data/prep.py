import csv
from pathlib import Path
from collections import Counter


def parse_int_or_none(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def make_stable_single_key(row: dict[str, str], row_index: int) -> tuple[str, str, str, int]:
    dataset = row.get("dataset", "")
    if row.get("file_id"):
        return ("single", dataset, row["file_id"], row_index)
    if row.get("filename"):
        return ("single", dataset, row["filename"], row_index)
    return ("single", dataset, "", row_index)


def drop_last_dataset_tail(rows: list[dict[str, str]], start_row: int = 3389) -> tuple[list[dict[str, str]], int, str]:
    dataset_order: list[str] = []
    seen: set[str] = set()

    for row in rows:
        dataset = row.get("dataset", "")
        if dataset not in seen:
            seen.add(dataset)
            dataset_order.append(dataset)

    if not dataset_order:
        return rows, 0, ""

    last_dataset = dataset_order[-1]
    filtered_rows: list[dict[str, str]] = []
    dropped = 0

    for source_row, row in enumerate(rows, start=2):
        if row.get("dataset", "") == last_dataset and source_row >= start_row:
            dropped += 1
            continue
        filtered_rows.append(row)

    return filtered_rows, dropped, last_dataset


def reindex_csv(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise ValueError("Input CSV has no header row.")
    if "filename" not in fieldnames or "patient_id" not in fieldnames:
        raise ValueError("CSV must contain 'filename' and 'patient_id' columns.")

    if "dataset" not in fieldnames:
        raise ValueError("CSV must contain a 'dataset' column.")

    rows, dropped_tail_count, last_dataset = drop_last_dataset_tail(rows, start_row=3389)

    old_group_counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        current_pid = parse_int_or_none(row.get("patient_id", ""))
        if current_pid is not None and current_pid != -1:
            dataset = row.get("dataset", "")
            old_group_counts[(dataset, current_pid)] += 1

    new_id_by_key: dict[tuple, int] = {}
    next_patient_id = 1
    grouped_rows = 0

    for row_index, row in enumerate(rows):
        current_pid = parse_int_or_none(row.get("patient_id", ""))
        dataset = row.get("dataset", "")

        if current_pid is not None and current_pid != -1 and old_group_counts[(dataset, current_pid)] > 1:
            key: tuple = ("group", dataset, current_pid)
            grouped_rows += 1
        else:
            key = make_stable_single_key(row, row_index)

        if key not in new_id_by_key:
            new_id_by_key[key] = next_patient_id
            next_patient_id += 1

        row["patient_id"] = str(new_id_by_key[key])

    # Keep output visually stable: patient_id order will be monotonic in the CSV.
    rows.sort(
        key=lambda row: (
            parse_int_or_none(row.get("patient_id", "")) or 0,
            row.get("dataset", ""),
            row.get("filename", ""),
        )
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Rows processed: {len(rows)}")
    print(f"Dropped rows from last dataset '{last_dataset}' starting at row 3389: {dropped_tail_count}")
    print(f"Rows kept in multi-record groups: {grouped_rows}")
    print(f"Final unique patient_id count: {len(new_id_by_key)}")
    print(f"Next available id after update: {next_patient_id}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "combined_respiratory_features.csv"
    output_path = base_dir / "combined_respiratory_features_reindexed.csv"
    reindex_csv(input_path, output_path)