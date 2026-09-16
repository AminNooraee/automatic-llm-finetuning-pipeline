UNIFIED_FIELDS = {"instruction", "input", "output"}


def validate_unified_dataset(rows):
    """Check the normalized rows before they are given to LLaMA-Factory."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must contain at least one example")

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Unified dataset row {index} must be an object")

        if set(row) != UNIFIED_FIELDS:
            raise ValueError(
                f"Unified dataset row {index} must have instruction, input, and output fields"
            )

        for field in ("instruction", "output"):
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Dataset row {index} needs a non-empty {field} string")

        if not isinstance(row["input"], str):
            raise ValueError(f"Dataset row {index} needs an input string")

    return rows
