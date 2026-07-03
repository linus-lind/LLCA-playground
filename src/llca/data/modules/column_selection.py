ALL_COLUMNS = "ALL"


def is_all_columns(value: object) -> bool:
    return isinstance(value, str) and value == ALL_COLUMNS
