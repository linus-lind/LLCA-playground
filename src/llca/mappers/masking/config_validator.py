from omegaconf import DictConfig, ListConfig

from llca.mappers.config_validation import check_fields, register_validator
from llca.mappers.modules.config_field import ConfigField

_DATE = "date"
_PAIR_FIELDS = [ConfigField("start", "str"), ConfigField("end", "str")]


def _entry_dtype(entry: object, default: str) -> str:
    if isinstance(entry, DictConfig):
        return str(entry.get("dtype", default))
    return default


def _date_columns(spec: DictConfig, time_role: str | None) -> set[str]:
    """Collect canonical columns explicitly parsed as dates in one dataset spec."""
    dates: set[str] = set()
    index = spec.get("index")
    if isinstance(index, DictConfig):
        for canonical, entry in index.items():
            default = _DATE if str(canonical) == time_role else "Int64"
            if _entry_dtype(entry, default) == _DATE:
                dates.add(str(canonical))
    for group in ("columns", "auxiliary"):
        mapping = spec.get(group)
        if isinstance(mapping, DictConfig):
            for canonical, entry in mapping.items():
                if _entry_dtype(entry, "numeric") == _DATE:
                    dates.add(str(canonical))
    return dates


def _primary_dataset(cfg: DictConfig) -> tuple[str | None, DictConfig | None]:
    """Resolve the feature dataset whose rows define masking activity spells."""
    model = cfg.get("model")
    inputs = model.get("inputs") if isinstance(model, DictConfig) else None
    features = inputs.get("features") if isinstance(inputs, DictConfig) else None
    primary = str(features) if features is not None else None

    data = cfg.get("data")
    datasets = data.get("datasets") if isinstance(data, DictConfig) else None
    spec = (
        datasets.get(primary) if isinstance(datasets, DictConfig) and primary is not None else None
    )
    return primary, spec if isinstance(spec, DictConfig) else None


def _time_role(cfg: DictConfig) -> str | None:
    data = cfg.get("data")
    index = data.get("index") if isinstance(data, DictConfig) else None
    time = index.get("time") if isinstance(index, DictConfig) else None
    return str(time) if time is not None else None


@register_validator
def _validate_masking(cfg: DictConfig) -> list[str]:
    """Validate activity-spell pairs against date-typed columns of the primary dataset."""
    masking = cfg.get("masking")
    if not isinstance(masking, DictConfig):
        return []
    subgroups = masking.get("subgroups")
    if subgroups is None:
        return []
    if not isinstance(subgroups, ListConfig):
        return ["masking.subgroups must be a list of {start, end} pairs"]

    primary, spec = _primary_dataset(cfg)
    date_columns = _date_columns(spec, _time_role(cfg)) if spec is not None else set()

    errors: list[str] = []
    for position, pair in enumerate(subgroups):
        prefix = f"masking.subgroups[{position}]"
        if not isinstance(pair, DictConfig):
            errors.append(f"{prefix} must be a mapping with 'start' and 'end'")
            continue
        field_errors = check_fields(pair, prefix, _PAIR_FIELDS)
        if field_errors:
            errors.extend(field_errors)
            continue
        if spec is None:
            errors.append(
                f"{prefix} cannot be validated: model.inputs.features dataset "
                f"'{primary}' is not a configured dataset"
            )
            continue
        for role in ("start", "end"):
            column = str(pair.get(role))
            if column not in date_columns:
                errors.append(
                    f"{prefix}.{role} '{column}' must be a date-typed column of dataset "
                    f"'{primary}' (declare it with dtype: date)"
                )
    return errors
