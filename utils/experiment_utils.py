import os
import re
from datetime import datetime


def _normalize_name_part(value):
    text = str(value).strip().replace(" ", "")
    for old, new in (("/", "-"), ("\\", "-"), (":", "-")):
        text = text.replace(old, new)
    return text if text else "unknown"


def format_input_size(fixed_size):
    if fixed_size is None or len(fixed_size) == 0:
        return "unknown"

    if len(fixed_size) == 1:
        return str(fixed_size[0])

    height, width = fixed_size[0], fixed_size[1]
    if height == width:
        return str(height)
    return f"{height}x{width}"


def format_learning_rate(lr):
    return format(float(lr), "g")


def infer_dataset_name(data_path):
    if not data_path:
        return "unknown_dataset"
    normalized = os.path.normpath(os.path.abspath(os.path.expanduser(data_path)))
    return _normalize_name_part(os.path.basename(normalized))


def build_experiment_name(method_name, fixed_size, data_path, lr, batch_size, run_mode):
    base_name = "_".join([
        _normalize_name_part(method_name),
        format_input_size(fixed_size),
        infer_dataset_name(data_path),
        format_learning_rate(lr),
        str(batch_size),
        _normalize_name_part(run_mode),
    ])
    return f"{format_timestamp_for_path()}_{base_name}"


def infer_experiment_name_from_weights(weights_path):
    if not weights_path:
        return None

    weights_path = os.path.abspath(os.path.expanduser(weights_path))
    parent_name = os.path.basename(os.path.dirname(weights_path))
    if parent_name:
        normalized = _normalize_name_part(parent_name)
        if has_timestamp_prefix(normalized):
            return normalized
        return f"{format_timestamp_for_path()}_{normalized}"

    stem = os.path.splitext(os.path.basename(weights_path))[0]
    normalized = _normalize_name_part(stem)
    if has_timestamp_prefix(normalized):
        return normalized
    return f"{format_timestamp_for_path()}_{normalized}"


def resolve_output_subdir(root_dir, subdir_name):
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    return os.path.join(root_dir, _normalize_name_part(subdir_name))


def write_text_block(file_path, lines):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def format_timestamp_for_path(dt=None):
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def has_timestamp_prefix(name):
    return bool(re.match(r"^\d{8}_\d{6}_", str(name)))
