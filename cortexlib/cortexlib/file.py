from pathlib import Path


def find_project_root(current_file: Path, marker='.git'):
    for parent in current_file.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Project root not found. Marker '{marker}' not found in any parent directory.")
