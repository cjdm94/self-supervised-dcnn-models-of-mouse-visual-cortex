from pathlib import Path
import json
import pandas as pd


def find_project_root(current_file: Path, marker='.git'):
    """
    Find the project root directory by looking for a marker file (default is '.git').
    This function traverses up the directory tree from the current file until it finds the marker file.

    Args:
        current_file: The file from which to start searching for the project root.
        marker: The name of the marker file to look for (default is '.git').
    Returns:
        The Path object representing the project root directory.
    """
    for parent in current_file.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Project root not found. Marker '{marker}' not found in any parent directory.")


def write_json_file(data, filepath: str):
    """
    Write data to a JSON file.

    Args:
        data: The data to write to the file.
        filename (str): The name of the file to write to.
    """
    with open(file=filepath, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def read_json_file_as_dataframe(filepath: str):
    """
    Read a JSON file and convert it to a pandas DataFrame.
    Args:
        filepath (str): The path to the JSON file.
    Returns:
        pd.DataFrame: A DataFrame containing the data from the JSON file.
    """
    with open(file=filepath, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        return pd.DataFrame(data)
