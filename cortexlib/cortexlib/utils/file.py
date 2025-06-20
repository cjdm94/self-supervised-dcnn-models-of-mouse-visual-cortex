from cortexlib.utils.logging import Logger
from pathlib import Path
import json
import pandas as pd
from enum import Enum
import torch


class Model(Enum):
    GABOR = "gabor"
    SIMCLR = "simclr"
    VGG19 = "vgg19"


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


def save_model_features(model: Model, mouse_id: str, features, labels):
    logger = Logger()

    model_name = model.value
    cwd = Path.cwd()
    parent_dir = cwd.parent
    features_dir = parent_dir / "_model_features"
    features_dir.mkdir(exist_ok=True)

    filename = f"{model_name}_features_mouse_{mouse_id}.pt"
    filepath = features_dir / filename

    if filepath.exists():
        logger.info(f"Skipping save, file already exists at {filepath}")
    else:
        logger.info(f"Saving model features to {filepath}")
        torch.save({'features': features, 'labels': labels}, filepath)
        logger.success("Model features saved")


def save_filtered_neural_data(mouse_id, neural_responses, neural_responses_mean):
    logger = Logger()

    cwd = Path.cwd()
    parent_dir = cwd.parent
    features_dir = parent_dir / "_neural_data"
    features_dir.mkdir(exist_ok=True)

    filename = f"neural_data_mouse_{mouse_id}.pt"
    filepath = features_dir / filename

    if filepath.exists():
        logger.info(f"Skipping save, file already exists at {filepath}")
    else:
        logger.info(f"Saving neural data to {filepath}")
        torch.save({
            'neural_responses': torch.from_numpy(neural_responses),
            'neural_responses_mean': torch.from_numpy(neural_responses_mean),
        }, filepath)
        logger.success("Neural data saved")
