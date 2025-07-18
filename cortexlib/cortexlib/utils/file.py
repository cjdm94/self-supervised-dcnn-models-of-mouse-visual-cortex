from pathlib import Path
from cortexlib.utils.logging import Logger
import json
import pandas as pd
from enum import Enum
import torch
import warnings


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
        if isinstance(data, dict) and all(not isinstance(v, (list, dict)) for v in data.values()):
            # Convert dict of scalars to list of dicts (rows)
            return pd.DataFrame([{'layer': k, 'value': v} for k, v in data.items()])
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


def load_model_features(model: Model, mouse_id: str):
    """
    Loads saved model features for a given model and mouse_id.

    Parameters:
    - model (Model): enum or object with `.value` as the model name (e.g. 'simclr')
    - mouse_id (str): e.g. 'm03_d4'
    - base_dir (Path, optional): directory containing '_model_features'. Defaults to parent of cwd.

    Returns:
    - dict with keys 'features' and 'labels'

    Raises:
    - FileNotFoundError if the file does not exist
    """
    logger = Logger()

    model_name = model.value
    cwd = Path.cwd()
    parent_dir = cwd.parent
    features_dir = parent_dir / "_model_features"

    filename = f"{model_name}_features_mouse_{mouse_id}.pt"
    filepath = features_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Model features not found at {filepath}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        logger.info(f"Loading model features from {filepath}")
        return torch.load(filepath)


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


def load_filtered_neural_data(mouse_id: str):
    """
    Loads neural data for the given mouse_id from a .pt file.

    Parameters:
    - mouse_id (str): e.g. 'm03_d4'
    - base_dir (Path): optional path to _neural_data directory

    Returns:
    - Loaded PyTorch object
    """
    logger = Logger()

    cwd = Path.cwd()
    parent_dir = cwd.parent
    features_dir = parent_dir / "_neural_data"
    filename = f"neural_data_mouse_{mouse_id}.pt"
    filepath = features_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Filtered neural data not found at {filepath}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        logger.info(f"Loading filtered neural data from {filepath}")
        return torch.load(filepath)


def get_mouse_id():
    """
    Returns the mouse ID (e.g. 'm03_d4') by searching parent directories.
    Assumes the directory is named like 'mouse_<id>'.
    """
    current_path = Path().resolve()
    for parent in current_path.parents:
        if parent.name.startswith('mouse_'):
            return parent.name.removeprefix('mouse_')
    return None  # If no matching directory is found


def get_model_target() -> str:
    """
    Extracts the model target (e.g. 'simclr_neural') from the current file path.
    """
    parent_dir = Path().resolve().name
    if "_" in parent_dir and any(model in parent_dir for model in ["simclr", "vgg19"]):
        return parent_dir.split("_", 1)[1]  # e.g., "simclr_neural" → "neural"
    return None
