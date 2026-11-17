import json
from typing import List

from .json_validator import Prompts, Functions


def prompts_loader(path: str) -> List[Prompts]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the JSON file containing prompts.

    Returns:
        List of validated Prompts objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the data does not match the Prompts schema.
    """
    with open(path, "r") as file:
        data = json.load(file)
    return [Prompts(**item) for item in data]


def function_loader(path: str) -> List[Functions]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the JSON file containing function definitions.

    Returns:
        List of validated Functions objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the data does not match the Functions schema.
    """
    with open(path, "r") as file:
        data = json.load(file)

    return [Functions(**item) for item in data]
