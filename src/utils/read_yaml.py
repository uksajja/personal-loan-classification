import yaml
from pathlib import Path
from typing import Union

def read_yaml(path: Union[str, Path]) -> dict:
    """
    Reads a YAML configuration file and returns it as a dictionary.

    Parameters:
    ----------
    path : str or Path
        Path to the YAML file.

    Returns:
    -------
    dict
        Parsed YAML content as a dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)
    

"""
lru_cache means the file is only actually read once per unique path argument, no matter how many places call load_config("config.yaml")
 — but there's no class, no __new__ override, no instance-check boilerplate. 
 This is strictly simpler than a singleton class and gets you the same practical benefit (read once, reuse everywhere in the process)
"""