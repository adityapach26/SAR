import yaml
from typing import Any


class Config:
    """Allow attribute-style access to nested dicts."""
    def __init__(self, mapping: dict[str, Any]):
        for key, value in mapping.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list):
                # Convert list items if they are dicts
                setattr(self, key, [Config(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Config({self.__dict__})"


def load_config(config_path: str) -> Config:
    """
    Load YAML config file and return a Config object with attribute access.
    Example: cfg.train.batch_size
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(data)


if __name__ == "__main__":
    # Simple test: load the config from the same repository
    cfg = load_config("configs/config.yaml")
    print("Config loaded successfully.")
    print(f"Dataset path: '{cfg.dataset.path}'")
    print(f"Train batch size: {cfg.train.batch_size}")
    print(f"Learning rate: {cfg.train.learning_rate}")
    print(f"Number of ensemble members: {cfg.ensemble.num_members}")
    print(f"Detection num classes: {cfg.detection.num_classes}")