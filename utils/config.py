# Simple helper to call the config.yaml file when required
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)