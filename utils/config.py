import yaml
from pathlib import Path


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_config_file_path():
    current_path = Path(__file__)
    return str(current_path.resolve().parents[1] / 'conf.d' / 'config.yaml')


def get_config():
    config_path = get_config_file_path()
    config = load_config(config_path)
    return config


def get_openai_key():
    config = get_config()
    openai_key = config['openai']['key']
    return openai_key
