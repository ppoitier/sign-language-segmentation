from pathlib import Path
import yaml

from sls.config import Config


def deep_merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def load_config_data(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)

    if 'from' in config:
        parent_path = (config_path.parent / Path(config['from'])).resolve()
        parent_config = load_config_data(parent_path)
        deep_merge(config, parent_config)
        return parent_config

    return config


def load_config(config_path) -> Config:
    config_data = load_config_data(config_path)
    return Config.model_validate(config_data)


if __name__ == '__main__':
    config = load_config_data('/home/ppoitier/Documents/dev/lsfb/sign-langage-segmentation/config/child_config.yaml')
    print(config)
