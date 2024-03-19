import yaml

def get_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.full_load(file)
