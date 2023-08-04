import os
from box.exceptions import BoxValueError
import yaml
from imageCaptioning import logger
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

def read_yaml(yaml_path):
    try:
        with open(yaml_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'YAML file {yaml_path} loaded successfully')
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError('file is empty')
    except Exception as e:
        raise e
    
def create_directories(path_to_directories, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'created directory at: {path}')

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=3)
    logger.info(f'json file saved at {path}')

def save_bin(data, path):
    joblib.dump(value=data, filename=path)
    logger.info(f'binary data saved at {path}')

def load_bin(path):
    data = joblib.load(path)
    logger.info(f'binary file loaded from {path}')
    return data

def get_size(path):
    return f'{round(os.path.getsize(path)/1024)} KB'

def decode_image(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()

def encode_image(image):
    with open(image, 'rb') as f:
        return base64.b64encode(f.read())
    