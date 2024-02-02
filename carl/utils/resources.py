import os
from typing import Any
from uuid import uuid4

import joblib as jl
from loguru import logger

NOT_READY_LABEL = 'not_ready'
stop_signal = '.carl_stop_signal'

def get_latest_file(file_paths):
    if not file_paths:
        return None

    # Get the file with the maximum modification time
    latest_file = max(file_paths, key=os.path.getmtime)
    return latest_file


def dump_resource(resource: Any, label: str):    # type: ignore
    """Dumps a resource to a file"""
    short_uuid = str(uuid4())[:8]
    filename = f'{label}_{short_uuid}.jl'
    tmp_filename = f'{NOT_READY_LABEL}_{filename}'
    jl.dump(resource, tmp_filename)

    # Rename file
    # Note: this trick is for avoiding reading not ready resources
    os.rename(tmp_filename, filename)
    logger.debug(f'Dumped resource {label} to {filename}')


def read_resource_and_delete(label: str, flatten: bool = True) -> Any:
    """Reads a resource from a file and deletes it. It is not thread safe."""

    def match_label(f: str) -> bool:
        return f.startswith(label)

    fs = list(filter(match_label, os.listdir('.')))
    objs = [jl.load(f) for f in fs]
    # Delete files
    for f in fs:
        os.remove(f)
        
    if flatten:
        objs = [obj for sublist in objs for obj in sublist]

    logger.debug(f'Read and deleted {len(objs)} resources with label {label}')
    return objs


def read_resource(label: str) -> Any:
    """Reads a resource from a file."""

    def match_label(f: str) -> bool:
        return f.startswith(label)

    fs = filter(match_label, os.listdir('.'))
    objs = [jl.load(f) for f in fs]
    logger.debug(f'Read {len(objs)} resources with label {label}')
    return objs


def exists_resource(label: str) -> bool:
    """Checks if a resource exists."""

    logger.debug(f'Checking if resource {label} exists')

    def match_label(f: str) -> bool:
        return f.startswith(label)

    fs = list(filter(match_label, os.listdir('.')))

    logger.debug(f'Resource {label} exists: {len(fs) > 0}, {fs}')

    return len(fs) > 0
