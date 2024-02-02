import pickle
from loguru import logger
logger.info('Setting pickle.HIGHEST_PROTOCOL to 5')
pickle.HIGHEST_PROTOCOL = 5
import os
import pathlib
import sys

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from carl.solve_instances.result_loggers import SubgoalSearchResultLogger, NeptuneCaRLLogger

# For neptune etc
DOTENV_PATH = './.tokens.env'
from transformers.integrations import NeptuneCallback
from neptune.utils import stringify_unsupported


def run(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))
    load_dotenv(DOTENV_PATH, override=True)

    algorithm = hydra.utils.instantiate(config.algorithm)
    logger.info(f'Registered algorithm: {algorithm}')
    logger.add(sink=lambda msg: print(msg, end=''), level='INFO')

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info('Setting recursion limit to 2**31 - 1')
    sys.setrecursionlimit(2**31 - 1)

    algorithm.run()


FILE_PATH = pathlib.Path(os.path.realpath(__file__))
CONFIG_ROOT = FILE_PATH.parent.parent / 'experiments'


# pylint: disable=missing-function-docstring
@hydra.main(version_base=None, config_path=str(CONFIG_ROOT))
def main(config: DictConfig) -> None:
    run(config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
