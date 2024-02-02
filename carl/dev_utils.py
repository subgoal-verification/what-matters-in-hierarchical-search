from carl.algorithms.algorithm import Algorithm
from carl.deployers.grid_search import CarlGrid

def instantiate_algorithm(config_name: str, config_path: str = "experiments", disable_gpu: bool = True, worker_type: str | None = None) -> Algorithm:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    from dotenv import load_dotenv
    load_dotenv('.tokens.env', override=True)
    if disable_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    initialize(config_path=config_path)
    config = compose(config_name=config_name)
    from hydra.utils import instantiate

    if worker_type is None:
        algo = instantiate(config.algorithm)
        return algo
    
    worker2config = CarlGrid(config).iter_workers(config)
    worker_config = worker2config[worker_type]
    algo = instantiate(worker_config.algorithm)
    return algo
    