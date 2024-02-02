from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import time
from typing import Callable
from loguru import logger
import loguru
import torch
from transformers import PreTrainedModel
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments
from carl.components.metrics import MetricsHF

@dataclass
class TrainingModule:
    trainer_class: type[HFTrainer]
    trainer_args: Callable[..., TrainingArguments]
    metrics_for_component: MetricsHF

class InferenceComponent(ABC):
    @abstractmethod
    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        """
        Returns the networks.
        Dict for nested inference components which consist of multiple networks.
        (such as adaptive subgoal generator).
        """

        raise NotImplementedError
    
    @abstractmethod
    def construct_network(self) -> None:
        """
        Construct the networks.
        """

        raise NotImplementedError
    
    def get_component_training_module(self) -> TrainingModule | dict[str, TrainingModule] | None:
        """
        Returns the training module.
        Dict for nested inference components which consist of multiple networks.
        (such as adaptive subgoal generator).
        """
        return None
    
    def instantiate_network(self, network_fn, weights_path: str):
        """
        Instantiate the networks.
        """
        logger.debug(f'Loading weights from {weights_path}')

        if weights_path is not None and not os.path.exists(weights_path):
            # checking if it exists in the nested directory
            
            fs = os.listdir(weights_path)
            fs = [f for f in fs if f.startswith('checkpoint')]
            assert len(fs) <= 1, f'Found multiple checkpoints in {weights_path}'
            if len(fs) == 1:
                weights_path = os.path.join(weights_path, fs[0])
            elif len(fs) == 0: 
                loguru.logger.info(f'Path {weights_path} does not exist. Trying to find it in the local folder.')
                base_folder = weights_path.split('/')[-1]
                weights_path = base_folder
                logger.info(f'Loading weights from {weights_path}')
        network = None
        for _ in range(5):
            try:
                network = network_fn(weights_path)
                break
            except Exception as e:
                logger.critical(f'Failed to load weights from {weights_path}. Retrying...')
                logger.critical(e)
                time.sleep(10)
        
        if network is None:
            raise RuntimeError(f'Failed to load weights from {weights_path}')    
    
        network.to(self.device)
        network.eval()
        logger.info('Skipping compiling model, Device:', self.device, 'torch.cuda.is_available()', torch.cuda.is_available())
        
        return network
        
    
    def is_trainable(self) -> bool:
        """
        Returns whether the component is trainable.
        """

        return self.get_component_training_module() is not None
    
    