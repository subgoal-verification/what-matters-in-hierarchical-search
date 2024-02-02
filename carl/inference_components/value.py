from abc import ABC, abstractmethod
import os

import numpy as np
import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.component import InferenceComponent, TrainingModule


class Value(InferenceComponent):
    @abstractmethod
    def __init__(
        self,
        value_network_class: type[nn.Module],
        path_to_value_network_weights: str,
        env: GameEnv,
        type_of_evaluation: str | None = None,
    ) -> None:
        """
        Initialize the value network.
        params:
            value_network: the value network.
            env: the environment.
        """

        if os.path.exists(path_to_value_network_weights) and any([f.startswith('checkpoint') for f in os.listdir(path_to_value_network_weights)]):
            f = [f for f in os.listdir(path_to_generator_weights) if f.startswith('checkpoint')][-1]
            path_to_generator_weights = os.path.join(path_to_generator_weights, f)

        self.value_network_class = value_network_class
        self.path_to_value_network_weights = path_to_value_network_weights
        self.env = env
        self.type_of_evaluation = type_of_evaluation

    @abstractmethod
    def get_value(self, state: np.ndarray) -> float:
        """
        Get the value of the given state.
        params:
            state: the state.
        return:
            the value, for example, distance to the goal.
        """

        raise NotImplementedError
    
class TransformerValue(Value):
    def __init__(
        self,
        value_network_class: type[PreTrainedModel],
        path_to_value_network_weights: str,
        env: GameEnv,
        type_of_evaluation: str = 'classification',
        training_module: TrainingModule | None = None,
    ) -> None:
        """
        Type of evaluation can be either 'classification' or 'regression'. It corresponds to the type of the head of
        the model (type of training).
        """

        super().__init__(value_network_class, path_to_value_network_weights, env, type_of_evaluation)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', self.device, 'torch.cuda.is_available()', torch.cuda.is_available())

        self.value_network: PreTrainedModel | None = None
        self.training_module = training_module

        assert self.type_of_evaluation in ['regression', 'classification']

    def get_component_training_module(self) -> TrainingModule | dict[str, TrainingModule] | None:
        return self.training_module

    def construct_network(self) -> None:
        # We do not put the value on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.value_network = self.instantiate_network(self.value_network_class, self.path_to_value_network_weights)
        
    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.value_network

    def get_value(self, states: list[np.ndarray]) -> list[float]:
        encoded_boards: torch.Tensor
                
        encoded_boards = torch.concat([self.env.tokenizer.x_y_tokenizer(
             x=state, y=0, training_goal=TrainingGoal.VALUE
        )[0] for state in states])
        encoded_boards = encoded_boards.to(self.device)
        with torch.no_grad():
            output: Tensor = self.value_network(encoded_boards).logits # logits is the output of the model. (BS, 1)

        if self.type_of_evaluation == 'classification':
            distribution: Tensor = torch.softmax(output, dim=1)
            distances: list[float] = [
                self.expected_value(distribution[i]) for i in range(len(distribution))
            ]
        else:
            distances = output.flatten().tolist()
            distances = [-d for d in distances]

        return distances

    @staticmethod
    def expected_value(state_distribution: Tensor) -> float:
        """
        Get the expected value of the given state distribution.
        :param state_distribution: the state distribution.
        :return: the expected value.
        """

        state_distribution = torch.flatten(state_distribution)
        return torch.sum(
            torch.tensor([i * state_distribution[i] for i in range(len(state_distribution))])
        ).item()
