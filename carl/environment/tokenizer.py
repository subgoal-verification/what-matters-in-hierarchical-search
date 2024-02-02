from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor

from carl.environment.training_goal import TrainingGoal


class GameTokenizer(ABC):
    @abstractmethod
    def board_tokenizer(self, board: np.ndarray) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def board_detokenizer(self, sequence_of_tokens: list[int]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def x_y_tokenizer(
        self,
        x: np.ndarray | tuple[np.ndarray, np.ndarray],
        y: np.ndarray | int,
        training_goal: TrainingGoal,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
