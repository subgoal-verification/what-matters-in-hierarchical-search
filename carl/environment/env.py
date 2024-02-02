from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor

from carl.environment.tokenizer import GameTokenizer


class GameEnv(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def tokenizer(self) -> GameTokenizer:
        raise NotImplementedError

    @abstractmethod
    def detect_action(self, board_before: np.ndarray, board_after: np.ndarray) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def distribution_to_action(distribution: Tensor) -> int:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_solved(self, board: np.ndarray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def show_state(
        self, state: np.ndarray, title: str | None = None, file_name: str | None = None
    ) -> None:
        pass

    @abstractmethod
    def show_many_states(self, states: list[np.ndarray], titles: list[str]):
        pass

    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        raise NotImplementedError
