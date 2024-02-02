from abc import ABC, abstractmethod

from lightning.pytorch.loggers.logger import Logger


class Algorithm(ABC):
    def __init__(self, logger: Logger | None = None) -> None:
        self.logger = logger

    @abstractmethod
    def run(self) -> None:
        pass
