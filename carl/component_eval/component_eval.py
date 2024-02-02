from abc import ABC, abstractmethod

from lightning import LightningModule, Trainer


class ComponentEval(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def evaluate(
        self, trainers: dict[str, Trainer], components: dict[str, LightningModule], *args, **kwargs
    ) -> None:
        raise NotImplementedError
