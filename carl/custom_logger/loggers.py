import os
from abc import ABC, abstractmethod
from typing import Any

import neptune
from transformers.integrations import NeptuneCallback


class CaRLLogger(ABC):
    """
    Abstract class for custom logger.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def return_logger(self) -> Any:
        raise NotImplementedError


class NeptuneCaRLLogger(CaRLLogger):
    """
    Custom logger for Neptune.
    """

    def __init__(
        self,
        name: str,
        description: str,
        project: str,
        tags,
        log_parameters: bool,
        api_token: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self.project = project
        self.tags = tags
        if api_token is None:
            self.api_token = os.getenv('NEPTUNE_API_TOKEN', neptune.ANONYMOUS_API_TOKEN)
            print(f'Retrieved Neptune API token from environment: {self.api_token}')
        else:
            self.api_token = api_token
        print(f'Using Neptune API token: {self.api_token}')
        self.log_parameters = log_parameters

        self.run = neptune.init_run(
            name=self.name,
            description=self.description,
            tags=self.tags,
            project=self.project,
            api_token=self.api_token,
        )

    def return_logger(self) -> NeptuneCallback:
        """
        Returns the Neptune logger.
        """

        return NeptuneCallback(run=self.run, log_parameters=self.log_parameters)
