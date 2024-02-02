from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningDataModule, LightningModule, Trainer
from loguru import logger as loguru_logger
from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel
from transformers import Trainer as HFTrainer

from carl.algorithms.algorithm import Algorithm
from carl.component_evals.component_eval import ComponentEval
from carl.components.metrics import MetricsHF
from carl.custom_logger.loggers import CaRLLogger
from carl.dataloader.game_data_module import GameDataModule


class TrainSupervised(Algorithm):
    """
    Train a model using the Lightning Trainer.
    """

    def __init__(
        self,
        component: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        custom_logger: CaRLLogger | None = None,
        component_evals: list[ComponentEval] | None = None,
    ) -> None:
        super().__init__()
        self.component = component
        self.datamodule = datamodule
        self.trainer = trainer
        self.component_evals = component_evals
        self.custom_logger = custom_logger

    def run(self) -> None:
        loguru_logger.info('Starting training')
        self.trainer.fit(self.component, self.datamodule)

        for component_eval in self.component_evals:
            component_eval.evaluate(
                trainers={'supervised_component': self.trainer},
                components={'supervised_component': self.component},
                goals='',
                initial_state='',
            )
        loguru_logger.info('Training finished')


class TrainSupervisedHF(Algorithm):
    """
    Train a model using the HuggingFace Trainer.
    """

    def __init__(
        self,
        trainer: Callable[..., HFTrainer],
        model: Callable[..., PreTrainedModel] | type[PreTrainedModel],
        datamodule: GameDataModule,
        custom_logger: CaRLLogger | None = None,
        custom_metrics: MetricsHF | None = None,
        config: PretrainedConfig | None = None,
        do_finetune: bool = False,
        path_to_model_weights: str | None = None,
    ) -> None:
        super().__init__()

        self.datamodule = datamodule
        self.custom_logger = custom_logger
        self.custom_metrics = custom_metrics

        self.model_to_train: PreTrainedModel | None = None
        self.ready_trainer: HFTrainer | None = None

        if not do_finetune:
            assert config is not None, 'config must be provided if do_finetune is False'
            self.config = config
            self.model_to_train = model(config=self.config)
        else:
            assert (
                path_to_model_weights is not None
            ), 'path_to_model_weights must be provided if do_finetune is True'
            self.model_to_train = model.from_pretrained(path_to_model_weights)

        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        compute_metrics: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]] | None

        if self.custom_metrics is not None:
            preprocess_logits_for_metrics, compute_metrics = self.custom_metrics.get_metrics()
        else:
            preprocess_logits_for_metrics = None
            compute_metrics = None

        self.ready_trainer = trainer(
            model=self.model_to_train,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            data_collator=self.data_collector,
        )

        if self.custom_logger is not None:
            logger: Any = self.custom_logger.return_logger()
            self.ready_trainer.add_callback(logger)

    @staticmethod
    def data_collector(xy: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            'input_ids': torch.stack([x[0] for x in xy]),
            'labels': torch.stack([y[1] for y in xy]),
        }

    def run(self) -> None:
        self.datamodule.prepare_data()
        self.datamodule.setup('fit')
        train_dataset: Dataset = self.datamodule.get_train_dataset()
        validation_dataset: Dataset = self.datamodule.get_val_dataset()

        self.ready_trainer.train_dataset = train_dataset
        # In HF, eval_dataset is used for validation. Meaning of "eval" here is different from ours.
        self.ready_trainer.eval_dataset = validation_dataset
        self.ready_trainer.data_collator = self.data_collector

        self.ready_trainer.train()
