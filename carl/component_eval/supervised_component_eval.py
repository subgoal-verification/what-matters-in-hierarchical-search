from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import NeptuneLogger
from loguru import logger as loguru_logger

from carl.component_evals.component_eval import ComponentEval
from carl.components.DEPRICATED_torch_wrappers import SupervisedTorchWrapper


class SupervisedComponentEval(ComponentEval):
    def __init__(
        self, eval_data_module: LightningDataModule, required_components: list[str]
    ) -> None:
        super().__init__()

        assert len(required_components) == 1

        self._eval_data_module = eval_data_module
        self.required_components = required_components
        self._eval_data_module.prepare_data()
        self._eval_data_module.setup(stage='test')

    def evaluate(
        self,
        trainers: dict[str, Trainer],
        components: dict[str, LightningModule],
        *args,
        **kwargs,
    ) -> None:

        component = components[self.required_components[0]]
        trainer = trainers[self.required_components[0]]

        if not isinstance(component, SupervisedTorchWrapper):
            raise TypeError(
                f'SupervisedComponentEval works only with components of SupervisedTorchWrapper type. '
                f'Nor are component dicts permissible for this eval. '
                f"The component's type is {type(component)}"
            )

        if not isinstance(trainer, Trainer):
            raise TypeError(
                f'SupervisedComponentEval works only with trainers of Trainer type. '
                f'Nor are trainer dicts permissible for this eval. '
                f"The trainer's type is {type(trainer)}"
            )

        loguru_logger.info('Starting evaluation')
        loguru_logger.info(f'Component is {component}')
        loguru_logger.info(f'Datamodule is {self._eval_data_module}')
        loguru_logger.info(f'Trainer is {trainer}')

        eval_trainer = Trainer(
            accelerator='auto', logger=trainer.logger, callbacks=None, inference_mode=True
        )

        component.component_eval_mode = True
        eval_values = eval_trainer.test(
            model=component, dataloaders=self._eval_data_module.test_dataloader()
        )
        if isinstance(trainer.logger, NeptuneLogger):
            for key, value in eval_values[0].items():
                eval_trainer.logger.experiment[f'{component}_eval/{key}'].log(value)
                trainer.logger.experiment[f'{component}_eval/{key}'].log(value)
        component.component_eval_mode = False
