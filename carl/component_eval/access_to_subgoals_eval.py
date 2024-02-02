import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import NeptuneLogger

from carl.component_evals.component_eval import ComponentEval
from carl.environment.env import GameEnv
from carl.environment.instance_generator import InstanceGenerator
from carl.inference_components.cllp_utils import verify_cllp_reaches_subgoals_from_initial_state
from carl.inference_components.conditional_low_level_policy import (
    TransformerConditionalLowLevelPolicy,
)
from carl.inference_components.subgoal_generator import TransformerSubgoalGenerator


class AccessToSubgoalsEval(ComponentEval):
    def __init__(
        self,
        env: GameEnv,
        instance_generator: InstanceGenerator,
        max_radius: int,
        add_first_batch_to_node_computations: bool,
    ) -> None:
        super().__init__()
        self.required_components = ['cllp', 'generator']

        self.env = env
        self.instance_generator = instance_generator
        self.max_radius = max_radius
        self.add_first_batch_to_node_computations = add_first_batch_to_node_computations

    def evaluate(
        self, trainers: dict[str, Trainer], components: dict[str, LightningModule], *args, **kwargs
    ) -> None:

        goals = kwargs['goals']
        assert type(goals) == torch.Tensor

        initial_state = self.instance_generator.reset_dataloader()

        generator = TransformerSubgoalGenerator(
            generator=components['generator'].net,
            env=kwargs['env'],
            subgoal_generation_kwargs=kwargs['subgoal_generation_kwargs'],
        )

        goals = generator.get_subgoals(initial_state)

        cllp = TransformerConditionalLowLevelPolicy(
            conditional_low_level_policy=components['cllp'].net, env=kwargs['env']
        )

        def env_creation_function():
            return self.env

        outputs = verify_cllp_reaches_subgoals_from_initial_state(
            cllp=cllp,
            goals=goals,
            initial_state=initial_state,
            env_creation_fn=env_creation_function,
            max_radius=self.max_radius,
            add_first_batch_to_node_computations=self.add_first_batch_to_node_computations,
        )

        logger = trainers['cllp'].logger

        if isinstance(logger, NeptuneLogger):
            logger.experiment['component_eval/access_to_subgoals_rate'].log(outputs.success_rate)
            logger.experiment['component_eval/access_to_subgoals_calls'].log(outputs.calls)
            logger.experiment['component_eval/access_to_subgoals_cllp_samples_in_calls'].log(
                outputs.cllp_samples_in_calls
            )
