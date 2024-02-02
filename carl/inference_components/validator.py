from abc import ABC, abstractmethod
from typing import Any, Callable
from loguru import logger

import numpy as np
from torch import Tensor
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.inference_components.component import InferenceComponent
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.solver.utils import SearchTreeNode, ValidationResult


class Validator(InferenceComponent):
    """
    General Validator class. Used to check if a subgoal is achievable from a given state.
    """

    @abstractmethod
    def __init__(self, env_class: Callable[..., GameEnv], cllp: ConditionalLowLevelPolicy, inference_batch_size: int) -> None:
        self.env_class = env_class
        self.cllp = cllp
        self.inference_batch_size = inference_batch_size
        self.envs = [env_class() for _ in range(inference_batch_size)]

    @abstractmethod
    def construct_network(self) -> None:
        """
        Constructs the networks of the validator.
        """

        raise NotImplementedError()

    @abstractmethod
    def is_valid(self, nodes: list[SearchTreeNode], subgoals: list[np.ndarray]) -> list[ValidationResult]:
        """
        Checks if a subgoal is achievable from a given state.
        param state: the state.
        param subgoal: the subgoal.
        return: the validation result. The validation result is a tuple of the form (is_valid, is_solved, path, nodes_visited, achieved_state).

        is_valid: True if the subgoal is achievable from the given state, False otherwise.
        is_solved: True if the subgoal is achievable from the given state and the subgoal is a goal state, False otherwise.
        path: the path to the subgoal from the given state.
        nodes_visited: the number of nodes visited during the validation, i.e. the number of nodes that were expanded during the validation.
        achieved_state: the state that was achieved during the validation.
        """

        raise NotImplementedError()


class BasicValidator(Validator):
    """
    Validator class for TransformerConditionalLowLevelPolicy. To check if a subgoal is achievable from a given state,
    we use the conditional low level policy to generate a moves from the given state to the subgoal.
    We then use the environment to simulate the execution of the actions and check if the subgoal is achieved.
    """

    def __init__(self, env_class, cllp: ConditionalLowLevelPolicy, inference_batch_size: int) -> None:
        super().__init__(env_class, cllp, inference_batch_size)

    def construct_network(self) -> None:
        self.cllp.construct_network()
        
    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.cllp.get_network()

    def is_valid(self, nodes: list[SearchTreeNode], subgoals: list[np.ndarray]) -> list[ValidationResult]:
        assert len(nodes) == len(subgoals)
        assert len(self.envs) >= len(nodes)
        
        step: int = 0
        action_path: list[list[int]] = [[] for _ in range(len(nodes))] # (|nodes|, |actions|)
        is_solved: np.ndarray = np.zeros(len(nodes), dtype=bool)
        is_valid: np.ndarray = np.zeros(len(nodes), dtype=bool)
        states = [node.state for node in nodes]
        budget_for_achieving_subgoal: np.ndarray = np.array([node.next_expand_with_k_generator + 2 for node in nodes])
        step_finished: np.ndarray = np.ones(len(nodes), dtype=bool) * budget_for_achieving_subgoal
 
        for i, env in zip(range(len(nodes)), self.envs):
            if i >= len(nodes):
                break
            if self.envs[i].is_solved(subgoals[i]):
                is_solved[i] = True
        
        for state, env in zip(states, self.envs):
            env.restore_full_state_from_np_array_version(state)
            
        finished_idxs: set[int] = set()
        
        while step < np.max(budget_for_achieving_subgoal):
            distribution_over_actions: Tensor = self.cllp.get_action(states, subgoals) # (|nodes|, |actions|)
            assert len(distribution_over_actions.shape) == 2
            assert distribution_over_actions.shape[0] == len(nodes)
            actions: np.ndarray = np.argmax(distribution_over_actions.cpu().numpy(), axis=1) 
            for i, env in zip(range(len(nodes)), self.envs):
                if i in finished_idxs:
                    continue
                action_path[i].append(actions[i])
                states[i] = env.next_state(states[i], actions[i])

                if np.array_equal(states[i], subgoals[i]):
                    is_valid[i] = True
                    is_solved[i] = env.is_solved(subgoals[i])
                    step_finished[i] = step
                    finished_idxs.add(i)
            step += 1
                        
        return [ValidationResult(is_valid[i], is_solved[i], action_path[i], step_finished[i], subgoals[i]) for i in range(len(nodes))]

class DummyValidator:
    """
    Dummy validator class. Always accepts the subgoal. Used with baseline policy.
    """
    def __init__(self, env: GameEnv) -> None:
        self.env = env

    def construct_network(self) -> None:
        pass

    def is_valid(self, state: np.ndarray, subgoal: np.ndarray) -> ValidationResult:
        return ValidationResult(True, self.env.is_solved(subgoal), [], 1, subgoal)
