from abc import ABC, abstractmethod

import numpy as np
import torch
from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.subgoal_generator import SubgoalGenerator
from torch import Tensor, nn
from transformers import PreTrainedModel


class Policy(ABC):
    @abstractmethod
    def __init__(
        self,
        policy_network: type[nn.Module] | None,
        env: GameEnv,
    ) -> None:
        """
        Initialize the policy network.
        params:
            policy_network: the policy network.
            env: the environment.
        """

        self.policy_network = policy_network
        self.env = env

    @abstractmethod
    def construct_network(self) -> None:
        """
        Construct the networks.
        """

        raise NotImplementedError

    @abstractmethod
    def get_actions(self, state: np.ndarray | str) -> Tensor:
        """
        Get a list of actions for the given state.
        params:
            state: the state.
        return:
            the actions.
        """

        raise NotImplementedError


class TransformerPolicy(Policy):
    def __init__(
        self,
        policy_network: type[PreTrainedModel],
        env: GameEnv,
        path_to_policy_weights: str,
        n_actions: int | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        super().__init__(policy_network, env)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path_to_policy_weights = path_to_policy_weights
        self.n_actions = n_actions
        self.confidence_threshold = confidence_threshold

        assert (n_actions is not None) or (
            confidence_threshold is not None
        ), 'Either n_actions or confidence_threshold must be specified.'

        self.policy: PreTrainedModel | None = None

    def construct_network(self) -> None:
        # We do not put the policy on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.policy = self.policy_network.from_pretrained(self.path_to_policy_weights)
        self.policy.to(self.device)
        # self.policy = torch.compile(self.policy, mode="max-autotune")

    def get_actions(self, state: np.ndarray | str) -> Tensor:
        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(
            x=state, y=0, training_goal=TrainingGoal.POLICY
        )
        encoded_board = encoded_board.to(self.device)

        with torch.no_grad():
            output: Tensor = self.policy(encoded_board).logits

        output = output.squeeze(dim=0)

        if self.confidence_threshold is not None:
            actions_to_return: list[int] = []
            cumulative_prob: float = 0.0
            action_with_prob: list[tuple[int, float]] = list(
                zip(range(len(output)), torch.softmax(output, dim=-1).cpu().numpy())
            )
            action_with_prob.sort(key=lambda x: x[1], reverse=True)

            for action, prob in action_with_prob:
                cumulative_prob += prob
                actions_to_return.append(action)
                if cumulative_prob >= self.confidence_threshold:
                    break
            return torch.tensor(actions_to_return)

        return torch.topk(output, self.n_actions, dim=-1).indices


class TransformerPolicyGeneration(Policy):
    def __init__(
        self,
        policy_network: type[PreTrainedModel],
        env: GameEnv,
        path_to_policy_weights: str,
        subgoal_generation_kwargs: dict[str, int] | None,
    ) -> None:
        super().__init__(policy_network, env)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path_to_policy_weights = path_to_policy_weights
        self.subgoal_generation_kwargs = subgoal_generation_kwargs

        self.policy: PreTrainedModel | None = None

    def construct_network(self) -> None:
        # We do not put the policy generation on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.policy = self.policy_network.from_pretrained(self.path_to_policy_weights)
        self.policy.to(self.device)
        # self.policy = torch.compile(self.policy, mode="max-autotune")

    def get_actions(self, state: np.ndarray | str) -> set[int]:
        max_new_tokens: int = self.subgoal_generation_kwargs['max_new_tokens']
        num_beams: int = self.subgoal_generation_kwargs['num_beams']
        num_return_sequences: int = self.subgoal_generation_kwargs['num_return_sequences']

        encoded_board: torch.Tensor
        encoded_board, _ = self.env.tokenizer.x_y_tokenizer(
            x=state, y=0, training_goal=TrainingGoal.POLICY_GENERATION
        )
        encoded_board = encoded_board.to(self.device)
        with torch.no_grad():
            outputs: list[list[int]] = self.policy.generate(
                encoded_board,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            ).tolist()

        moves: set[int] = set()

        for output in outputs:
            move: int | None = self.env.tokenizer.action_detokenizer(output)
            if move is not None:
                moves.add(move)

        return moves


class ExhaustiveBaselinePolicy(Policy):
    def __init__(
        self,
        n_actions: int,
    ) -> None:
        super().__init__(None, None)
        self.n_actions: int = n_actions

    def construct_network(self) -> None:
        pass

    def get_actions(self, state: np.ndarray) -> Tensor:
        return torch.arange(self.n_actions)


class PolicyGeneratorWrapper(SubgoalGenerator):
    def __init__(
        self,
        policy: Policy,
        env: GameEnv,
    ) -> None:
        super().__init__(None, '', env, None)
        self.policy = policy

    def construct_network(self) -> None:
        self.policy.construct_network()

    def get_subgoals(self, state: np.ndarray) -> list[np.ndarray]:
        actions: Tensor = self.policy.get_actions(state)

        return [self.env.next_state(state, action) for action in actions]
