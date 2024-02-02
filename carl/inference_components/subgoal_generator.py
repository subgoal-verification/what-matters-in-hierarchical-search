from abc import ABC, abstractmethod
import os
import time
from loguru import logger
import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.inference_components.component import InferenceComponent, TrainingModule
from carl.solver.utils import SearchTreeNode


class SubgoalGenerator(InferenceComponent):
    @abstractmethod
    def __init__(
        self,
        generator_class: type[nn.Module] | None,
        path_to_generator_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int] | None = None,
        training_module: TrainingModule | None = None,
    ) -> None:
        """
        Initialize the sub-goals generator.
        params:
            generator: the generator.
            env: the environment.
            subgoal_generation_kwargs: the subgoal generation kwargs.
        """
        self.generator_class = generator_class
        self.path_to_generator_weights = path_to_generator_weights
        self.env = env
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.training_module = training_module

    @abstractmethod
    def get_subgoals(self, nodes: list[SearchTreeNode]) -> np.ndarray: # (BatchSize, NumSubgoals, *BoardShape)
        """
        Generate sub-goals for the given state.
        params:
            node: node with the state to expand.
        return:
            the subgoals.
        """

        raise NotImplementedError
    
    def get_component_training_module(self) -> TrainingModule | dict[str, TrainingModule] | None:
        """
        Returns the training module.
        Dict for nested inference components which consist of multiple networks.
        (such as adaptive subgoal generator).
        """
        return self.training_module


class TransformerSubgoalGenerator(SubgoalGenerator):
    def __init__(
        self,
        generator_class: type[PreTrainedModel],
        path_to_generator_weights: str,
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int] | None,
        training_module: TrainingModule | None = None,
    ) -> None:
        super().__init__(generator_class, path_to_generator_weights, env, subgoal_generation_kwargs, training_module)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.subgoal_generator_network: PreTrainedModel | None = None

    def construct_network(self) -> None:
        # We do not put the generator on the eval mode, because "from_pretrained" does it for us.
        # See: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
        self.subgoal_generator_network = self.instantiate_network(self.generator_class, self.path_to_generator_weights)

    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return self.subgoal_generator_network # Already initialized in construct_network

    def get_subgoals(self, nodes: list[SearchTreeNode], neptune_callback = None) -> np.ndarray: # (BatchSize, NumSubgoals, *BoardShape)
        """
        Generate sub-goals for the given state.
        :param state: the state.
        :return: the subgoals.
        """
        """
        Generate sub-goals for the given state.
        :param state: the state.
        :return: the subgoals.
        """
        max_new_tokens: int = self.subgoal_generation_kwargs['max_new_tokens']
        num_beams: int = self.subgoal_generation_kwargs['num_beams']
        num_return_sequences: int = self.subgoal_generation_kwargs['num_return_sequences']

        time0 = time.time()
        encoded_boards: list[torch.Tensor]
        encoded_boards = torch.concat([
            self.env.tokenizer.x_y_tokenizer(
                x=node.state, y=node.state, training_goal=TrainingGoal.GENERATOR
            )[0] for node in nodes])
                
        encoded_boards = encoded_boards.to(self.device)
        time1 = time.time()
        # logger.info(f'encoded_boards shape: {encoded_boards.shape}')
        if neptune_callback is not None:
            neptune_callback.run['encode_subgoals_time'].append(time1 - time0)
        with torch.no_grad():
            # https://huggingface.co/docs/transformers/perf_infer_gpu_one
            outputs: torch.Tensor = self.subgoal_generator_network.generate(
                encoded_boards,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
            ) # (BS*num_return_sequences, MaxSeqLen)
            # logger.info(f'subgoal-network outputs shape: {outputs.shape}')
        
        assert outputs.shape[0] == len(nodes) * num_return_sequences
        time0 = time.time()
        subgoals: list[list[np.ndarray]] = []
        for i in range(len(nodes)):
            subgoals.append([])
            for j in range(num_return_sequences):
                tokens = outputs[i * num_return_sequences + j]
                list_of_tokens: list[int] = tokens.cpu().numpy().tolist()
                current_subgoal: np.ndarray | None = self.env.tokenizer.board_detokenizer(list_of_tokens)
                if current_subgoal is not None:
                    subgoals[i].append(current_subgoal)
                else:
                    # Add the original state as a subgoal.
                    # This is done to avoid having an empty subgoal. (which is not allowed during batched solve)
                    subgoals[i].append(nodes[i].state)
        time1 = time.time()
        if neptune_callback is not None:
            neptune_callback.run['decode_subgoals_time'].append(time1 - time0)
                    
        subgoals = [np.stack(subgoal) for subgoal in subgoals] # (array[BS]: NumSubgoals, *BoardShape)
        # logger.info(f'shapes: {[subgoal.shape for subgoal in subgoals]}')
        subgoals = np.stack(subgoals) # (BS, NumSubgoals, *BoardShape)
        # logger.info(f'subgoals shape: {subgoals.shape}')
        return subgoals

class AdaptiveSubgoalGenerator(InferenceComponent):
    def __init__(
        self,
        generator_k_list: list[int],
        subgoal_generator_class: type[SubgoalGenerator],
        paths_to_generator_weights: list[str],
        env: GameEnv,
        subgoal_generation_kwargs: dict[str, int] | None,
    ) -> None:
        print(f'generator_k_list: {generator_k_list}, paths: {paths_to_generator_weights}')
        self.env = env
        self.subgoal_generation_kwargs = subgoal_generation_kwargs
        self.subgoal_generator_class = subgoal_generator_class
        self.generator_k_list = generator_k_list
        self.paths_to_generator_weights = paths_to_generator_weights
        self.subgoal_generators = {idx: subgoal_generator_class(env=env, 
                                                                subgoal_generation_kwargs=subgoal_generation_kwargs,
                                                                path_to_generator_weights=generator_ckpt_path)
                                   for idx, generator_ckpt_path in zip(generator_k_list, paths_to_generator_weights)}
    
    def get_component_training_module(self) -> TrainingModule | dict[str, TrainingModule] | None:
        """
        Returns the training module.
        Dict for nested inference components which consist of multiple networks.
        (such as adaptive subgoal generator).
        """
        return {
            k: subgoal_generator.get_component_training_module() for k, subgoal_generator in self.subgoal_generators.items()
        }

    def construct_network(self) -> None:
        for subgoal_generator in self.subgoal_generators.values():
            subgoal_generator.construct_network()
            
    def get_network(self) -> PreTrainedModel | dict[str, PreTrainedModel]:
        return {k: subgoal_generator.get_network() for k, subgoal_generator in self.subgoal_generators.items()}

    def get_subgoals(self, nodes: list[np.ndarray], neptune_callback = None) -> list[np.ndarray]: # (BatchSize, NumSubgoals, *BoardShape)
        """
        Generate sub-goals for the given state.
        :param state: the state.
        :return: the subgoals.
        """
        ks = set([node.next_expand_with_k_generator for node in nodes])
        
        nodes_idxs_by_k = {k: [] for k in ks}
        for idx, node in enumerate(nodes):
            nodes_idxs_by_k[node.next_expand_with_k_generator].append(idx)
            
        for k in ks:
            nodes_idxs_by_k[k] = np.array(nodes_idxs_by_k[k])
            
        board_shape = nodes[0].state.shape
        num_subgoals = self.subgoal_generation_kwargs['num_return_sequences']
        subgoals = np.zeros((len(nodes), num_subgoals, *board_shape))
            
        for k, nodes_idxs in nodes_idxs_by_k.items():
            subgoals[nodes_idxs] = self.subgoal_generators[k].get_subgoals([nodes[idx] for idx in nodes_idxs], neptune_callback)
            
        return subgoals
