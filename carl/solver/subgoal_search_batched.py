from collections.abc import Callable
from typing import Optional
from loguru import logger

import numpy as np

from carl.inference_components.subgoal_generator import SubgoalGenerator
from carl.inference_components.validator import Validator
from carl.inference_components.value import Value
from carl.solver.planners import BestFSPlanner, Planner
from carl.solver.utils import SearchTreeNode, ValidationResult
import time
from dataclasses import dataclass
from transformers.integrations import NeptuneCallback

@dataclass
class SearchTreeBatch:
    max_len: int
    planners: list[Planner]
    nodes_visited: np.ndarray
    nodes_valid: np.ndarray
    nodes_unreachable: np.ndarray
    global_instance_ids: np.ndarray # indexes of instances in the whole dataset, used for debug or tracking particular instances
    solving_node: list[SearchTreeNode | None]
    search_info: list[dict[str, str | int | None]]
    current_len: int = 0
    
    def __post_init__(self):
        self.current_len = len(self.planners)
        assert len(self.nodes_visited) == self.current_len
        assert len(self.nodes_valid) == self.current_len
        assert len(self.nodes_unreachable) == self.current_len
        assert len(self.global_instance_ids) == self.current_len
        assert len(self.solving_node) == self.current_len
        assert len(self.search_info) == self.current_len
        assert self.current_len <= self.max_len

    def __len__(self):
        return self.current_len
        
    def fill_batch(self, initial_states: list[tuple[int, np.ndarray]], planner_class):
        assert self.max_len >= len(initial_states) + len(self)
        
        for initial_state in initial_states:
            i, instance = initial_state
            self.planners.append(planner_class(instance))
            self.solving_node.append(None)
            self.search_info.append({'finished_reason': None})
            self.nodes_visited = np.append(self.nodes_visited, 0)
            self.nodes_valid = np.append(self.nodes_valid, 0)
            self.nodes_unreachable = np.append(self.nodes_unreachable, 0)
            self.global_instance_ids = np.append(self.global_instance_ids, i)
            self.current_len += 1

    def drop_finished(self) -> 'SearchTreeBatch':
        # drop finished inplace
        finished_idxs = [i for i, n in enumerate(self.search_info) if n['finished_reason'] is not None]
        finished_mask = np.zeros((len(self),), dtype=bool)
        finished_mask[finished_idxs] = True
        
        planners_finished = [self.planners[i] for i in finished_idxs]
        solving_node_finished = [self.solving_node[i] for i in finished_idxs]
        search_info_finished = [self.search_info[i] for i in finished_idxs]
        nodes_visited_finished = self.nodes_visited[finished_idxs]
        nodes_valid_finished = self.nodes_valid[finished_idxs]
        nodes_unreachable_finished = self.nodes_unreachable[finished_idxs]
        global_instance_ids_finished = self.global_instance_ids[finished_idxs]
        
        finished_batch = SearchTreeBatch(
            planners=planners_finished,
            solving_node=solving_node_finished,
            search_info=search_info_finished,
            nodes_visited=nodes_visited_finished,
            nodes_valid=nodes_valid_finished,
            nodes_unreachable=nodes_unreachable_finished,
            global_instance_ids=global_instance_ids_finished,
            current_len=len(finished_idxs),
            max_len=self.max_len,
        )
        
        # update own batch
        self.planners = [p for i, p in enumerate(self.planners) if not finished_mask[i]]
        self.solving_node = [n for i, n in enumerate(self.solving_node) if not finished_mask[i]]
        self.search_info = [n for i, n in enumerate(self.search_info) if not finished_mask[i]]
        self.nodes_visited = self.nodes_visited[~finished_mask]
        self.nodes_valid = self.nodes_valid[~finished_mask]
        self.nodes_unreachable = self.nodes_unreachable[~finished_mask]
        self.global_instance_ids = self.global_instance_ids[~finished_mask]
        self.current_len = len(self.planners)
        
        return finished_batch
    

class SubgoalSearchSolver:
    def __init__(
        self,
        max_nodes: int,
        planner_class: Callable[[np.ndarray], Planner],
        subgoal_generator: SubgoalGenerator,
        validator: Validator,
        value_function: Value,
        inference_batch_size: int,
    ) -> None:        
        self.max_nodes = max_nodes
        self.planner_class = planner_class
        self.subgoal_generator = subgoal_generator
        self.validator = validator
        self.value_function = value_function
        self.inference_batch_size = inference_batch_size
    
    def construct_networks(self) -> None:
        self.subgoal_generator.construct_network()
        self.validator.construct_network()
        self.value_function.construct_network()
                  
    def _get_nodes_for_current_iteration(self, batch: SearchTreeBatch) -> list[SearchTreeNode | None]:
        """Returns nodes that will be expanded in the current iteration.
        """
        return [planner.get() for planner in batch.planners]
    
    def _find_expandable_idxs_and_update_batch_info(self, batch: SearchTreeBatch, current_nodes: list[SearchTreeNode | None]) -> np.ndarray:
        """Returns indexes of instances that are finished because of 'nothing_to_expand' communicate.
        """
        nothing_to_expand_idxs = [i for i, n in enumerate(current_nodes) if n is None]
        if len(nothing_to_expand_idxs) > 0:
            # There is nothing more to expand.
            for i in nothing_to_expand_idxs:
                batch.search_info[i]['finished_reason'] = 'nothing_to_expand'
        idxs_to_expand = set(range(len(batch.planners))) - set(nothing_to_expand_idxs)
        return np.array(sorted(list(idxs_to_expand)), dtype=int)
    
    def _filter_unseen_subgoals(self, nodes_to_expand: list[SearchTreeNode | None], planners_of_nodes_to_expand: list[Planner], subgoals: np.ndarray) -> tuple[list[SearchTreeNode | None], np.ndarray, np.ndarray]:
        """Returns nodes, subgoals and mask that are not seen by the planner.
        """
        if len(subgoals) == 0:
            return [], np.array([]), np.array([])
        
        unseen_subgoals = np.array([not planner.is_seen(subgoal) for planner, subgoal in zip(planners_of_nodes_to_expand, list(subgoals))], dtype=bool) # (|nodes_to_expand|,)
        # nodes_to_validate = [n for n, unseen in zip(nodes_to_expand, unseen_subgoals) if unseen]
        nodes_to_validate = [n for i, n in enumerate(nodes_to_expand) if unseen_subgoals[i]]
        subgoals_to_validate = subgoals[unseen_subgoals]
        return nodes_to_validate, subgoals_to_validate, unseen_subgoals
        
        
    def solve_one_iteration(self, batch: SearchTreeBatch, neptune_callback) -> SearchTreeBatch:
        """Executes a single iteration of the search tree on batch of instances.
        """
        time0 = time.time()
        # Note: all indexes which we keep here, are relative to the whole batch.
        current_nodes = self._get_nodes_for_current_iteration(batch)

        # Take only nodes that will be expanded. Those unexpandable are finished and marked as 'nothing_to_expand'.
        nodes_idxs_to_expand = self._find_expandable_idxs_and_update_batch_info(batch, current_nodes)
        # logger.info(f'nodes_idxs_to_expand: {len(nodes_idxs_to_expand)}')
        nodes_to_expand = [current_nodes[i] for i in nodes_idxs_to_expand]
        planners_of_nodes_to_expand = [batch.planners[i] for i in nodes_idxs_to_expand]

        if len(nodes_to_expand) == 0:
            # logger.info('Nothing to expand. for the whole batch')
            return batch
        
        time1 = time.time()
        neptune_callback.run['solve_one_iteration/get_nodes_to_expand_time'].append(time1 - time0)

        # We get subgoals with zero padding to MAX_SUBGOALS
        time0 = time.time()
        subgoals: np.ndarray = self.subgoal_generator.get_subgoals(nodes_to_expand, neptune_callback) # (|nodes_to_expand|, MAX_SUBGOALS, *STATE_SHAPE)
        time1 = time.time()
        neptune_callback.run['solve_one_iteration/get_subgoals_time'].append(time1 - time0)

        # Time to validate proposed subgoals.
        # We do it a layer at a time because subgoals are sorted by value, so we
        # omit the false-positive solved boards in case of exceeding the budget on earlier subgoals.
        time0 = time.time()
        max_subgoals = subgoals.shape[1]
        for subgoal_idx in range(max_subgoals):
            subgoals_ith_propositions = subgoals[:, subgoal_idx, ...] # (|nodes_to_expand|, *STATE_SHAPE)
            # logger.info(f'subgoals_ith_propositions shape: {subgoals_ith_propositions.shape}')
            nodes_to_validate, subgoals_to_validate, unseen_mask = self._filter_unseen_subgoals(nodes_to_expand, planners_of_nodes_to_expand, subgoals_ith_propositions)
            
            assert len(nodes_to_validate) == len(subgoals_to_validate)

            if len(nodes_to_validate) == 0:
                continue # All subgoals are already seen.

            
            validations: list[ValidationResult] = self.validator.is_valid(nodes_to_validate, subgoals_to_validate)
            values_of_validate = self.value_function.get_value(subgoals_to_validate)
            
            for i, validation in enumerate(validations):
                ori_idx = nodes_idxs_to_expand[unseen_mask][i]
                
                if validation.is_valid:
                    subgoal_node: SearchTreeNode = SearchTreeNode(
                        state=validation.achieved_state,
                        value=values_of_validate[i],
                        low_level_path=validation.path,
                        parent_node=nodes_to_validate[i],
                    )
                    batch.planners[ori_idx].add(subgoal_node)
                    
                    if validation.is_solved:
                        batch.solving_node[ori_idx] = subgoal_node
                        batch.search_info[ori_idx]['finished_reason'] = 'solved'
                        
                    batch.nodes_valid[ori_idx] += 1
                else:
                    batch.nodes_unreachable[ori_idx] += 1
                    
                batch.nodes_visited[ori_idx] += validation.nodes_visited
                
        for i in range(len(batch.planners)):
            if batch.nodes_visited[i] > self.max_nodes:
                batch.search_info[i]['finished_reason'] = 'budget_exceeded'
                batch.solving_node[i] = None

        time1 = time.time()
        neptune_callback.run['solve_one_iteration/validate_subgoals_time'].append(time1 - time0)
        
        return batch
    
    def solve(self, initial_states: list[np.ndarray], neptune_callback: Optional[NeptuneCallback] = None) -> list[tuple[dict, dict]]:
        iter_label = 'batched_iteration_monitor'
        
        if neptune_callback is not None:
            neptune_callback.run[f'{iter_label}/boards_to_solve'].append(len(initial_states))
        
        
        batch_size = min(self.inference_batch_size, len(initial_states))
        # logger.info(f'batch_size: {batch_size}')
        last_unused_idx = 0
        
        batch = SearchTreeBatch(self.inference_batch_size, [], np.array([]), np.array([]), np.array([]), np.array([]), [], [])
        assert len(batch) == 0
        batch.fill_batch(list(zip([j for j in range(0, batch_size)], initial_states[0:batch_size])), self.planner_class)
        assert len(batch) == batch_size

        last_unused_idx = batch_size
        solve_rate = 0
        somehow_finished_instances_count = 0
        j = 0
        results = []

        while len(batch) > 0:
            j += 1
            time0 = time.time()
            batch = self.solve_one_iteration(batch, neptune_callback)
            time1 = time.time()
            neptune_callback.run[f'{iter_label}/solve_one_iteration_time'].append(time1 - time0)
            
            time0 = time.time()
            old_batch_size = len(batch)
            finished_batch = batch.drop_finished()
            assert len(batch) == old_batch_size - len(finished_batch)
            
            for local_batch_idx, global_instance_idx in enumerate(finished_batch.global_instance_ids):
                result = finished_batch.planners[local_batch_idx].get_solution_data(
                                                    finished_batch.solving_node[local_batch_idx],
                                                    finished_batch.search_info[local_batch_idx])
                
                # Search Info Update
                result[1].update(
                    {
                        'nodes_visited': finished_batch.nodes_visited[local_batch_idx],
                        'nodes_valid': finished_batch.nodes_valid[local_batch_idx],
                        'nodes_unreachable': finished_batch.nodes_unreachable[local_batch_idx],
                    }
                )

                
                results.append((global_instance_idx, result))
                
                # solve rate update
                somehow_finished_instances_count += 1
                if result[0]['solved']:
                    solve_rate = (solve_rate * (somehow_finished_instances_count - 1) + 1) / somehow_finished_instances_count
                else:
                    solve_rate = (solve_rate * (somehow_finished_instances_count - 1)) / somehow_finished_instances_count
                
            if last_unused_idx < len(initial_states):
                # how many can we fit
                assert len(batch) <= self.inference_batch_size
                assert last_unused_idx < len(initial_states)
                can_be_added_to_batch = min(self.inference_batch_size - len(batch), len(initial_states) - last_unused_idx)
                
                batch.fill_batch(
                    list(zip(
                        [j for j in range(last_unused_idx, last_unused_idx + can_be_added_to_batch)],
                        initial_states[last_unused_idx:last_unused_idx + can_be_added_to_batch]
                )), self.planner_class)
            
                last_unused_idx = last_unused_idx + can_be_added_to_batch
            time1 = time.time()
            neptune_callback.run[f'{iter_label}/batch_update_time'].append(time1 - time0)
            # update logs
            if neptune_callback is not None:
                neptune_callback.run[f'{iter_label}/iteration_finished'].append(j)
                if somehow_finished_instances_count > 0:
                    neptune_callback.run[f'{iter_label}/somehow_finished_instances_count'].append(somehow_finished_instances_count)
                    
        assert len(results) == len(initial_states)
        
        # sort by idx and return results
        results = sorted(results, key=lambda x: x[0])
        results = [r[1] for r in results]
        return results
