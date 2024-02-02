from collections.abc import Callable
import sys

import numpy as np

from carl.inference_components.subgoal_generator import SubgoalGenerator
from carl.inference_components.validator import Validator
from carl.inference_components.value import Value
from carl.solver.planners import Planner
from carl.solver.utils import SearchTreeNode, ValidationResult
import time

class Solver:
    def __init__(
        self,
        max_nodes: int,
        planner_class: Callable[[np.ndarray], Planner],
        subgoal_generator: SubgoalGenerator,
        validator: Validator,
        value_function: Value,
    ) -> None:

        self.max_nodes = max_nodes

        self.planner_class = planner_class
        self.planner: Planner | None = None
        self.subgoal_generator = subgoal_generator
        self.validator = validator
        self.value_function = value_function

    def construct_networks(self) -> None:
        self.subgoal_generator.construct_network()
        self.validator.construct_network()
        self.value_function.construct_network()

    def solve(self, initial_state: np.ndarray) -> tuple[dict, dict]:
        self.planner = self.planner_class(initial_state)

        recursion_limit = 2**31 - 1
        curr_recursion_limit = sys.getrecursionlimit()

        if curr_recursion_limit < recursion_limit:
            print(f'Rising recursion limit for worker from {curr_recursion_limit} to {recursion_limit}.')
            sys.setrecursionlimit(recursion_limit)

        nodes_visited: int = 0
        nodes_valid: int = 0
        nodes_unreachable: int = 0
        solving_node: SearchTreeNode | None = None

        search_info: dict[str, str | int | None] = {
            'finished_reason': 'budget_exceeded',
        }
        # !! ASSUMPTION: we are using adaptive generator here. If you want to use non-adaptive one, please proceed with only 1 generator in adagenerator.
        ks = self.subgoal_generator.generator_k_list
        subgoals_reachable_count_per_k: dict[int, int] = {k: 0 for k in ks}
        subgoals_unreachable_count_per_k: dict[int, int] = {k: 0 for k in ks}

        while nodes_visited < self.max_nodes and solving_node is None:
            current_node: SearchTreeNode | None = self.planner.get()
            if current_node is None:
                # There is nothing more to expand.
                search_info['finished_reason'] = 'nothing_to_expand'
                break

            subgoals: list[np.ndarray] = self.subgoal_generator.get_subgoals([current_node]) # (nodes, subgoals, *state_dim)
            assert subgoals.shape[0] == 1
            subgoals = subgoals[0]

            for subgoal in subgoals:
                if self.planner.is_seen(subgoal):
                    continue

                validation: ValidationResult = self.validator.is_valid([current_node], [subgoal])
                assert len(validation) == 1
                validation = validation[0]

                nodes_visited += validation.nodes_visited

                if not validation.is_valid:
                    nodes_unreachable += 1
                    subgoals_unreachable_count_per_k[current_node.next_expand_with_k_generator] += 1
                    continue
            

                valid_subgoal: np.ndarray = validation.achieved_state

                nodes_valid += 1
                subgoals_reachable_count_per_k[current_node.next_expand_with_k_generator] += 1
                
                value: float = self.value_function.get_value([valid_subgoal])[0]

                subgoal_node: SearchTreeNode = SearchTreeNode(
                    state=valid_subgoal,
                    value=value,
                    low_level_path=validation.path,
                    parent_node=current_node,
                )
                self.planner.add(subgoal_node)

                if validation.is_solved:
                    solving_node = subgoal_node
                    search_info['finished_reason'] = 'solved'
                    break

        search_info.update(
            {
                'nodes_visited': nodes_visited,
                'nodes_valid': nodes_valid,
                'nodes_unreachable': nodes_unreachable,
            }
        )
        
        for k in ks:
            search_info[f'subgoals_reachable_count_per_k/{k}'] = subgoals_reachable_count_per_k[k]
            search_info[f'subgoals_unreachable_count_per_k/{k}'] = subgoals_unreachable_count_per_k[k]
            
            if subgoals_reachable_count_per_k[k] + subgoals_unreachable_count_per_k[k] == 0:
                search_info[f'subgoals_reachable_rate/{k}'] = 0
            else:
                rate = subgoals_reachable_count_per_k[k] / (subgoals_reachable_count_per_k[k] + subgoals_unreachable_count_per_k[k])
                search_info[f'subgoals_reachable_rate/{k}'] = rate

        # The computational budget is over.
        return self.planner.get_solution_data(solving_node, search_info)
