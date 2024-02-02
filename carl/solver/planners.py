from abc import abstractmethod

import numpy as np
from loguru import logger

from carl.solver.utils import SafePriorityQueue, SearchTreeNode, get_solving_path_data

GeneratorIdx = int

from carl.solver.utils import SearchTreeNode

def prune_search_tree_from_solving_node(solving_node: SearchTreeNode):
    if len(solving_node.children) > 0:
        logger.error('Solving node should not have children')
        raise ValueError('Solving node should not have children')
        
    current_node = solving_node
    while current_node.parent_node is not None:
        current_node.parent_node.children = [current_node]
        current_node = current_node.parent_node

def dfs(tree_node: SearchTreeNode):
    for child_node in tree_node.children:
        dfs(child_node)
    print(tree_node.state)
    
def get_root_from_solving_node(solving_node: SearchTreeNode):
    current_node = solving_node
    while current_node.parent_node is not None:
        current_node = current_node.parent_node
    return current_node
    
def high_level_tree_size(tree_node: SearchTreeNode):
    size = 1
    for child_node in tree_node.children:
        size += high_level_tree_size(child_node)
    return size

def low_level_tree_size(tree_node: SearchTreeNode):
    # Includes also intermediate low level states
    size = 1
    for child_node in tree_node.children:
        size += low_level_tree_size(child_node)
    
    if tree_node.low_level_path is not None:
        size += len(tree_node.low_level_path)
    
    return size

def search_tree_stats(tree_node: SearchTreeNode):
    return f'''
    High level tree size: {high_level_tree_size(tree_node)}
    Low level tree size: {low_level_tree_size(tree_node)}
    '''
        
class Planner:
    """
    General Planner class.

    Manages the search tree, selects consecutive nodes to expand and returns
    the final solution once the problem is solved.
    """

    @abstractmethod
    def __init__(self, root_state: np.ndarray) -> None:
        self.root_state = root_state

    @abstractmethod
    def add(self, node: SearchTreeNode) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get(self) -> SearchTreeNode | None:
        raise NotImplementedError()

    @abstractmethod
    def is_seen(self, state: np.ndarray) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_solution_data(
        self, solving_node: SearchTreeNode | None, search_info: dict
    ) -> tuple[dict, dict]:
        raise NotImplementedError()


class BestFSPlanner(Planner):
    """Basic planner that always selects the node with extreme value function."""

    def __init__(self, root_state: np.ndarray, k: int) -> None:
        super().__init__(root_state)
        self.k = k
        self.seen_states: set[
            tuple[
                np.ndarray,
            ]
        ] = set()
        
        self.root_node = SearchTreeNode(root_state, [], None, None, k)

        self.nodes_queue: SafePriorityQueue = SafePriorityQueue()
        self.nodes_queue.put(data=self.root_node, key=0)

    def add(self, node: SearchTreeNode) -> None:
        self.seen_states.add(tuple(node.state.flatten()))
        node.next_expand_with_k_generator = self.k
        self.nodes_queue.put(data=node, key=-node.value)

    def get(self) -> SearchTreeNode | None:
        if self.nodes_queue.empty():
            return None

        return self.nodes_queue.get()

    def is_seen(self, state: np.ndarray) -> bool:
        return tuple(state.flatten()) in self.seen_states

    def get_solution_data(
        self, solving_node: SearchTreeNode | None, search_info: dict
    ) -> tuple[dict, dict]:
        search_info['subgoals_visited'] = len(self.seen_states)
        search_info['search_tree'] = self.root_node
        search_info['solving_node'] = solving_node

        if solving_node is None:
            return {'solved': False}, search_info

        subgoal_path: list[np.ndarray]
        action_path: list[int]
        subgoal_path, action_path, subgoal_distance_path, _ = get_solving_path_data(
            solving_node, include_state_path=False
        )

        solution: dict[str, bool | list[np.ndarray]] = {
            'solved': True,
            'subgoal_path': subgoal_path,
            'action_path': action_path,
            'subgoal_distance_path': subgoal_distance_path,
        }

        return solution, search_info


class AdasubsPlanner(Planner):
    def __init__(self, root_state: np.ndarray, generators_k_list: list[int], prune_search_trees: bool = False) -> None:
        super().__init__(root_state)
        self.generators_k_list = generators_k_list
        self.seen_states: set[
            tuple[
                np.ndarray,
            ]
        ] = set()

        root_key = (0, 0)

        self.nodes_queue: SafePriorityQueue = SafePriorityQueue()
        self.root_node = SearchTreeNode(root_state, 0, None, None, None)
        
        self.subgoals_added = {k: 0 for k in self.generators_k_list}
        self.subgoals_selected_for_expansion = {k: 0 for k in self.generators_k_list}
        self.prune_search_trees = prune_search_trees

        for k in self.generators_k_list:
            root_key = (
                -k,
                0,
            )   # Note: 0 is the maximum value in priority queue since this queue is min-heap
            self.nodes_queue.put(data=SearchTreeNode(root_state, 0, None, self.root_node, k), key=root_key)
            self.subgoals_added[k] += 1

    def get(self) -> SearchTreeNode | None:
        if self.nodes_queue.empty():
            return None

        node = self.nodes_queue.get()
        self.subgoals_selected_for_expansion[node.next_expand_with_k_generator] += 1
        return node

    def is_seen(self, state: np.ndarray) -> bool:
        return tuple(state.flatten()) in self.seen_states

    def add(self, node: SearchTreeNode) -> None:
        self.seen_states.add(tuple(node.state.flatten()))
                
        if node.next_expand_with_k_generator is not None:
            logger.error(
                'Node during adding to queue should have next_expand_with_k_generator set to None'
            )
            logger.error(
                'Planner decides which k-distances should be expanded, not the search itself'
            )
            raise ValueError(
                'Node during adding to queue should have next_expand_with_k_generator set to None'
            )

        for k in self.generators_k_list:
            new_node = SearchTreeNode(
                node.state, node.value, node.low_level_path, node.parent_node, k
            )
            self.nodes_queue.put(data=new_node, key=(-k, -node.value))
            self.subgoals_added[k] += 1

    def get_solution_data(
        self, solving_node: SearchTreeNode | None, search_info: dict
    ) -> tuple[dict, dict]:
        search_info['subgoals_visited'] = len(self.seen_states)
        search_info['search_tree'] = self.root_node
        
        if self.prune_search_trees and solving_node is not None:
            prune_search_tree_from_solving_node(solving_node)
        
        search_info['solving_node'] = solving_node
        
        search_info['subgoals_added'] = self.subgoals_added
        search_info['subgoals_selected_for_expansion'] = self.subgoals_selected_for_expansion

        if solving_node is None:
            return {'solved': False}, search_info

        subgoal_path: list[np.ndarray]
        action_path: list[int]
        subgoal_path, action_path, subgoal_distance_path, _ = get_solving_path_data(
            solving_node, include_state_path=False
        )

        solution: dict[str, bool | list[np.ndarray]] = {
            'solved': True,
            'subgoal_path': subgoal_path,
            'action_path': action_path,
            'subgoal_distance_path': subgoal_distance_path,
        }

        return solution, search_info
