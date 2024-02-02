import queue
from collections import namedtuple


class SearchTreeNode:
    """
    A high-level node in the search tree.

    Stores information about the environment state and search metadata,
    such as its value_function, depth in the tree, list of child nodes, parent node
    and the low-level path that leads to it from its parent.
    """

    def __init__(
        self,
        state,
        value,
        low_level_path,
        parent_node,
        next_expand_with_k_generator: int | None = None,
    ):
        # expected_k_distance is the expected distance to the subgoal represented by this instance of SearchTreeNode.
        # None if the node is a root
        self.state = state
        self.value = value
        self.low_level_path = low_level_path
        self.parent_node = parent_node

        self.children = []
        self.is_on_solving_path = False

        self.next_expand_with_k_generator = (
            next_expand_with_k_generator  # None only if root or before adding to planner
        )
        if self.parent_node is not None:
            self.parent_node.children.append(self)


# Result of a validation.
ValidationResult = namedtuple(
    'ValidationResult', ['is_valid', 'is_solved', 'path', 'nodes_visited', 'achieved_state']
)


class SafePriorityQueue:
    """
    Priority queue that uses a counter to ensure unique keys.
    """

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.queue = queue.PriorityQueue()

    def put(self, data, key):
        self.queue.put((key, self.counter, data))
        self.counter += 1

    def get(self):
        if self.queue.empty():
            return None

        return self.queue.get()[-1]

    def empty(self):
        return self.queue.empty()


def get_solving_path_data(solving_node, include_state_path=False, env=None):
    subgoal_path = []
    action_path = []
    subgoal_distance_path = []
    state_path = None

    current_node = solving_node

    while current_node.parent_node is not None:
        subgoal_path.append(current_node.state)
        current_node.is_on_solving_path = True

        if current_node.low_level_path is not None:
            action_path.append(current_node.low_level_path)

        subgoal_distance_path.append(current_node.parent_node.next_expand_with_k_generator)

        current_node = current_node.parent_node

    subgoal_distance_path.append(0)
    subgoal_path.append(current_node.state)
    current_node.is_on_solving_path = True

    assert len(subgoal_path) == len(subgoal_distance_path)

    subgoal_path.reverse()
    action_path.reverse()
    subgoal_distance_path.reverse()
    action_path = flatten(action_path)

    if include_state_path:
        env.restore_full_state_from_np_array_version(subgoal_path[0])
        state_path = [env.get_state()]

        for action in action_path:
            state, _, _, _ = env.step(action)
            state_path.append(state)

    return subgoal_path, action_path, subgoal_distance_path, state_path

def flatten(l):
    return [item for sublist in l for item in sublist]

