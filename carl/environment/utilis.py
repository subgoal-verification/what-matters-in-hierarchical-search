import numpy as np


class HashableState:
    state = np.random.get_state()
    np.random.seed(0)
    hash_key = np.random.normal(size=10000)
    np.random.set_state(state)

    def __init__(self, one_hot, agent_pos, unmached_boxes, fast_eq=False):
        self.one_hot = one_hot
        self.agent_pos = agent_pos
        self.unmached_boxes = unmached_boxes
        self._hash = None
        self.fast_eq = fast_eq
        self._initial_state_hash = None

    def __iter__(self):
        yield from [self.one_hot, self.agent_pos, self.unmached_boxes]

    def __hash__(self):
        if self._hash is None:
            flat_np = self.one_hot.flatten()
            self._hash = int(np.dot(flat_np, HashableState.hash_key[: len(flat_np)]) * 10e8)
        return self._hash

    def __eq__(self, other):
        if self.fast_eq:
            return hash(self) == hash(other)  # This is a conscious decision to speed up.
        else:
            return np.array_equal(self.one_hot, other.one_hot)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_raw(self):
        return self.one_hot, self.agent_pos, self.unmached_boxes

    def get_np_array_version(self):
        return self.one_hot
