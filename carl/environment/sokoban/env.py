### Start Core Sokoban - ported and adapted to carl from https://gitlab.com/awarelab/gym-sokoban/-/blob/master/gym_sokoban/envs/sokoban_env_fast.py
import enum
import itertools

import numpy as np
import pkg_resources
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from carl.environment.env import GameEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.environment.tokenizer import GameTokenizer
from carl.environment.utilis import HashableState

RENDERING_MODES = ['one_hot', 'rgb_array', 'tiny_rgb_array']


class _SokobanEnvCore:
    metadata = {'render.modes': RENDERING_MODES}

    def __init__(
        self,
        dim_room=(10, 10),
        max_steps=np.inf,
        num_boxes=4,
        num_gen_steps=None,
        mode='one_hot',
        fast_state_eq=False,
        penalty_for_step=-0.1,
        reward_box_on_target=1,
        reward_finished=10,
        seed=None,
        load_boards_from_file=None,
        load_boards_lazy=True,
    ):
        self._seed = seed
        self.mode = mode
        self.num_gen_steps = num_gen_steps
        self.dim_room = dim_room
        self.max_steps = max_steps
        self.num_boxes = num_boxes

        # Penalties and Rewards
        self.penalty_for_step = penalty_for_step
        self.reward_box_on_target = reward_box_on_target
        self.reward_finished = reward_finished

        self._internal_state = None
        self.fast_state_eq = fast_state_eq
        self._surfaces = load_surfaces()
        self.initial_internal_state_hash = None
        self.load_boards_from_file = load_boards_from_file
        self.boards_from_file = None
        if not load_boards_lazy:
            self.boards_from_file = np.load(self.load_boards_from_file)

    def step(self, action):
        raw_state, rew, done = self._step(
            self._internal_state.get_raw(),
            action,
            self.penalty_for_step,
            self.reward_box_on_target,
            self.reward_finished,
        )
        self._internal_state = HashableState(*raw_state, fast_eq=self.fast_state_eq)
        return self._internal_state.one_hot, rew, done, {'solved': done}

    def render(self, mode='one_hot'):
        assert mode in RENDERING_MODES, f'Only {RENDERING_MODES} are supported, not {mode}'
        if mode == 'one_hot':
            return self._internal_state.one_hot
        render_surfaces = None
        if mode == 'rgb_array':
            render_surfaces = self._surfaces['16x16pixels']
        if mode == 'tiny_rgb_array':
            render_surfaces = self._surfaces['8x8pixels']

        size_x = self._internal_state.one_hot.shape[0] * render_surfaces.shape[1]
        size_y = self._internal_state.one_hot.shape[1] * render_surfaces.shape[2]

        res = np.tensordot(self._internal_state.one_hot, render_surfaces, (-1, 0))
        res = np.transpose(res, (0, 2, 1, 3, 4))
        return np.reshape(res, (size_x, size_y, 3))

    def clone_full_state(self):
        internal_state = self._internal_state
        internal_state._initial_state_hash = self.initial_internal_state_hash
        return internal_state

    def get_state(self):
        return self._internal_state.get_np_array_version()

    def restore_full_state(self, state):
        self._internal_state = state
        self.initial_internal_state_hash = state._initial_state_hash

    def restore_full_state_from_np_array_version(self, state_np, quick=False):
        if (state_np > 255).any() or (state_np < 0).any():
            raise ValueError(
                f'restore_full_state_from_np_array_version() got '
                f'data out of range 0-255 {state_np}'
            )
        if quick:
            agent_pos = None
            unmatched_boxes = None
        else:
            shape = state_np.shape[:2]
            agent_pos = np.unravel_index(
                np.argmax(
                    state_np[..., _SokobanEnvCore.FieldStates.player]
                    + state_np[..., _SokobanEnvCore.FieldStates.player_target]
                ),
                shape=shape,
            )
            unmatched_boxes = int(np.sum(state_np[..., _SokobanEnvCore.FieldStates.box]))
        self._internal_state = HashableState(
            state_np, agent_pos, unmatched_boxes, fast_eq=self.fast_state_eq
        )

    class FieldStates(enum.IntEnum):
        wall = 0
        empty = 1
        target = 2
        box_target = 3
        box = 4
        player = 5
        player_target = 6

    def _step(self, state, action, penalty_for_step, reward_box_on_target, reward_finished):
        empty = 1
        target = 2
        box_target = 3
        box = 4
        player = 5
        player_target = 6

        delta_x, delta_y = None, None
        if action == 0:
            delta_x, delta_y = -1, 0
        elif action == 1:
            delta_x, delta_y = 1, 0
        elif action == 2:
            delta_x, delta_y = 0, -1
        elif action == 3:
            delta_x, delta_y = 0, 1

        one_hot, agent_pos, unmatched_boxes = state

        arena = np.zeros(shape=(3,), dtype=np.uint8)
        for i in range(3):
            index_x = agent_pos[0] + i * delta_x
            index_y = agent_pos[1] + i * delta_y
            if index_x < one_hot.shape[0] and index_y < one_hot.shape[0]:
                arena[i] = np.where(one_hot[index_x, index_y, :] == 1)[0][0]

        new_unmatched_boxes_ = unmatched_boxes
        new_agent_pos = agent_pos
        new_arena = np.copy(arena)

        box_moves = (arena[1] == box or arena[1] == box_target) and (
            arena[2] == empty or arena[2] == 2
        )

        agent_moves = arena[1] == empty or arena[1] == target or box_moves

        if agent_moves:
            targets = (
                (arena == target).astype(np.int8)
                + (arena == box_target).astype(np.int8)
                + (arena == player_target).astype(np.int8)
            )
            if box_moves:
                last_field = box - 2 * targets[2]  # Weirdness due to inconsistent target non-target
            else:
                last_field = arena[2] - targets[2]

            new_arena = np.array([empty, player, last_field]).astype(np.uint8) + targets.astype(
                np.uint8
            )
            new_agent_pos = (agent_pos[0] + delta_x, agent_pos[1] + delta_y)

            if box_moves:
                new_unmatched_boxes_ = int(unmatched_boxes - (targets[2] - targets[1]))

        new_one_hot = np.copy(one_hot)
        for i in range(3):
            index_x = agent_pos[0] + i * delta_x
            index_y = agent_pos[1] + i * delta_y
            if index_x < one_hot.shape[0] and index_y < one_hot.shape[0]:
                one_hot_field = np.zeros(shape=7)
                one_hot_field[new_arena[i]] = 1
                new_one_hot[index_x, index_y, :] = one_hot_field

        done = new_unmatched_boxes_ == 0
        reward = penalty_for_step - reward_box_on_target * (
            float(new_unmatched_boxes_) - float(unmatched_boxes)
        )
        if done:
            reward += reward_finished

        new_state = (new_one_hot, new_agent_pos, new_unmatched_boxes_)

        return new_state, reward, done


def load_surfaces():
    # Necessarily keep the same order as in FieldStates
    assets_file_name = [
        'wall.png',
        'floor.png',
        'box_target.png',
        'box_on_target.png',
        'box.png',
        'player.png',
        'player_on_target.png',
    ]
    sizes = ['8x8pixels', '16x16pixels']

    resource_package = __name__
    surfaces = {}
    for size in sizes:
        surfaces[size] = []
        for asset_file_name in assets_file_name:
            asset_path = pkg_resources.resource_filename(
                resource_package, '/'.join(('surface', size, asset_file_name))
            )
            asset_np_array = np.array(Image.open(asset_path))
            surfaces[size].append(asset_np_array)

        surfaces[size] = np.stack(surfaces[size])

    return surfaces


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes, dtype=np.uint8)[a.reshape(-1)]).reshape(
        a.shape + (num_classes,)
    )


# End Core


class SokobanEnv(GameEnv):
    name: str = 'sokoban'

    def __init__(self, tokenizer: SokobanTokenizer, num_boxes=4) -> None:
        self._tokenizer = tokenizer
        self.dim_room = tokenizer.size_of_board
        self.num_boxes = num_boxes
        self.core = _SokobanEnvCore(dim_room=self.dim_room, num_boxes=num_boxes)
        self.done = False

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.core.step(action)
        return state, reward, done, info

    def next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.core.restore_full_state_from_np_array_version(state)
        state, _, done, _ = self.core.step(action)
        self.done = self.done
        return state

    def restore_full_state_from_np_array_version(self, state) -> None:
        self.core.restore_full_state_from_np_array_version(state)

    def set_state(self, state: np.ndarray) -> np.ndarray:
        self.core.restore_full_state_from_np_array_version(state)
        return state

    def get_state(self):
        return self.core.get_state()

    @property
    def tokenizer(self) -> GameTokenizer:
        return self._tokenizer

    def detect_action(self, board_before: np.ndarray, board_after: np.ndarray) -> int:
        x_before: int
        y_before: int
        x_after: int
        y_after: int
        x_before, y_before = self.get_agent_position(board_before)
        x_after, y_after = self.get_agent_position(board_after)
        delta_x: int = x_after - x_before
        delta_y: int = y_after - y_before

        return self.agent_coordinates_to_action(delta_x, delta_y)

    def get_agent_position(self, board: np.ndarray) -> tuple[int, int]:
        width: int
        height: int
        width, height, _ = board.shape
        for xy in itertools.product(list(range(width)), list(range(height))):
            x: int
            y: int
            x, y = xy
            obj: str = self.get_field_name_from_index(int(np.argmax(board[x][y])))

            if obj == 'agent':
                return x, y

            if obj == 'agent_on_goal':
                return x, y

        raise AssertionError('No agent on the board')

    def num_boxes_on_target(self, state: np.ndarray) -> int:
        box_on_target: int = 0
        for x in range(self.dim_room[0]):
            for y in range(self.dim_room[1]):
                if np.argmax(state[x][y]) == self.get_field_index_from_name('box_on_goal'):
                    box_on_target += 1

        return box_on_target

    def is_solved(self, state: np.ndarray) -> bool:
        box_on_target: int = self.num_boxes_on_target(state)
        return box_on_target == self.num_boxes

    @staticmethod
    def get_field_name_from_index(x: int) -> str:
        objects: dict[int, str] = {
            0: 'wall',
            1: 'empty',
            2: 'goal',
            3: 'box_on_goal',
            4: 'box',
            5: 'agent',
            6: 'agent_on_goal',
        }
        return objects[x]

    @staticmethod
    def agent_coordinates_to_action(delta_x: int, delta_y: int) -> int:
        """
        Returns action corresponding to given change of agent's coordinates (-1, 0 or 1). Correspondence was taken from
        here: https://gitlab.com/awarelab/gym-sokoban/-/blob/master/gym_sokoban/envs/sokoban_env_fast.py#L166
        """
        assert delta_x in (-1, 0, 1), 'Wrong value for delta_x argument'
        assert delta_y in (-1, 0, 1), 'Wrong value for delta_y argument'

        translation: dict[tuple[int, int], int] = {
            (-1, 0): 0,
            (1, 0): 1,
            (0, -1): 2,
            (0, 1): 3,
        }
        translation_key: tuple[int, int] = (delta_x, delta_y)

        assert translation_key in translation, 'Action should consists of exactly one move'

        return translation[translation_key]

    @staticmethod
    def distribution_to_action(distribution: Tensor) -> int:
        return int(distribution.argmax(dim=-1))

    @staticmethod
    def get_field_index_from_name(x: str) -> int:
        objects_class = {
            'wall': 0,
            'empty': 1,
            'goal': 2,
            'box_on_goal': 3,
            'box': 4,
            'agent': 5,
            'agent_on_goal': 6,
        }
        return objects_class[x]

    def state_to_pic(self, state: np.ndarray):
        self.core.restore_full_state_from_np_array_version(state)
        return self.core.render(mode='rgb_array').astype(int)

    def show_state(
        self, state: np.ndarray, title: str | None = None, file_name: str | None = None
    ) -> None:
        pic = self.state_to_pic(state)
        plt.clf()
        if title is not None:
            plt.title(title)
        plt.imshow(pic)
        if file_name is not None:
            plt.savefig(file_name)

    def state_to_fig(self, state: np.ndarray, title: str | None):
        pic = self.state_to_pic(state)
        plt.clf()

        if title is not None:
            plt.title(title)
        plt.imshow(pic)
        plt.close()
        return plt.gcf()

    def many_states_to_fig(self, states: list[np.ndarray], titles: list[str]):
        def draw_and_describe(plot, state, title):
            pic = self.state_to_pic(state)
            plot.set_title(f'{title}')
            plot.imshow(pic)

        plt.clf()
        n_states = len(states)
        fig, plots = plt.subplots(1, n_states)
        fig.set_size_inches(
            3 * n_states,
            3,
        )
        for idx, plot in enumerate(plots):
            draw_and_describe(plot, states[idx], titles[idx])

        plt.close()
        return fig

    def show_many_states(self, states: list[np.ndarray], titles: list[str]):
        self.many_states_to_fig(states, titles)
        plt.show()
