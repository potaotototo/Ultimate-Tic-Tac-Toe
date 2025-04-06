import time
import numpy as np
from utils import State, Action
from abc import ABC, abstractmethod

class StudentAgentWithCache(ABC):
    def __init__(self):
        self.global_cache = {}
        self.last_action_scores = {}

    def choose_action(self, state: State) -> Action:
        self.global_cache.clear()  # clear cache for fresh search
        start_time = time.time()
        time_limit = 2.9
        best_action = state.get_random_valid_action()

        depth = 1
        while time.time() - start_time < time_limit:
            try:
                action = self.search_with_depth(state, depth, start_time, time_limit)
                if action is not None:
                    best_action = action
                depth += 1
            except TimeoutError:
                break

        return best_action

    def search_with_depth(self, state, depth, start_time, time_limit):
        best_action = None
        best_value = -np.inf
        alpha, beta = -np.inf, np.inf
        valid_actions = state.get_all_valid_actions()

        prev_scores = self.last_action_scores.get(depth - 1, {})
        valid_actions.sort(key=lambda a: prev_scores.get(a, 0), reverse=True)

        current_scores = {}

        for action in valid_actions:
            if time.time() - start_time > time_limit:
                raise TimeoutError

            new_state = state.change_state(action)
            value = self.minimax(new_state, depth - 1, alpha, beta, False, start_time, time_limit)
            current_scores[action] = value

            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)

        self.last_action_scores[depth] = current_scores
        return best_action

    def minimax(self, state, depth, alpha, beta, maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError

        key = self.canonical_hash(state)
        if key in self.global_cache:
            return self.global_cache[key]

        if state.is_terminal() or depth == 0:
            score = self.evaluate(state)
            self.global_cache[key] = score
            return score

        valid_actions = state.get_all_valid_actions()
        best_value = -np.inf if maximizing else np.inf

        for action in valid_actions:
            new_state = state.change_state(action)
            eval_value = self.minimax(
                new_state, depth - 1, alpha, beta, not maximizing, start_time, time_limit
            )

            if maximizing:
                best_value = max(best_value, eval_value)
                alpha = max(alpha, eval_value)
            else:
                best_value = min(best_value, eval_value)
                beta = min(beta, eval_value)

            if beta <= alpha:
                break

        self.global_cache[key] = best_value
        return best_value

    @abstractmethod
    def evaluate(self, state):
        """To be implemented by each agent"""
        pass

    def rotate_board_90(self, board: np.ndarray) -> np.ndarray:
        new_board = np.zeros_like(board)
        for i in range(3):
            for j in range(3):
                rotated_subboard = np.rot90(board[i][j], k=3)
                new_board[j][2 - i] = rotated_subboard
        return new_board

    def rotate_local_action_90(self, action):
        if action is None:
            return None
        i, j = action
        return (j, 2 - i)

    def canonical_hash(self, state):
        board = state.board
        action = state.prev_local_action
        fill = state.fill_num

        hashes = []
        for k in range(4):
            rotated_outer = np.rot90(board, k, axes=(0, 1))
            rotated_action = action
            for _ in range(k):
                rotated_action = self.rotate_local_action_90(rotated_action)

            rotated_board = np.empty((3, 3), dtype=object)
            for i in range(3):
                for j in range(3):
                    rotated_board[i, j] = np.rot90(rotated_outer[i, j], k)

            h = hash((rotated_board.tobytes(), rotated_action, fill))
            hashes.append(h)

        return min(hashes)
