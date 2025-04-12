import time
import numpy as np
from utils import State, Action

# Build upon ThirdAgent (>> (Agent27 == Agent26))
# Intended to be more offensive
class Agent41:
    def __init__(self):
        self.hashboard = {}

    def choose_action(self, state: State) -> Action:
        start_time = time.time()
        time_limit = 2.95
        best_action = state.get_random_valid_action()
        valid_actions = state.get_all_valid_actions()

        if len(valid_actions) == 81:            
            return (1, 1, 1, 1) # start from the centre

        depth = 4
        while True:
            if time.time() - start_time > time_limit:
                break

            try:
                action = self.search_with_depth(state, depth, start_time, time_limit)
                if action is not None:
                    best_action = action
            except TimeoutError:
                break

            depth += 1

        return best_action

    def search_with_depth(self, state, depth, start_time, time_limit):
        best_action = None
        best_value = -np.inf
        alpha, beta = -np.inf, np.inf
        valid_actions = state.get_all_valid_actions()

        for action in valid_actions:
            if time.time() - start_time > time_limit:
                raise TimeoutError

            new_state = state.change_state(action)
            value = self.minimax(new_state, depth - 1, alpha, beta, False, start_time, time_limit)

            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)

        return best_action

    def minimax(self, state, depth, alpha, beta, maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError

        if state.is_terminal() or depth == 0:
            return self.evaluate(state)

        valid_actions = state.get_all_valid_actions()

        if maximizing:
            max_eval = -np.inf
            for action in valid_actions:
                new_state = state.change_state(action)
                eval_value = self.minimax(new_state, depth - 1, alpha, beta, False, start_time, time_limit)
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for action in valid_actions:
                new_state = state.change_state(action)
                eval_value = self.minimax(new_state, depth - 1, alpha, beta, True, start_time, time_limit)
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, state):
        if state.is_terminal():
            return state.terminal_utility()

        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)
        board_win_diff = my_won - opp_won

        global_weights = np.array([[2, 1, 2], [1, 3, 1], [2, 1, 2]])
        global_contrib = np.sum(global_weights * (lbs == my_fill)) - np.sum(global_weights * (lbs == opp_fill))

        win_in_one = 0
        opp_win_in_one = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] != 0:
                    continue
                local = board[i][j]
                if self.has_local_win_in_one(local, my_fill):
                    win_in_one += 1
                if self.has_local_win_in_one(local, opp_fill):
                    opp_win_in_one += 1
        win_in_one /= 9
        opp_win_in_one /= 9

        filled_ratio = np.sum(state.board != 0) / 729

        win_lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]
        two_in_line = 0
        for line in win_lines:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1

        features = np.array([
            board_win_diff,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            filled_ratio,
            two_in_line
        ])

        weights = np.array([
            +0.06,
            +0.20,
            +1.10,
            -0.85,
            +0.14,
            +0.08
        ])
        intercept = 0.0034

        return float(np.dot(weights, features) + intercept)

    def has_local_win_in_one(self, sub_board, fill):
        for i in range(3):
            row = sub_board[i, :]
            col = sub_board[:, i]
            if np.count_nonzero(row == fill) == 2 and np.count_nonzero(row == 0) == 1:
                return True
            if np.count_nonzero(col == fill) == 2 and np.count_nonzero(col == 0) == 1:
                return True
        diag1 = [sub_board[i, i] for i in range(3)]
        diag2 = [sub_board[i, 2 - i] for i in range(3)]
        if diag1.count(fill) == 2 and diag1.count(0) == 1:
            return True
        if diag2.count(fill) == 2 and diag2.count(0) == 1:
            return True
        return False

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

    def hash_state(self, state):
        board = state.board
        action = state.prev_local_action
        fill_num = state.fill_num

        hashes = []
        for _ in range(4):
            h = hash((board.tobytes(), action, fill_num))
            hashes.append(h)
            board = self.rotate_board_90(board)
            action = self.rotate_local_action_90(action)

        return min(hashes)
