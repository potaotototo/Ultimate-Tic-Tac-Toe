import time
import numpy as np
from utils import State, Action

def generate_win_in_one_table():
        table = {}
        all_cells = [0, 1, 2]
        from itertools import product

        for config in product(all_cells, repeat=9):
            board = np.array(config).reshape(3, 3)
            for fill in [1, 2]:
                count = 0
                for i in range(3):
                    row = board[i, :]
                    col = board[:, i]
                    if np.count_nonzero(row == fill) == 2 and np.count_nonzero(row == 0) == 1:
                        count += 1
                    if np.count_nonzero(col == fill) == 2 and np.count_nonzero(col == 0) == 1:
                        count += 1
                diag1 = [board[i, i] for i in range(3)]
                diag2 = [board[i, 2 - i] for i in range(3)]
                if diag1.count(fill) == 2 and diag1.count(0) == 1:
                    count += 1
                if diag2.count(fill) == 2 and diag2.count(0) == 1:
                    count += 1

                table[(tuple(config), fill)] = count > 0
        return table

WIN_IN_ONE_LOOKUP = generate_win_in_one_table()

# Agent22 with more offensive early-mid game
# Targets D2
class Agent25:
    def __init__(self):
        self.global_cache = {}
        self.killer_moves = {}  # store killer moves by depth
        self.last_action_scores = {}

    def choose_action(self, state: State) -> Action:
        start_time = time.time()
        time_limit = 2.9
        best_action = state.get_random_valid_action()

        depth = 4  # start searching further
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

        # Move ordering: use killer moves + previous scores
        killer = self.killer_moves.get(depth, None)
        prev_scores = self.last_action_scores.get(depth - 1, {})

        def score_key(a):
            if a == killer:
                return float('inf')  # highest priority
            return prev_scores.get(a, 0)

        valid_actions.sort(key=score_key, reverse=True)

        current_scores = {}

        for action in valid_actions:
            if time.time() - start_time > time_limit:
                raise TimeoutError

            new_state = state.change_state(action)
            value = self.minimax(new_state, depth - 1, alpha, beta, False, start_time, time_limit, depth)
            current_scores[action] = value

            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, value)

            if beta <= alpha:
                self.killer_moves[depth] = action  # store killer move, prioritise later
                break

        self.last_action_scores[depth] = current_scores
        return best_action

    def minimax(self, state, depth, alpha, beta, maximizing, start_time, time_limit, root_depth):
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

        if maximizing:
            max_eval = -np.inf
            for action in valid_actions:
                new_state = state.change_state(action)
                eval_value = self.minimax(new_state, depth - 1, alpha, beta, False, start_time, time_limit, root_depth)
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            self.global_cache[key] = max_eval
            return max_eval
        else:
            min_eval = np.inf
            for action in valid_actions:
                new_state = state.change_state(action)
                eval_value = self.minimax(new_state, depth - 1, alpha, beta, True, start_time, time_limit, root_depth)
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            self.global_cache[key] = min_eval
            return min_eval

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

    def evaluate(self, state):
        if state.is_terminal():
            return state.terminal_utility()

        my_fill = 1
        opp_fill = 2

        lbs = state.local_board_status
        board = state.board

        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)

        center_control = 0
        if board[1][1][1, 1] == my_fill:
            center_control = 1
        elif board[1][1][1, 1] == opp_fill:
            center_control = -1

        global_weights = np.array([
            [2, 1, 2],
            [1, 3, 1],
            [2, 1, 2]
        ])
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

        filled_ratio = np.sum(state.board != 0) / 81 / 9

        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        player_turn_advantage = 0.1 if state.fill_num % 2 == 0 else -0.1

        two_in_line = 0
        opponent_two_in_line = 0
        win_lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]
        for line in win_lines:
            cells = [lbs[i][j] for i, j in line]
            if cells.count(my_fill) == 2 and cells.count(0) == 1:
                two_in_line += 1
            if cells.count(opp_fill) == 2 and cells.count(0) == 1:
                opponent_two_in_line += 1

        total_available_moves = 9
        opponent_valid_moves = sum(lbs[i][j] == 0 for i in range(3) for j in range(3))
        restricted_freedom = 1 - (opponent_valid_moves / total_available_moves)
        restricted_and_blocked = restricted_freedom + blocking_opportunities

        features = np.array([
            my_won - opp_won,
            center_control,
            global_contrib,
            win_in_one,
            filled_ratio,
            player_turn_advantage,
            two_in_line,
            opponent_two_in_line,
            restricted_and_blocked
        ])

        weights = np.array([
            +0.0369,
            +0.0407,
            +0.0725,
            +0.7170,
            +0.0000,
            -1.7778,
            +0.682, # typo but works
            -0.0822,
            -0.4630
        ])
        intercept = -0.0134

        if filled_ratio < 0.10:
            weights[2] *= 5.0  # global_contrib
        elif filled_ratio < 0.40:
            weights[3] *= 3.0  # win_in_one
            weights[6] *= 3.0
            weights[8] *= 2.0  # blocked
        else:
            weights[3] *= 2.5
            weights[8] *= 2.5
            weights[7] *= 1.5  # opponent_two_in_line

        return float(np.dot(weights, features) + intercept)

    def has_local_win_in_one(self, sub_board, fill):
        return WIN_IN_ONE_LOOKUP[(tuple(sub_board.flatten()), fill)]
