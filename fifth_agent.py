import time
import numpy as np
from utils import State, Action

class FifthAgent:
    def __init__(self):
        self.hashboard_by_depth = {}
        self.last_action_scores = {}

    def choose_action(self, state: State) -> Action:
        start_time = time.time()
        time_limit = 2.9
        best_action = state.get_random_valid_action()

        depth = 1
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

        prev_scores = self.last_action_scores.get(depth - 1, {})
        valid_actions.sort(key=lambda a: prev_scores.get(a, 0), reverse=True)

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

        self.last_action_scores[depth] = current_scores
        return best_action

    def minimax(self, state, depth, alpha, beta, maximizing, start_time, time_limit, root_depth):
        if time.time() - start_time > time_limit:
            raise TimeoutError

        key = self.canonical_hash(state)
        if root_depth not in self.hashboard_by_depth:
            self.hashboard_by_depth[root_depth] = {}
        if key in self.hashboard_by_depth[root_depth]:
            return self.hashboard_by_depth[root_depth][key]

        if state.is_terminal() or depth == 0:
            score = self.evaluate(state)
            self.hashboard_by_depth[root_depth][key] = score
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
            self.hashboard_by_depth[root_depth][key] = max_eval
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
            self.hashboard_by_depth[root_depth][key] = min_eval
            return min_eval

    def evaluate(self, state):
        if state.is_terminal():
            return state.terminal_utility()
        
        lbs = state.local_board_status
        board = state.board
        my_fill = 1
        opp_fill = 2

        # Feature 1: Local boards won
        my_won = np.sum(lbs == my_fill)
        opp_won = np.sum(lbs == opp_fill)

        # Feature 2: Centre priority (revived)
        center_control = 0
        if board[1][1][1, 1] == my_fill:  
            center_control = 1
        elif board[1][1][1, 1] == opp_fill:  
            center_control = -1

        # Feature 8: Global win contribution (weighted board control)
        global_weights = np.array([
            [2, 1, 2],
            [1, 3, 1],
            [2, 1, 2]
        ])
        global_contrib = np.sum(global_weights * (lbs == my_fill)) - np.sum(global_weights * (lbs == opp_fill))

        # Feature 9: Can win a local board in one move
        # Feature 10: Opponent can win a local board in one move
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

        # Feature 13: Number of filled cells
        filled_ratio = np.sum(state.board != 0) / 81 / 9

        # Feature 14: Freedom of next move
        freedom = 0.0
        if state.prev_local_action is not None:
            i, j = state.prev_local_action
            if state.local_board_status[i][j] != 0:
                freedom = 1.0

        # Feature 15: Track if the opponent is about to win local board
        blocking_opportunities = 0
        for i in range(3):
            for j in range(3):
                if lbs[i][j] == 0:  # Empty spot
                    local_board = board[i][j]
                    if self.has_local_win_in_one(local_board, opp_fill):
                        blocking_opportunities += 1
        blocking_opportunities /= 9

        # Feature 16: Prioritise player turn
        player_turn_advantage = 0.1 if state.fill_num % 2 == 0 else -0.1

        weights = np.array([
                +0.0800,
                +0.0448,
                +0.1067,
                +1.1254,
                -0.5692,
                +0.0229,
                +0.0000,
                -0.5692,
                -1.5228
            ])
        intercept = -0.0004

        features = np.array([
            my_won - opp_won,
            center_control,
            global_contrib,
            win_in_one,
            opp_win_in_one,
            filled_ratio,
            freedom,
            blocking_opportunities,
            player_turn_advantage
        ])

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

    def canonical_hash(self, state):
        board = state.board.copy()
        action = state.prev_local_action
        fill = state.fill_num

        hashes = []
        for k in range(4):
            rotated_board = np.rot90(board, k, axes=(0, 1))
            rotated_board = np.array([[np.rot90(rotated_board[i, j], k) for j in range(3)] for i in range(3)])
            rotated_action = action
            for _ in range(k):
                rotated_action = self.rotate_local_action_90(rotated_action)
            h = hash((rotated_board.tobytes(), rotated_action, fill))
            hashes.append(h)

        return min(hashes)
