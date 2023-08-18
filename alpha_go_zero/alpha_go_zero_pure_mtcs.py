import copy
import utilities
from alpha_go_zero.alpha_go_zero import AlphaGoZero


class AlphaGoZeroPureMTCS(AlphaGoZero):

    def __init__(self):
        super().__init__()

    def get_selection_value(self, move):
        return move.get_upper_confidence_bound()

    def get_action_values(self, win_detected, tie_detected, mcts_game, mcts_move, turn_count, player_num):
        if win_detected or tie_detected:
            return self.get_finished_game_values(win_detected)
        return self.perform_rollout(mcts_game, turn_count, player_num)

    def perform_rollout(self, mcts_game, turn_count, player_num):
        game = copy.deepcopy(mcts_game)
        rollout_turn_count = turn_count

        while True:
            rollout_player_num = utilities.get_player_num(rollout_turn_count)
            potential_moves = game.fetch_potential_moves()
            random_move = self.get_random_move(potential_moves)
            row = random_move[0]
            column = random_move[1]
            game.update_board(row, column, rollout_player_num)
            rollout_turn_count += 1
            if game.detect_winner():
                if rollout_player_num == player_num:
                    return self.win_value, self.lose_value
                else:
                    return self.lose_value, self.win_value
            elif game.detect_tie():
                return self.tie_value, self.tie_value
