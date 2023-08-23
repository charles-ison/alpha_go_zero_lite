import copy
import utilities
from alpha_go_zero.alpha_go_zero import AlphaGoZero


class AlphaGoZeroPureMTCS(AlphaGoZero):

    def __init__(self):
        super().__init__()

    def get_selection_value(self, move):
        return move.get_upper_confidence_bound()

    def get_action_values(self, win_detected, tie_detected, mcts_game, mcts_move, turn_count, player_num):
        return self.get_finished_game_values(win_detected)

    def should_stop_rollout(self, is_expansion_move):
        return False

    def initialize_monte_carlo_tree_search(self, mcts_turn_count, player_num, mcts_game, last_move):
        return
