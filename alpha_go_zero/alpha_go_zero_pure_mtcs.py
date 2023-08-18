from alpha_go_zero.alpha_go_zero import AlphaGoZero


class AlphaGoZeroPureMTCS(AlphaGoZero):

    def __init__(self):
        super().__init__()

    def get_selection_value(self, move):
        return move.get_upper_confidence_bound()

    def should_stop_rollout(self, is_simulation):
        return False

    def get_action_values(self, win_detected, tie_detected, mcts_game, expansion_move, turn_count, player_num):
        return self.get_finished_game_values(win_detected)
