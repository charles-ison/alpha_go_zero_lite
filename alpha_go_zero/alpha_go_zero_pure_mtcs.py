from alpha_go_zero.alpha_go_zero import AlphaGoZero


class AlphaGoZeroPureMTCS(AlphaGoZero):

    def __init__(self):
        super().__init__()

    def get_selection_move(self, last_expansion_move_children):
        best_move = last_expansion_move_children[0]
        for move in last_expansion_move_children[1:]:
            if move.get_upper_confidence_bound() > best_move.get_upper_confidence_bound():
                best_move = move
        return best_move

    def should_stop_rollout(self, is_simulation):
        return False

    def get_expanded_node_number_of_visits(self):
        return 1

    def save_game_analysis(self, mcts_game, expansion_move):
        return

    def get_action_values(self, win_detected, tie_detected, mcts_game, expansion_move, turn_count, player_num):
        return self.get_finished_game_values(win_detected)
