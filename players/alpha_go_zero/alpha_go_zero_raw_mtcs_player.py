from players.alpha_go_zero.alpha_go_zero_player import AlphaGoZeroPlayer


class AlphaGoZeroRawMTCSPlayer(AlphaGoZeroPlayer):

    def __init__(self):
        super().__init__()

    def get_selection_value(self, move, add_noise):
        return move.get_upper_confidence_bound()

    def get_action_values(self, win_detected, tie_detected, mcts_game, mcts_move, turn_count, player_num):
        return self.get_finished_game_values(win_detected)

    def should_stop_rollout(self, expansion_move_performed):
        return False

    def initialize_run_mcts(self, mcts_move, mcts_turn_count, mcts_game, mcts_player_num):
        return

    def get_next_move(self, potential_moves, add_noise):
        return self.get_most_visited_potential_move(potential_moves)
