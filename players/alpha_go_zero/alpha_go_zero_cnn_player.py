from players.alpha_go_zero.alpha_go_zero_player import AlphaGoZeroPlayer


class AlphaGoZeroCNNPlayer(AlphaGoZeroPlayer):

    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def get_selection_value(self, move):
        return move.get_mean_action_value_plus_puct()

    def get_action_values(self, win_detected, tie_detected, mcts_game, mcts_move, turn_count, player_num):
        if win_detected or tie_detected:
            return self.get_finished_game_values(win_detected)
        else:
            board_history = mcts_game.get_board_history(turn_count, player_num)
            self.cnn.eval()
            child_probabilities, action_value = self.cnn(board_history.unsqueeze(dim=0))
            mcts_move.child_probabilities = child_probabilities.squeeze()
            return action_value, -action_value

    def should_stop_rollout(self, is_expansion_move):
        return is_expansion_move