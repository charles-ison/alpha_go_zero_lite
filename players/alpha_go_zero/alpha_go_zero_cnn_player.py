from players.alpha_go_zero.alpha_go_zero_player import AlphaGoZeroPlayer


class AlphaGoZeroCNNPlayer(AlphaGoZeroPlayer):

    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def call_cnn(self, mcts_move, mcts_game, mcts_turn_count, player_num):
        board_history = mcts_game.get_board_history(mcts_turn_count, player_num)
        self.cnn.eval()
        child_probabilities, action_value = self.cnn(board_history.unsqueeze(dim=0))
        mcts_move.child_probabilities = child_probabilities.squeeze()
        return action_value, -action_value

    def get_selection_value(self, move):
        return move.get_mean_action_value_plus_puct()

    def get_action_values(self, win_detected, tie_detected, mcts_game, mcts_move, mcts_turn_count, player_num):
        if win_detected or tie_detected:
            return self.get_finished_game_values(win_detected)
        else:
            return self.call_cnn(mcts_move, mcts_game, mcts_turn_count, player_num)

    def should_stop_rollout(self, expansion_move_performed):
        return expansion_move_performed

    def initialize_run_mcts(self, mcts_move, mcts_turn_count, mcts_game):
        if mcts_turn_count == 0:
            # Using 0 as the last players number for the first move of the game
            self.call_cnn(mcts_move, mcts_game, mcts_turn_count, 0)
