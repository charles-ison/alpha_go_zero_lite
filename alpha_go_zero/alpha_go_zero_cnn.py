from neural_networks.tic_tac_toe_cnn import TicTacToeCNN
from alpha_go_zero.alpha_go_zero import AlphaGoZero

class AlphaGoZeroCNN(AlphaGoZero):

    def __init__(self):
        super().__init__()
        self.cnn = TicTacToeCNN()

    def get_selection_move(self, last_expansion_move_children):
        best_move = last_expansion_move_children[0]
        for move in last_expansion_move_children[1:]:
            if move.get_mean_action_value_plus_puct() > best_move.get_mean_action_value_plus_puct():
                best_move = move
        return best_move

    def should_stop_rollout(self, is_simulation):
        return is_simulation

    def get_action_values(self, win_detected, tie_detected, mcts_game, expansion_move):
        if win_detected or tie_detected:
            return self.get_finished_game_values(win_detected)
        else:
            child_probabilities, action_value = self.cnn(mcts_game.get_most_recent_board_history())
            expansion_move.child_probabilities = child_probabilities
            return action_value, -action_value

