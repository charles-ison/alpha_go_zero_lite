from neural_networks.tic_tac_toe_cnn import TicTacToeCNN
from alpha_go_zero.alpha_go_zero import AlphaGoZero

class AlphaGoZeroCNN(AlphaGoZero):

    def __init__(self):
        super().__init__()
        self.cnn = TicTacToeCNN()
        policy, value = self.cnn(mcts_game.board_history)

    def get_selection_move(self, last_expansion_move_children):
        best_move = last_expansion_move_children[0]
        for move in last_expansion_move_children[1:]:
            move_PUCT = move.get_predictor_upper_confidence_bound_applied_to_trees()
            best_PUCT = best_move.get_predictor_upper_confidence_bound_applied_to_trees()
            if move_PUCT > best_PUCT:
                best_move = move
        return best_move
