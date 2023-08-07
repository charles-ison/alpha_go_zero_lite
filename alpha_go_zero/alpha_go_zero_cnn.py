from neural_networks.tic_tac_toe_cnn import TicTacToeCNN
from alpha_go_zero.alpha_go_zero import AlphaGoZero

class AlphaGoZeroCNN(AlphaGoZero):

    def __init__(self):
        super().__init__()
        self.cnn = TicTacToeCNN()
        policy, value = self.cnn(mcts_game.board_history)
