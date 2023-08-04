from neural_networks.tic_tac_toe_cnn import TicTacToeCNN
from alpha_go_zero_lite import AlphaGoZeroLite

class AlphaGoZeroLiteCNN(AlphaGoZeroLite):

    def __init__(self):
        super().__init__()
        self.cnn = TicTacToeCNN()
        #policy, value = cnn(game.board_history)
