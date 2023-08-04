from neural_networks.tic_tac_toe_cnn import TicTacToeCNN
from alpha_go_zero_lite import Alpha_Go_Zero_Lite

class Alpha_Go_Zero_Lite_CNN(Alpha_Go_Zero_Lite):

    def __init__(self):
        super().__init__()
        self.cnn = TicTacToeCNN()
        #policy, value = cnn(game.board_history)
