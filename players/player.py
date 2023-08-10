import torch
from players.player_type import PlayerType
from alpha_go_zero.alpha_go_zero_pure_mtcs import AlphaGoZeroPureMTCS
from alpha_go_zero.alpha_go_zero_cnn import AlphaGoZeroCNN
from neural_networks.tic_tac_toe_cnn import TicTacToeCNN


class Player:
    def __init__(self, player_type, cnn=None):
        self.player_type = player_type
        if player_type == PlayerType.Pure_MCTS:
            self.alpha_go_zero_lite = AlphaGoZeroPureMTCS()
        elif player_type == PlayerType.MCTS_CNN:
            if cnn is None:
                cnn = TicTacToeCNN()
                cnn.load_state_dict(torch.load("tic_tac_toe_cnn.pt"))
            self.alpha_go_zero_lite = AlphaGoZeroCNN(cnn)
