from player_type import PlayerType
from alpha_go_zero.alpha_go_zero_pure_mtcs import AlphaGoZeroPureMTCS
from alpha_go_zero.alpha_go_zero_cnn import AlphaGoZeroCNN


class Player:
    def __init__(self, player_type):
        self.player_type = player_type
        if player_type == PlayerType.Pure_MCTS:
            self.alpha_go_zero_lite = AlphaGoZeroPureMTCS()
        elif player_type == PlayerType.MCTS_CNN:
            self.alpha_go_zero_lite = AlphaGoZeroCNN()