import torch
from players.manual_player import ManualPlayer
from players.player_type import PlayerType
from players.alpha_go_zero.alpha_go_zero_raw_mtcs_player import AlphaGoZeroRawMTCSPlayer
from players.alpha_go_zero.alpha_go_zero_cnn_player import AlphaGoZeroCNNPlayer
from neural_networks.tic_tac_toe_cnn import TicTacToeCNN


class PlayerFactory:

    def get_player(self, player_type, device):
        player = None
        if player_type == PlayerType.Manual:
            player = ManualPlayer()
        if player_type == PlayerType.Raw_MCTS:
            player = AlphaGoZeroRawMTCSPlayer()
        elif player_type == PlayerType.MCTS_CNN:
            cnn = TicTacToeCNN()
            cnn.load_state_dict(torch.load("neural_networks/saved_models/best_tic_tac_toe_cnn.pt"))
            cnn.to(device)
            player = AlphaGoZeroCNNPlayer(cnn)
        elif player_type == PlayerType.Untrained_MCTS_CNN:
            cnn = TicTacToeCNN()
            cnn.to(device)
            player = AlphaGoZeroCNNPlayer(cnn)
        return player

