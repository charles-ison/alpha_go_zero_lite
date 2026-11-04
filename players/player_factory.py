import torch
from games.game_type import GameType
from players.manual_player import ManualPlayer
from players.player_type import PlayerType
from players.alpha_go_zero.alpha_go_zero_raw_mtcs_player import AlphaGoZeroRawMTCSPlayer
from players.alpha_go_zero.alpha_go_zero_cnn_player import AlphaGoZeroCNNPlayer
from players.alpha_go_zero.alpha_go_zero_probability import AlphaGoZeroProbability
from models.tic_tac_toe_cnn import TicTacToeCNN


class PlayerFactory:

    def get_player(self, player_type, game_type, device):
        player = None
        if player_type == PlayerType.Manual:
            player = ManualPlayer()
        elif player_type == PlayerType.Raw_MCTS:
            player = AlphaGoZeroRawMTCSPlayer()
        elif player_type == PlayerType.MCTS_CNN:
            cnn = self.get_trained_cnn(game_type, device)
            player = AlphaGoZeroCNNPlayer(cnn)
        elif player_type == PlayerType.Untrained_MCTS_CNN:
            cnn = self.get_cnn(game_type)
            cnn.to(device)
            player = AlphaGoZeroCNNPlayer(cnn)
        elif player_type == PlayerType.Probability:
            cnn = self.get_trained_cnn(game_type, device)
            player = AlphaGoZeroProbability(cnn)
        return player

    def get_cnn(self, game_type):
        cnn = None
        if game_type == GameType.TicTacToe:
            cnn = TicTacToeCNN()
        return cnn

    def get_trained_cnn(self, game_type, device):
        cnn = self.get_cnn(game_type)
        cnn.load_state_dict(torch.load(get_best_model_path(game_type), map_location=device))
        cnn.to(device)
        return cnn


def get_model_path(game_type):
    return "models/saved/" + game_type.value + "_cnn.pt"


def get_best_model_path(game_type):
    return "models/saved/best_" + game_type.value + "_cnn.pt"
