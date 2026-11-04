from enum import Enum
from games.tic_tac_toe import TicTacToe


class GameType(Enum):
    TicTacToe = "tic_tac_toe"


class GameFactory:

    def get_game(self, game_type, device):
        game = None
        if game_type == GameType.TicTacToe:
            game = TicTacToe(device)
        return game
