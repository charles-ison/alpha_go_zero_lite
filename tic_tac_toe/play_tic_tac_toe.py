import torch
from players.player_factory import PlayerFactory
from players.player_type import PlayerType
from games.tic_tac_toe import TicTacToe
from play_games import play_games


def get_player(player_factory, player_num, device):
    player_type = get_player_type(player_num)
    return player_factory.get_player(player_type, device)


def get_player_type(player_num):
    print("Please select Player " + str(player_num) + " type.")
    print("1. AlphaGo Zero Lite with pure Monte Carlo Tree Search")
    print("2. AlphaGo Zero Lite with CNN")
    print("3. AlphaGo Zero Lite with Untrained CNN")
    print("4. Manual player")
    return PlayerType(int(input("Please enter your selection: ")))


def get_num_games():
    return int(input("\nHow many games would you like to play? "))


# About 2 seconds is required for pure MCTS to achieve perfect play
time_limit = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
player_factory = PlayerFactory()
players = [get_player(player_factory, 1, device), get_player(player_factory, 2, device)]
num_games = get_num_games()
game = TicTacToe(device)
play_games(game, num_games, players, time_limit)

