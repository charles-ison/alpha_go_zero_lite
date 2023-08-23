import torch
from players.player_type import PlayerType
from players.player import Player
from games.tic_tac_toe import TicTacToe
from play_games import play_games


def get_player(player_num, device):
    player_type = get_player_type(player_num)
    return Player(player_type, device)


def get_player_type(player_num):
    print("Please select Player " + str(player_num) + " type.")
    print("1. AlphaGo Zero Lite with pure Monte Carlo Tree Search")
    print("2. AlphaGo Zero Lite with CNN")
    print("3. AlphaGo Zero Lite with Untrained CNN")
    print("4. Manual player")
    return PlayerType(int(input("Please enter your selection: ")))


def get_num_games():
    return int(input("\nHow many games would you like to play? "))


# About 500 searches is required for the pure MCTS to achieve perfect play
num_searches = 500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
players = [get_player(1, device), get_player(2, device)]
num_games = get_num_games()
game = TicTacToe(device)
play_games(game, num_games, players, num_searches)

