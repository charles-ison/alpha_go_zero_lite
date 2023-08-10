from players.player_type import PlayerType
from players.player import Player
from games.tic_tac_toe import TicTacToe
from play_games import play_games


def get_player(player_num):
    player_type = get_player_type(player_num)
    return Player(player_type)

def get_player_type(player_num):
    print("Please select Player " + str(player_num) + " type.")
    print("1. AlphaGo Zero Lite with pure Monte Carlo Tree Search")
    print("2. AlphaGo Zero Lite with CNN")
    print("3. Manual player")
    return PlayerType(int(input("Please enter your selection: ")))

def get_num_games():
    return int(input("\nHow many games would you like to play? "))

time_threshold = 4
players = [get_player(1), get_player(2)]
num_games = get_num_games()
game = TicTacToe()
play_games(game, num_games, players, time_threshold)