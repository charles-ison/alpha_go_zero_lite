import copy
from play_games import play
from games.TicTacToe import TicTacToe
from players.player import Player
from players.player_type import PlayerType
from utilities import get_player_num

def swap_players(players):
    temp = players[0]
    players[0] = players[1]
    players[1] = temp

def get_best_player(game, num_games, best_player, old_player, time_threshold):
    win_counts = [0, 0]
    tie_count = 0
    players = [best_player, old_player]
    for game_num in range(num_games):
        new_game = copy.deepcopy(game)
        result_num = play(new_game, players, time_threshold, False)
        swap_players(players)
        print("\nGame " + str(game_num + 1) + " finished!")

        if result_num == 0:
            tie_count += 1
            print("The result was a tie.")
        elif result_num == get_player_num(game_num):
            win_counts[0] += 1
            print("Best player won.")
        elif result_num == get_player_num(game_num + 1):
            win_counts[1] += 1
            print("Old player won.")

    print("\nNumber of best player wins: " + str(win_counts[0]))
    print("Number of old player wins: " + str(win_counts[1]))
    print("Number of ties: " + str(tie_count))

    most_winning_index = win_counts.index(max(win_counts))
    return players[most_winning_index]

time_threshold = 4
num_evaluation_games = 2
game = TicTacToe()

best_player = Player(PlayerType.MCTS_CNN)
old_player = Player(PlayerType.MCTS_CNN)
players = [best_player, old_player]

best_player = get_best_player(game, num_evaluation_games, best_player, old_player, time_threshold)
