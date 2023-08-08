import monte_carlo_tree as mct
import utilities
from player_type import PlayerType
from games.TicTacToe import TicTacToe
from player import Player


def play_move(game, player_num, players, turn_count, last_move, time_threshold):
    player_num_index = player_num - 1
    player = players[player_num_index]
    player_type = player.player_type
    if player_type == PlayerType.Manual:
        return get_manual_move(player_num, game, last_move)
    else:
        player_type_name = str(player_type.name)
        print("\nAlphaGo Zero Lite (" + player_type_name + ") is running Monte Carlo Tree Search for " + str(time_threshold) + " seconds. . .")
        return player.alpha_go_zero_lite.get_move(turn_count, game, last_move, time_threshold)


def append_move(game, player_num, last_move, row, column):
    for child in last_move.children:
        if child.row == row and child.column == column and child.player_num == player_num:
            return child

    manual_move = mct.Move(game.board_size, player_num, row, column, last_move)
    last_move.children.append(manual_move)
    return manual_move


def get_manual_move(player_num, game, last_move):
    row, column = input("\nPlayer " + str(player_num) + " please enter move coordinates: ").split(",")
    row, column = int(row), int(column)

    if game.is_valid_move(row, column):
        return append_move(game, player_num, last_move, row, column)
    else:
        print("\nInvalid move. Please try again.")
        return get_manual_move(player_num, game, last_move)


def get_current_player_tree(player_num, player_1_node, player_2_node):
    if player_num == 1:
        return player_1_node
    else:
        return player_2_node


def play(game, players, time_threshold):
    turn_count = 0
    game.print_board()
    current_player_node, waiting_player_node = mct.Node(game.board_size), mct.Node(game.board_size)

    while True:
        player_num = utilities.get_player_num(turn_count)
        current_player_node = play_move(game, player_num, players, turn_count, current_player_node, time_threshold)
        waiting_player_node = append_move(game, player_num, waiting_player_node, current_player_node.row, current_player_node.column)

        game.update_board(current_player_node.row, current_player_node.column, player_num)
        game.print_board()

        current_player_node, waiting_player_node = waiting_player_node, current_player_node
        turn_count += 1

        if game.detect_winner():
            return player_num
        elif game.detect_tie():
            return 0


def get_num_games():
    return int(input("\nHow many games would you like to play? "))


def get_player(player_num):
    player_type = get_player_type(player_num)
    return Player(player_type)

def get_player_type(player_num):
    print("Please select Player " + str(player_num) + " type.")
    print("1. AlphaGo Zero Lite with pure Monte Carlo Tree Search")
    print("2. AlphaGo Zero Lite with CNN")
    print("3. Manual player")
    return PlayerType(int(input("Please enter your selection: ")))


def play_games(num_games, players, time_threshold):
    num_ties = 0
    num_player_1_wins = 0
    num_player_2_wins = 0
    for game_num in range(num_games):
        tic_tac_toe = TicTacToe()
        result_num = play(tic_tac_toe, players, time_threshold)
        player_1_type = str(players[0].player_type.name)
        player_2_type = str(players[1].player_type.name)
        print("\nGame " + str(game_num) + " finished!")

        if result_num == 0:
            num_ties += 1
            print("The result was a tie.")
        elif result_num == 1:
            num_player_1_wins += 1
            print("Player 1 (" + player_1_type + ") won.")
        elif result_num == 2:
            num_player_2_wins += 1
            print("Player 2 (" + player_2_type + ") won.")

    print("\nFinal Statistics: ")
    print("Number of games: ", num_games)
    print("Number of Player 1 (" + player_1_type + ") wins: ", num_player_1_wins)
    print("Number of Player 2 (" + player_2_type + ") wins: ", num_player_2_wins)
    print("Number of ties: ", num_ties)


time_threshold = 4
players = [get_player(1), get_player(2)]
num_games = get_num_games()
play_games(num_games, players, time_threshold)
