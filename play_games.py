import copy
import monte_carlo_tree as mct
import utilities
from players.player_type import PlayerType


#TODO: This should be refactored as a function that exists on a player class
def play_move(game, player_num, players, turn_count, last_move, num_searches, print_games):
    player_num_index = player_num - 1
    player = players[player_num_index]
    player_type = player.player_type
    if player_type == PlayerType.Manual:
        return get_manual_move(player_num, game, last_move)
    else:
        player_type_name = str(player_type.name)
        if print_games:
            print("\nAlphaGo Zero Lite (" + player_type_name + ") is running " + str(num_searches) + "  Monte Carlo Tree Searches. . .")
        return player.alpha_go_zero_lite.get_move(turn_count, player_num, game, last_move, num_searches, print_games)


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


def play(game, players, num_searches, print_games):
    turn_count = 0
    if print_games:
        game.print_board()
    player_1_root, player_2_root = mct.Node(game.board_size), mct.Node(game.board_size)
    current_player_node, waiting_player_node = player_1_root, player_2_root

    while True:
        player_num = utilities.get_player_num(turn_count)
        current_player_node = play_move(game, player_num, players, turn_count, current_player_node, num_searches, print_games)
        waiting_player_node = append_move(game, player_num, waiting_player_node, current_player_node.row, current_player_node.column)

        game.update_board(current_player_node.row, current_player_node.column, player_num)
        if print_games:
            game.print_board()

        current_player_node.was_played, waiting_player_node.was_played = True, True
        current_player_node, waiting_player_node = waiting_player_node, current_player_node
        turn_count += 1

        if game.detect_winner():
            return player_num, player_1_root, player_2_root
        elif game.detect_tie():
            return 0, player_1_root, player_2_root


def play_games(game, num_games, players, num_searches):
    num_ties = 0
    num_player_1_wins = 0
    num_player_2_wins = 0
    for game_num in range(num_games):
        new_game = copy.deepcopy(game)
        result_num, _, _ = play(new_game, players, num_searches, True)
        player_1_type = str(players[0].player_type.name)
        player_2_type = str(players[1].player_type.name)
        print("\nGame " + str(game_num + 1) + " finished!")

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
