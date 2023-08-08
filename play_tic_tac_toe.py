import monte_carlo_tree as mct
import utilities
from game_mode import GameMode
from games.TicTacToe import TicTacToe
from alpha_go_zero.alpha_go_zero_configuration import AlphaGoZeroConfiguration
from alpha_go_zero.alpha_go_zero_pure_mtcs import AlphaGoZeroPureMTCS
from alpha_go_zero.alpha_go_zero_cnn import AlphaGoZeroCNN


def print_alpha_go_zero_lite_status(player_num, game_mode):
    if game_mode == GameMode.Manual:
        print("\nAlphaGo Zero Lite is running Monte Carlo Tree Search. . .")
    elif game_mode == GameMode.Self_Play:
        print("\nAlphaGo Zero Lite Player " + str(player_num) + " is running Monte Carlo Tree Search. . .")


def play_move(game, player_num, game_mode, alpha_go_zero, opponent_start_priority, turn_count, last_move, time_threshold):
    if game_mode == GameMode.Manual and player_num == opponent_start_priority:
        return get_manual_move(player_num, game, last_move)
    else:
        print_alpha_go_zero_lite_status(player_num, game_mode)
        return alpha_go_zero.get_move(turn_count, game, last_move, time_threshold)


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


def get_opponent_start_priority(game_mode):
    if game_mode == GameMode.Manual:
        return int(input("\nWould you like to be Player 1 or 2? "))
    else:
        return True


def get_current_player_tree(player_num, player_1_node, player_2_node):
    if player_num == 1:
        return player_1_node
    else:
        return player_2_node


def play(game, game_mode, alpha_go_zero, opponent_start_priority, time_threshold):
    turn_count = 0
    game.print_board()
    current_player_node, waiting_player_node = mct.Node(game.board_size), mct.Node(game.board_size)

    while True:
        player_num = utilities.get_player_num(turn_count)
        current_player_node = play_move(game, player_num, game_mode, alpha_go_zero, opponent_start_priority, turn_count, current_player_node, time_threshold)
        waiting_player_node = append_move(game, player_num, waiting_player_node, current_player_node.row, current_player_node.column)

        game.update_board(current_player_node.row, current_player_node.column, player_num)
        game.print_board()

        current_player_node, waiting_player_node = waiting_player_node, current_player_node
        turn_count += 1

        if game.detect_winner():
            return player_num
        elif game.detect_tie():
            return 0


def get_num_games(game_mode):
    if GameMode.Manual == game_mode:
        return 1
    else:
        return int(input("\nHow many simulations would you like to run? "))


def get_game_mode():
    print("Would you like to: ")
    print("1. Play against AlphaGo Zero Lite")
    print("2. Run a simulation where AlphaGo Zero Lite plays against itself")
    return GameMode(int(input("Please enter your selection: ")))


def get_alpha_go_zero_model(alpha_go_zero_configuration):
    if AlphaGoZeroConfiguration.Pure_MCTS == alpha_go_zero_configuration:
        return AlphaGoZeroPureMTCS()
    else:
        return AlphaGoZeroCNN()


def get_alpha_go_zero():
    print("Which AlphaGo Zero Lite configuration would you like used?: ")
    print("1. Pure Monte Carlo Tree Search")
    print("2. Monte Carlo Tree Search With CNN")
    alpha_go_zero_configuration = AlphaGoZeroConfiguration(int(input("Please enter your selection: ")))
    return get_alpha_go_zero_model(alpha_go_zero_configuration)


def play_games(num_games, game_mode, alpha_go_zero, opponent_start_priority, time_threshold):
    num_ties = 0
    num_player_1_wins = 0
    num_player_2_wins = 0
    for game_num in range(num_games):
        tic_tac_toe = TicTacToe()
        result_num = play(tic_tac_toe, game_mode, alpha_go_zero, opponent_start_priority, time_threshold)
        print("\nGame " + str(game_num) + " finished!")

        if result_num == 0:
            num_ties += 1
            print("The result was a tie.")
        elif result_num == 1:
            num_player_1_wins += 1
            print("Player 1 won.")
        elif result_num == 2:
            num_player_2_wins += 1
            print("Player 2 won.")

    print("\nFinal Statistics: ")
    print("Number of games: ", num_games)
    print("Number of Player 1 wins: ", num_player_1_wins)
    print("Number of Player 2 wins: ", num_player_2_wins)
    print("Number of ties: ", num_ties)


time_threshold = 4
game_mode = get_game_mode()
opponent_start_priority = get_opponent_start_priority(game_mode)
num_games = get_num_games(game_mode)
alpha_go_zero = get_alpha_go_zero()
play_games(num_games, game_mode, alpha_go_zero, opponent_start_priority, time_threshold)