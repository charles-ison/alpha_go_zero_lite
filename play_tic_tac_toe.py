import time
import random
import copy
import games
import monte_carlo_tree as mct

def get_move(player_num, opponent_start_priority, turn_count, game, last_move, mcts_time_limit):
    if player_num == opponent_start_priority:
        return get_opponet_move(player_num, game)
    else:
        return get_alpha_go_zero_lite_move(player_num, turn_count, game, last_move, mcts_time_limit)

def get_alpha_go_zero_lite_move(player_num, turn_count, game, last_move, mcts_time_limit):
    print("AlphaGo Zero Lite is running simulations. . .")
    run_expansions(player_num, turn_count, game, last_move, mcts_time_limit)
    return 0, 0

def get_opponet_move(player_num, game):
    row, column = input("\nPlayer " + str(player_num) + " please enter move coordinates: ").split(",")
    row, column = int(row), int(column)

    if game.is_valid_move(row, column):
        return row, column
    else:
        print("\nInvalid move. Please try again.")
        return get_opponet_move(player_num, game)

def get_random_expansion_move(expansion_game):
    potential_moves = expansion_game.fetch_potential_moves()
    random_move_tuple = potential_moves[random.randint(0, len(potential_moves)-1)]
    return random_move_tuple[0], random_move_tuple[1]

def run_expansions(player_num, turn_count, game, last_move, time_limit):
    expansion_game = copy.deepcopy(game)
    expansion_root = last_move
    expansion_turn_count = turn_count
    last_expansion_move = expansion_root
    num_simulations = 0
    stop_time = time.time() + time_limit
    while time.time() < stop_time:
        expansion_player_num = get_player_num(expansion_turn_count)
        row, column = get_random_expansion_move(expansion_game)
        expansion_game.board[row][column] = player_num
        last_expansion_move = last_expansion_move.play_new_move(player_num, row, column)
        expansion_turn_count += 1

        if expansion_game.detect_winner():
            # backpropogate results
            expansion_game = copy.deepcopy(game)
            last_expansion_move = expansion_root
            expansion_turn_count = turn_count
            num_simulations += 1
    print("Alpha Go Zero ran " + str(num_simulations) + " simulations in " + str(time_limit) + " seconds.")

def get_player_num(turn_count):
    return (turn_count % 2) + 1

def play(game, game_root):

    opponent_start_priority = int(input("\nWould you like to be Player 1 or 2? "))
    last_move = game_root
    turn_count = 0
    player_num = 1

    while game.detect_winner() is False:
        game.print_board()
        player_num = get_player_num(turn_count)
        row, column = get_move(player_num, opponent_start_priority, turn_count, game, last_move, mcts_time_limit)
        game.board[row][column] = player_num
        last_move = last_move.play_new_move(player_num, row, column)
        turn_count += 1

    game.print_board()
    print("\nPlayer " + str(player_num) + " won!")


mcts_time_limit = 5
tic_tac_toe = games.TicTacToe()
game_root = mct.Root("Tic-Tac-Toe")
play(tic_tac_toe, game_root)
