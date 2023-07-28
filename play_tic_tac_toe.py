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
    print("\nAlphaGo Zero Lite is running simulations. . .")
    run_expansions(player_num, turn_count, game, last_move, mcts_time_limit)

    potential_moves = last_move.children
    if len(potential_moves) == 0:
        print("Bug encountered, no potential AlphaGo Zero Lite moves found. More simulations need to be run.")

    best_move = potential_moves[0]
    for move in potential_moves:
        if move.get_success_rate() > best_move.get_success_rate():
            best_move = move

    return best_move.row, best_move.column

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

def get_num_points(alpha_go_zero_lite_win_detected, tie_detected):
    if alpha_go_zero_lite_win_detected:
        return 3
    elif tie_detected:
        return 1
    else:
        return 0

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
        expansion_game.board[row][column] = expansion_player_num
        last_expansion_move = last_expansion_move.play_new_move(expansion_player_num, row, column)
        expansion_turn_count += 1

        tie_detected = expansion_game.detect_tie()
        win_detected = expansion_game.detect_winner()
        if win_detected or tie_detected:
            alpha_go_zero_lite_win_detected = win_detected and player_num == expansion_player_num
            num_points = get_num_points(alpha_go_zero_lite_win_detected, tie_detected)
            run_backpropagation(last_expansion_move, expansion_root, num_points)
            expansion_game = copy.deepcopy(game)
            last_expansion_move = expansion_root
            expansion_turn_count = turn_count
            num_simulations += 1
    print("AlphaGo Zero Lite ran " + str(num_simulations) + " simulations in " + str(time_limit) + " seconds.")

def run_backpropagation(last_expansion_move, expansion_root, num_points):
    backpropagation_move = last_expansion_move
    while backpropagation_move != expansion_root:
        backpropagation_move.num_points += num_points
        backpropagation_move.num_simulations += 1.0
        backpropagation_move = backpropagation_move.parent

def get_player_num(turn_count):
    return (turn_count % 2) + 1

def play(game, game_root):

    opponent_start_priority = int(input("\nWould you like to be Player 1 or 2? "))
    last_move = game_root
    turn_count = 0
    game.print_board()

    while True:
        player_num = get_player_num(turn_count)
        row, column = get_move(player_num, opponent_start_priority, turn_count, game, last_move, mcts_time_limit)

        game.board[row][column] = player_num
        game.print_board()

        last_move = last_move.play_new_move(player_num, row, column)
        turn_count += 1

        winner_detected = game.detect_winner()
        if game.detect_winner():
            if player_num == opponent_start_priority:
                print("\nCongratulations you won!")
            else:
                print("\nAlphaGo Zero Lite has won!")
            break
        elif game.detect_tie():
            print("\nThe game ended in a tie!")
            break


mcts_time_limit = 5
tic_tac_toe = games.TicTacToe()
game_root = mct.Root("Tic-Tac-Toe")
play(tic_tac_toe, game_root)
