import time
import random
import copy
import games
import monte_carlo_tree as mct

def get_move(player_num, opponent_start_priority, turn_count, game, last_move, mcts_time_limit):
    if player_num == opponent_start_priority:
        return get_opponent_move(player_num, game)
    else:
        return get_alpha_go_zero_lite_move(player_num, turn_count, game, last_move, mcts_time_limit)

def get_alpha_go_zero_lite_move(player_num, turn_count, game, last_move, mcts_time_limit):
    print("\nAlphaGo Zero Lite is running simulations. . .")
    run_expansions(player_num, turn_count, game, last_move, mcts_time_limit)

    potential_moves = last_move.children
    if len(potential_moves) == 0:
        print("Bug encountered, no potential AlphaGo Zero Lite moves found. More simulations need to be run.")

    best_move = potential_moves[0]
    for move in potential_moves[1:]:
        if move.num_visits > best_move.num_visits:
            best_move = move
    return best_move.row, best_move.column

def get_opponent_move(player_num, game):
    row, column = input("\nPlayer " + str(player_num) + " please enter move coordinates: ").split(",")
    row, column = int(row), int(column)

    if game.is_valid_move(row, column):
        return row, column
    else:
        print("\nInvalid move. Please try again.")
        return get_opponent_move(player_num, game)

def move_unexplored(potential_move_tuple, last_expansion_move):
    potential_row = potential_move_tuple[0]
    potential_column = potential_move_tuple[1]
    for child in last_expansion_move.children:
        if child.row == potential_row and child.column == potential_column and not child.unexplored_subtrees:
            return False
    return True

def get_random_move(potential_move_tuples, last_expansion_move):
    unexplored_potential_tuples = [tuple for tuple in potential_move_tuples if move_unexplored(tuple, last_expansion_move)]
    random_move_tuple = unexplored_potential_tuples[random.randint(0, len(unexplored_potential_tuples)-1)]
    return random_move_tuple[0], random_move_tuple[1]

def get_move_with_highest_upper_confidence_bound(last_expansion_move_children):
    best_move = last_expansion_move_children[0]
    for move in last_expansion_move_children[1:]:
        if move.get_upper_confidence_bound() > best_move.get_upper_confidence_bound():
            best_move = move
    return best_move.row, best_move.column

def get_expansion_move(expansion_game, last_expansion_move):
    potential_move_tuples = expansion_game.fetch_potential_moves()
    num_potential_moves = len(potential_move_tuples)
    last_expansion_move_children = last_expansion_move.children

    if num_potential_moves == len(last_expansion_move_children):
        row, column = get_move_with_highest_upper_confidence_bound(last_expansion_move_children)
    else:
        row, column = get_random_move(potential_move_tuples, last_expansion_move)
    return row, column, num_potential_moves

def get_num_points(alpha_go_zero_lite_win_detected, tie_detected):
    if alpha_go_zero_lite_win_detected:
        return 3
    elif tie_detected:
        return 1
    else:
        return -3

def run_expansions(player_num, turn_count, game, last_move, time_threshold):
    expansion_game = copy.deepcopy(game)
    expansion_root = last_move
    expansion_turn_count = turn_count
    last_expansion_move = expansion_root
    num_potential_moves_list = []
    num_simulations = 0
    time_limit = time.time() + time_threshold
    start_time = time.time()
    while time.time() < time_limit:
        expansion_player_num = get_player_num(expansion_turn_count)
        row, column, num_potential_moves = get_expansion_move(expansion_game, last_expansion_move)
        num_potential_moves_list.append(num_potential_moves)
        expansion_game.board[row][column] = expansion_player_num
        last_expansion_move = last_expansion_move.play_new_move(expansion_player_num, row, column)
        expansion_turn_count += 1

        tie_detected = expansion_game.detect_tie()
        win_detected = expansion_game.detect_winner()
        if win_detected or tie_detected:
            alpha_go_zero_lite_win_detected = win_detected and player_num == expansion_player_num
            num_points = get_num_points(alpha_go_zero_lite_win_detected, tie_detected)
            last_expansion_move.unexplored_subtrees = False
            num_potential_moves_list.append(0)
            run_backpropagation(last_expansion_move, expansion_root, num_points, num_potential_moves_list)
            expansion_game = copy.deepcopy(game)
            last_expansion_move = expansion_root
            expansion_turn_count = turn_count
            num_simulations += 1

    stop_time = time.time()
    run_time = int(stop_time - start_time)
    print("AlphaGo Zero Lite ran " + str(num_simulations) + " simulations in " + str(run_time) + " seconds.")

def run_backpropagation(last_expansion_move, expansion_root, num_points, num_potential_moves_list):
    backprop_move = last_expansion_move
    while backprop_move != expansion_root:
        backprop_move.num_points += num_points
        backprop_move.num_visits += 1.0
        num_potential_moves = num_potential_moves_list.pop()
        backprop_move.unexplored_subtrees = check_unexplored_subtrees(backprop_move, num_potential_moves)
        backprop_move = backprop_move.parent

    num_potential_moves = num_potential_moves_list.pop()
    backprop_move.unexplored_subtrees = check_unexplored_subtrees(backprop_move, num_potential_moves)
    backprop_move.num_visits += 1.0

    if isinstance(backprop_move, mct.Move):
        backprop_move.num_points += num_points

def check_unexplored_subtrees(backpropagation_move, num_potential_moves):
    if not backpropagation_move.unexplored_subtrees:
        return False
    if len(backpropagation_move.children) != num_potential_moves:
        return True
    for child in backpropagation_move.children:
        if child.unexplored_subtrees:
            return True
    return False

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

        if game.detect_winner():
            if player_num == opponent_start_priority:
                print("\nCongratulations you won!")
            else:
                print("\nAlphaGo Zero Lite has won!")
            break
        elif game.detect_tie():
            print("\nThe game ended in a tie!")
            break

mcts_time_limit = 60
tic_tac_toe = games.TicTacToe()
game_root = mct.Root("Tic-Tac-Toe")
play(tic_tac_toe, game_root)
