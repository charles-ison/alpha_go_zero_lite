import time
import random
import copy
import monte_carlo_tree as mct
from game_mode import GameMode
from games.TicTacToe import TicTacToe


def print_alpha_go_zero_lite_status(player_num, game_mode):
    if game_mode == GameMode.Manual:
        print("\nAlphaGo Zero Lite is running simulations. . .")
    elif game_mode == GameMode.Self_Play:
        print("\nAlphaGo Zero Lite Player " + str(player_num) + " is running simulations. . .")


def play_move(game, player_num, game_mode, opponent_start_priority, turn_count, last_move, time_threshold):
    if game_mode == GameMode.Manual and player_num == opponent_start_priority:
        return get_manual_move(player_num, game, last_move)
    else:
        print_alpha_go_zero_lite_status(player_num, game_mode)
        return get_alpha_go_zero_lite_move(player_num, turn_count, game, last_move, time_threshold)


def get_alpha_go_zero_lite_move(player_num, turn_count, game, last_move, time_threshold):
    run_monte_carlo_tree_search(player_num, turn_count, game, last_move, time_threshold)
    potential_moves = last_move.children
    if len(potential_moves) == 0:
        print("Bug encountered, no potential AlphaGo Zero Lite moves found. More searches need to be run.")

    best_move = potential_moves[0]
    for move in potential_moves[1:]:
        if move.num_visits > best_move.num_visits:
            best_move = move
    return best_move


def append_move(player_num, last_move, row, column):
    for child in last_move.children:
        if child.row == row and child.column == column and child.player_num == player_num:
            return child

    manual_move = mct.Move(player_num, row, column, last_move)
    last_move.children.append(manual_move)
    return manual_move


def get_manual_move(player_num, game, last_move):
    row, column = input("\nPlayer " + str(player_num) + " please enter move coordinates: ").split(",")
    row, column = int(row), int(column)

    if game.is_valid_move(row, column):
        return append_move(player_num, last_move, row, column)
    else:
        print("\nInvalid move. Please try again.")
        return get_manual_move(player_num, game, last_move)


def move_unexplored(potential_move_tuple, last_mcts_move_children):
    potential_row = potential_move_tuple[0]
    potential_column = potential_move_tuple[1]
    for child in last_mcts_move_children:
        if child.row == potential_row and child.column == potential_column:
            return False
    return True


def get_simulation_move(mcts_player_num, potential_move_tuples, last_mcts_move):
    unexplored_potential_tuples = [tuple for tuple in potential_move_tuples if move_unexplored(tuple, last_mcts_move.children)]
    random_move_tuple = unexplored_potential_tuples[random.randint(0, len(unexplored_potential_tuples)-1)]
    return mct.Move(mcts_player_num, random_move_tuple[0], random_move_tuple[1], last_mcts_move)


def get_selection_move(last_expansion_move_children):
    best_move = last_expansion_move_children[0]
    for move in last_expansion_move_children[1:]:
        if move.get_upper_confidence_bound() > best_move.get_upper_confidence_bound():
            best_move = move
    return best_move


def get_num_points(win_detected):
    if win_detected:
        return 3, -3
    else:
        return 1, 1


#TODO: Figure out why this check is needed, suggests there is a bug somewhere
def all_children_visited(last_mcts_move_children):
    for child in last_mcts_move_children:
        if child.num_visits == 0.0:
            return False
    return True


def should_run_selection_move(potential_moves, last_mcts_move_children):
    return len(potential_moves) == len(last_mcts_move_children) and all_children_visited(last_mcts_move_children)


def get_next_mcts_move(mcts_game, mcts_player_num, last_mcts_move, expansion_move):
    if should_run_selection_move(mcts_game.fetch_potential_moves(), last_mcts_move.children):
        return get_selection_move(last_mcts_move.children), expansion_move
    else:
        next_mcts_move = get_simulation_move(mcts_player_num, mcts_game.fetch_potential_moves(), last_mcts_move)
        if expansion_move == None:
            expansion_move = next_mcts_move
            last_mcts_move.children.append(expansion_move)
        return next_mcts_move, expansion_move


def get_backpropagation_leaf(expansion_move, mcts_move):
    if expansion_move is None:
        return mcts_move
    else:
        return expansion_move


def run_monte_carlo_tree_search(player_num, turn_count, game, last_move, time_threshold):
    mcts_game = copy.deepcopy(game)
    mcts_move = last_move
    expansion_move = None
    mcts_turn_count = turn_count
    num_searches = 0
    time_limit = time.time() + time_threshold
    start_time = time.time()
    while time.time() < time_limit:
        mcts_player_num = get_player_num(mcts_turn_count)
        mcts_move, expansion_move = get_next_mcts_move(mcts_game, mcts_player_num, mcts_move, expansion_move)
        mcts_game.board[mcts_move.row][mcts_move.column] = mcts_move.player_num
        mcts_turn_count += 1

        win_detected, tie_detected = mcts_game.detect_winner(), mcts_game.detect_tie()
        if win_detected or tie_detected:
            current_player_points, opposing_player_points = get_num_points(win_detected)
            points_dict = {
                mcts_player_num: current_player_points,
                mcts_move.parent.player_num: opposing_player_points
            }
            backpropagation_leaf = get_backpropagation_leaf(expansion_move, mcts_move)
            run_backpropagation(backpropagation_leaf, last_move, points_dict)
            mcts_game = copy.deepcopy(game)
            expansion_move = None
            mcts_move = last_move
            mcts_turn_count = turn_count
            num_searches += 1

    stop_time = time.time()
    run_time = int(stop_time - start_time)
    print("AlphaGo Zero Lite ran " + str(num_searches) + " searches in " + str(run_time) + " seconds.")


def run_backpropagation(backpropagation_leaf, mcts_root, points_dict):
    mcts_root.num_visits += 1.0
    backprop_move = backpropagation_leaf
    while backprop_move != mcts_root:
        backprop_move.num_points += points_dict[backprop_move.player_num]
        backprop_move.num_visits += 1.0
        backprop_move = backprop_move.parent


def get_player_num(turn_count):
    return (turn_count % 2) + 1


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


def play(game, game_mode, opponent_start_priority, time_threshold):
    turn_count = 0
    game.print_board()
    current_player_node, waiting_player_node = mct.Node(), mct.Node()

    while True:
        player_num = get_player_num(turn_count)
        current_player_node = play_move(game, player_num, game_mode, opponent_start_priority, turn_count, current_player_node, time_threshold)
        waiting_player_node = append_move(player_num, waiting_player_node, current_player_node.row, current_player_node.column)

        game.board[current_player_node.row][current_player_node.column] = player_num
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


def play_multiple_games(num_games, game_mode, opponent_start_priority, time_threshold):
    num_ties = 0
    num_player_1_wins = 0
    num_player_2_wins = 0
    for game_num in range(num_games):
        tic_tac_toe = TicTacToe()
        result_num = play(tic_tac_toe, game_mode, opponent_start_priority, time_threshold)
        print("\nGame " + str(game_num) + " finished!")

        if result_num == 0:
            num_ties += 1
            print("The result was a tie.")
        elif result_num == 1:
            num_player_1_wins += 1
            print("Player 1 won.")
        elif result_num == 2:
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
play_multiple_games(num_games, game_mode, opponent_start_priority, time_threshold)