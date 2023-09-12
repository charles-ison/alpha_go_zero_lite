import copy
import monte_carlo_tree as mct
import utilities


def get_current_player_tree(player_num, player_1_node, player_2_node):
    if player_num == 1:
        return player_1_node
    else:
        return player_2_node


def play(game, players, time_limit, add_noise, print_games):
    turn_count = 0
    if print_games:
        game.print_board()
    player_1_root, player_2_root = mct.Node(game.board_size, 1), mct.Node(game.board_size, 1)
    current_player_node, waiting_player_node = player_1_root, player_2_root

    while True:
        player_num = utilities.get_player_num(turn_count)
        player = players[player_num - 1]
        current_player_node = player.play_move(game, player_num, turn_count, current_player_node, time_limit, add_noise, print_games)
        waiting_player_node = utilities.append_move(game, player_num, waiting_player_node, current_player_node.row, current_player_node.column)
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


def play_games(game, num_games, players, time_limit):
    num_ties = 0
    num_player_1_wins = 0
    num_player_2_wins = 0
    for game_num in range(num_games):
        new_game = copy.deepcopy(game)
        result_num, _, _ = play(new_game, players, time_limit, False, True)
        player_1_type = type(players[0]).__name__
        player_2_type = type(players[1]).__name__
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
