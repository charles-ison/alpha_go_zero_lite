import copy
import monte_carlo_tree as mct
import utilities


def play_game(game, players, time_limit, add_noise, print_games):
    if print_games:
        game.print_board()
    player_1_root = mct.Node(game.action_space_size, game.dirichlet_alpha, 1)
    player_2_root = mct.Node(game.action_space_size, game.dirichlet_alpha, 1)
    player_nodes = {1: player_1_root, 2: player_2_root}

    while True:
        player_num = game.current_player
        opposing_player_num = utilities.get_opposing_player_num(player_num)
        player = players[player_num - 1]
        player_nodes[player_num] = player.play_move(game, player_nodes[player_num], time_limit, add_noise, print_games)
        action = player_nodes[player_num].action
        player_nodes[opposing_player_num] = utilities.append_move(game, player_num, player_nodes[opposing_player_num], action)
        game.apply_action(action)

        if print_games:
            game.print_board()

        player_nodes[player_num].was_played, player_nodes[opposing_player_num].was_played = True, True

        if game.detect_winner():
            return player_num, player_1_root, player_2_root
        elif game.detect_tie():
            return 0, player_1_root, player_2_root


def play_match(game, num_games, players, time_limit):
    num_ties = 0
    num_player_1_wins = 0
    num_player_2_wins = 0
    for game_num in range(num_games):
        new_game = copy.deepcopy(game)
        result_num, _, _ = play_game(new_game, players, time_limit, False, True)
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
