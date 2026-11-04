import monte_carlo_tree as mct


def get_player_num(turn_count):
    return (turn_count % 2) + 1


def get_opposing_player_num(player_num):
    if player_num == 1:
        return 2
    else:
        return 1


def append_move(game, player_num, last_move, action):
    for child in last_move.children:
        if child.action == action and child.player_num == player_num:
            if child.num_visits == 0:
                child.num_visits += 1
            return child

    manual_move = mct.Move(game.action_space_size, game.dirichlet_alpha, player_num, action, last_move, 1)
    last_move.children.append(manual_move)
    return manual_move