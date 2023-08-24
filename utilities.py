import monte_carlo_tree as mct


def get_player_num(turn_count):
    return (turn_count % 2) + 1


def get_opposing_player_num(player_num):
    if player_num == 1:
        return 2
    else:
        return 1


def append_move(game, player_num, last_move, row, column):
    for child in last_move.children:
        if child.row == row and child.column == column and child.player_num == player_num:
            if child.num_visits == 0:
                child.num_visits += 1
            return child

    manual_move = mct.Move(game.board_size, player_num, row, column, last_move, 1)
    last_move.children.append(manual_move)
    return manual_move