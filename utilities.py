def get_player_num(turn_count):
    return (turn_count % 2) + 1


def get_opposing_player_num(player_num):
    if player_num == 1:
        return 2
    else:
        return 1