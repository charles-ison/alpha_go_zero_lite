class Node:
    def __init__(self):
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def play_new_move(self, player_num, row, column):
        for child in self.children:
            if child.row == row and child.column == column:
                return child

        new_move = Move(player_num, row, column, self)
        self.add_child(new_move)
        return new_move

class Root(Node):
    def __init__(self, game_name):
        super().__init__()
        self.game_name = game_name

class Move(Node):
    def __init__(self, player, row, column, parent):
        super().__init__()
        self.player = player
        self.row = row
        self.column = column
        self.parent = parent
        self.num_wins = 0
        self.num_simulations = 0

    def get_win_rate(self):
        return float(self.num_wins) / float(self.num_simulations)