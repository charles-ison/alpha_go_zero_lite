import math

class Node:
    def __init__(self):
        self.children = []
        self.unexplored_subtrees = True
        self.num_visits = 0.0

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
        self.num_points = 0.0
        self.exploration_factor = math.sqrt(2)

    def get_upper_confidence_bound(self):
        win_ratio = self.num_points / self.num_visits
        parent_visit_ratio = math.log(self.parent.num_visits) / self.num_visits
        return win_ratio + self.exploration_factor * math.sqrt(parent_visit_ratio)