import math

class Node:
    def __init__(self):
        self.children = []
        self.num_visits = 0.0


class Move(Node):
    def __init__(self, player_num, row, column, parent):
        super().__init__()
        self.player_num = player_num
        self.num_points = 0.0
        self.row = row
        self.column = column
        self.parent = parent
        self.exploration_factor = math.sqrt(2)

    def get_upper_confidence_bound(self):
        point_ratio = self.num_points / self.num_visits
        parent_visit_ratio = math.log(self.parent.num_visits) / self.num_visits
        return point_ratio + self.exploration_factor * math.sqrt(parent_visit_ratio)