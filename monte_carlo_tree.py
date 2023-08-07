import math


class Node:
    def __init__(self):
        self.children = []
        self.num_visits = 0.0


class Move(Node):
    def __init__(self, player_num, row, column, parent):
        super().__init__()
        self.player_num = player_num
        self.action_value = 0.0
        self.total_action_value = 0.0
        self.probability = 0.0
        self.row = row
        self.column = column
        self.parent = parent

    def get_mean_action_value(self):
        return self.total_action_value / self.num_visits

    def get_upper_confidence_bound(self):
        exploration_factor = math.sqrt(2)
        action_value_ratio = self.action_value / self.num_visits
        parent_visit_ratio = math.log(self.parent.num_visits) / self.num_visits
        return action_value_ratio + exploration_factor * math.sqrt(parent_visit_ratio)

    def get_predictor_upper_confidence_bound_applied_to_trees(self):
        exploration_factor = math.sqrt(2)
        parent_visit_numerator = math.sqrt(self.parent.num_visits)
        self_visit_denominator = 1 + self.num_visits
        return exploration_factor * self.probability * (parent_visit_numerator/self_visit_denominator)

    def get_mean_action_value_puct(self):
        return self.get_mean_action_value() + self.get_predictor_upper_confidence_bound_applied_to_trees()
