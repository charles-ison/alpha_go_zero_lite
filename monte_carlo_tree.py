import math
import torch


class Node:
    def __init__(self, board_size):
        self.children = []
        self.child_probabilities = torch.zeros(board_size, board_size)
        self.num_visits = 0.0


class Move(Node):
    def __init__(self, board_size, player_num, row, column, parent, was_played=False):
        super().__init__(board_size)
        self.player_num = player_num
        self.action_value = 0.0
        self.row = row
        self.column = column
        self.parent = parent
        self.was_played = was_played

    def get_mean_action_value(self):
        return self.action_value / self.num_visits

    def get_upper_confidence_bound(self):
        exploration_factor = math.sqrt(2)
        action_value_ratio = self.action_value / self.num_visits
        parent_visit_ratio = math.log(self.parent.num_visits) / self.num_visits
        return action_value_ratio + exploration_factor * math.sqrt(parent_visit_ratio)

    def get_predictor_upper_confidence_bound_applied_to_trees(self):
        exploration_factor = math.sqrt(2)
        parent_visit_numerator = math.sqrt(self.parent.num_visits)
        self_visit_denominator = 1 + self.num_visits
        probability = self.parent.child_probabilities[self.row][self.column]
        return exploration_factor * probability * (parent_visit_numerator/self_visit_denominator)

    def get_mean_action_value_plus_puct(self):
        return self.get_mean_action_value() + self.get_predictor_upper_confidence_bound_applied_to_trees()
