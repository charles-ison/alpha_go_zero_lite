import math
import torch


class Node:
    def __init__(self, action_space_size, dirichlet_alpha, num_visits):
        self.children = []
        self.child_probabilities = None
        dirichlet_concentration = torch.full((action_space_size,), dirichlet_alpha)
        dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(dirichlet_concentration)
        self.probability_noise = dirichlet_distribution.sample()
        self.num_visits = num_visits


class Move(Node):
    def __init__(self, action_space_size, dirichlet_alpha, player_num, action, parent, num_visits):
        super().__init__(action_space_size, dirichlet_alpha, num_visits)
        self.player_num = player_num
        self.action_value = 0.0
        self.action = action
        self.parent = parent
        self.was_played = False

    def get_upper_confidence_bound(self):
        exploration_factor = math.sqrt(2)
        # Adding 1 to the denominator is not 100% faithful to the UCT algorithm, but it allows for
        # consistent code with AlphaGo Zero
        parent_visit_ratio = math.log(self.parent.num_visits) / (self.num_visits + 1)
        return self.get_mean_action_value() + exploration_factor * math.sqrt(parent_visit_ratio)

    def get_predictor_upper_confidence_bound_applied_to_trees(self, add_noise):
        exploration_factor = math.sqrt(2)
        alternative_move_visits = math.sqrt(self.parent.num_visits)
        self_visit_denominator = 1 + self.num_visits
        probability = self.get_probability(add_noise)
        return exploration_factor * probability * (alternative_move_visits/self_visit_denominator)

    def get_probability(self, add_noise):
        probability = self.parent.child_probabilities[self.action]
        if add_noise:
            epsilon = 0.25
            probability_noise = self.parent.probability_noise[self.action]
            probability = (1 - epsilon) * probability + epsilon * probability_noise
        return probability

    def get_mean_action_value_plus_puct(self, add_noise):
        return self.get_mean_action_value() + self.get_predictor_upper_confidence_bound_applied_to_trees(add_noise)

    def get_mean_action_value(self):
        if self.num_visits == 0:
            return 0.0
        else:
            return self.action_value / self.num_visits
