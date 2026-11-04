import torch
import utilities
from players.player import Player


class AlphaGoZeroProbability(Player):

    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def play_move(self, game, last_move, time_limit, add_noise, print_games):
        state = game.encode_state()
        self.cnn.eval()
        policy_logits, _ = self.cnn(state.unsqueeze(dim=0))
        child_probabilities = torch.softmax(policy_logits.flatten(), dim=0)
        action = self.get_max_probability_action(child_probabilities, game.get_legal_actions())
        return utilities.append_move(game, game.current_player, last_move, action)

    def get_max_probability_action(self, child_probabilities, legal_actions):
        return max(legal_actions, key=lambda action: child_probabilities[action])
