import torch
import utilities
from players.player import Player


class AlphaGoZeroProbability(Player):

    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def play_move(self, game, player_num, turn_count, last_move, num_searches, add_noise, print_games):
        board_history = game.get_board_history(turn_count, player_num)
        self.cnn.eval()
        child_probabilities, _ = self.cnn(board_history.unsqueeze(dim=0))
        child_probabilities = child_probabilities.squeeze()
        most_recent_board = board_history[0]
        masked_child_probabilities = self.mask_child_probabilities(child_probabilities, most_recent_board)
        row, column = self.get_max_probability_indices(masked_child_probabilities)
        return utilities.append_move(game, player_num, last_move, row, column)

    def mask_child_probabilities(self, child_probabilities, most_recent_board):
        for row_index, row in enumerate(most_recent_board):
            for column_index, element in enumerate(row):
                if element != 0:
                    child_probabilities[row_index][column_index] = 0
        return child_probabilities

    def get_max_probability_indices(self, masked_child_probabilities):
        return (masked_child_probabilities == torch.max(masked_child_probabilities)).nonzero().squeeze()