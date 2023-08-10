import torch
import copy


class Game:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = torch.zeros(board_size, board_size)
        self.board_history = []
        self.board_history.append(torch.zeros(self.board_size, self.board_size))
        self.board_history.append(torch.zeros(self.board_size, self.board_size))
        self.board_history.append(torch.zeros(self.board_size, self.board_size))

    def update_board(self, row, column, player_num):
        self.board[row][column] = player_num
        self.board_history.append(copy.deepcopy(self.board))

    def get_current_player_board_representation(self, player_num):
        if player_num == 2:
            return torch.ones(self.board_size, self.board_size)
        else:
            return torch.zeros(self.board_size, self.board_size)

    def fetch_potential_moves(self):
        remaining_moves = []
        for row_index, row in enumerate(self.board):
            for column_index, value in enumerate(row):
                if value == 0:
                    remaining_moves.append((row_index, column_index))
        return remaining_moves

    def is_valid_move(self, row, column):
        return row < self.board_size and column < self.board_size and self.board[row][column] == 0