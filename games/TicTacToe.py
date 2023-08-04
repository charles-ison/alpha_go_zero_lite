import torch


class TicTacToe:

    def __init__(self):
        self.board = torch.zeros(3, 3)
        self.board_history = torch.zeros(4, 3, 3)

    def update_board(self, row, column, player_num):
        self.board[row][column] = player_num
        self.update_board_history(player_num)

    def get_current_player_board_representation(self, player_num):
        if player_num == 1:
            return torch.ones(3, 3)
        else:
            return torch.zeros(3, 3)

    def update_board_history(self, player_num):
        self.board_history[3] = self.get_current_player_board_representation(player_num)
        self.board_history[2] = self.board_history[1]
        self.board_history[1] = self.board_history[0]
        self.board_history[0] = self.board

    def is_valid_move(self, row, column):
        return row < 3 and column < 3 and self.board[row][column] == 0

    def fetch_potential_moves(self):
        remaining_moves = []
        for row_index, row in enumerate(self.board):
            for column_index, value in enumerate(row):
                if value == 0:
                    remaining_moves.append((row_index, column_index))
        return remaining_moves

    def detect_winner(self):
        return self.detect_row_winner() or self.detect_column_winner() or self.detect_diagonal_winner()

    def detect_row_winner(self):
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return True
        return False

    def detect_column_winner(self):
        for column in self.board.t():
            if column[0] == column[1] == column[2] != 0:
                return True
        return False

    def detect_diagonal_winner(self):
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return True
        elif self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return True
        return False

    def detect_tie(self):
        remaining_moves = 0
        for row in self.board:
            for value in row:
                if value == 0:
                    return False
        return True

    def print_board(self):
        print("\n  0 1 2")
        for index, row in enumerate(self.board):
            print(str(index), end=" ")
            for value in row:
                if value == 0:
                    print("_", end=" ")
                elif value == 1:
                    print("X", end=" ")
                elif value == 2:
                    print("O", end=" ")
            print()