import torch
from games.game import Game


class TicTacToe(Game):

    def __init__(self):
        super().__init__(3)

    def get_board_history(self, turn_count, player_num):
        return torch.stack((
            self.board_history[turn_count + 2],
            self.board_history[turn_count + 1],
            self.board_history[turn_count],
            self.get_current_player_board_representation(player_num)
        ))

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