import copy
import torch
import utilities
from games.game import Game


class TicTacToe(Game):

    def __init__(self, device):
        super().__init__(3, 9, 1.0 / 3, 3)
        self.device = device
        self.board_history.append(torch.zeros(self.board_size, self.board_size))
        self.board_history.append(torch.zeros(self.board_size, self.board_size))

    def get_legal_actions(self):
        legal_actions = []
        for row_index, row in enumerate(self.board):
            for column_index, value in enumerate(row):
                if value == 0:
                    legal_actions.append(row_index * self.board_size + column_index)
        return legal_actions

    def apply_action(self, action):
        row, column = divmod(action, self.board_size)
        self.board[row][column] = self.current_player
        self.board_history.append(copy.deepcopy(self.board))
        self.player_history.append(self.current_player)
        self.turn_count += 1
        self.current_player = utilities.get_opposing_player_num(self.current_player)

    # Not the board history is not required for tic-tac-toe, but leaving one move history here to be faithful
    # to the original Alpha Zero paper design
    def encode_state(self, turn_count=None):
        if turn_count is None:
            turn_count = self.turn_count
        last_layer_num = self.get_last_player_num(turn_count)
        board_history = torch.stack((
            self.board_history[turn_count + 1],
            self.board_history[turn_count],
            torch.full((self.board_size, self.board_size), last_layer_num)
        ))
        return board_history.to(self.device)

    def parse_action(self, text):
        try:
            row, column = (int(value) for value in text.split(","))
        except ValueError:
            return None
        if not (0 <= row < self.board_size and 0 <= column < self.board_size):
            return None
        action = row * self.board_size + column
        if action not in self.get_legal_actions():
            return None
        return action

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
