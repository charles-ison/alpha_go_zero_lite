import torch
from games.game import Game


class Go(Game):
    def __init__(self):
        super().__init__(9)
        self.board_history.append(torch.zeros(self.board_size, self.board_size))
        self.board_history.append(torch.zeros(self.board_size, self.board_size))
        self.board_history.append(torch.zeros(self.board_size, self.board_size))
        self.board_history.append(torch.zeros(self.board_size, self.board_size))
