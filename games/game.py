import torch


class Game:
    def __init__(self, board_size, action_space_size, dirichlet_alpha, num_exploration_moves):
        self.board_size = board_size
        self.action_space_size = action_space_size
        self.dirichlet_alpha = dirichlet_alpha
        self.num_exploration_moves = num_exploration_moves
        self.board = torch.zeros(board_size, board_size)
        self.board_history = []
        self.player_history = []
        self.current_player = 1
        self.turn_count = 0

    def get_last_player_num(self, turn_count):
        if turn_count == 0:
            return 0
        return self.player_history[turn_count - 1]

    def get_legal_actions(self):
        raise NotImplementedError("Must override get_legal_actions().")

    def apply_action(self, action):
        raise NotImplementedError("Must override apply_action().")

    def encode_state(self, turn_count=None):
        raise NotImplementedError("Must override encode_state().")

    def parse_action(self, text):
        raise NotImplementedError("Must override parse_action().")

    def detect_winner(self):
        raise NotImplementedError("Must override detect_winner().")

    def detect_tie(self):
        raise NotImplementedError("Must override detect_tie().")

    def print_board(self):
        raise NotImplementedError("Must override print_board().")
