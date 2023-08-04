from neural_networks.tic_tac_toe_cnn import TicTacToeCNN
from alpha_go_zero_lite import AlphaGoZeroLite

class AlphaGoZeroLiteCNN(AlphaGoZeroLite):

    def __init__(self):
        super().__init__()
        self.cnn = TicTacToeCNN()

    def get_next_mcts_move(self, mcts_game, mcts_player_num, last_mcts_move, expansion_move):
        policy, value = self.cnn(mcts_game.board_history)

        potential_moves = mcts_game.fetch_potential_moves()
        if self.should_run_selection_move(potential_moves, last_mcts_move.children):
            return self.get_selection_move(last_mcts_move.children), expansion_move
        else:
            next_mcts_move = self.get_simulation_move(mcts_player_num, potential_moves, last_mcts_move)
            if expansion_move == None:
                expansion_move = next_mcts_move
                last_mcts_move.children.append(expansion_move)
            return next_mcts_move, expansion_move