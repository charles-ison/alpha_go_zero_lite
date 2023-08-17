from alpha_go_zero.alpha_go_zero import AlphaGoZero


class AlphaGoZeroCNN(AlphaGoZero):

    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def get_selection_move(self, last_expansion_move_children):
        best_move = last_expansion_move_children[0]
        for move in last_expansion_move_children[1:]:
            if move.get_mean_action_value_plus_puct() > best_move.get_mean_action_value_plus_puct():
                best_move = move
        return best_move

    def should_stop_rollout(self, is_simulation):
        return is_simulation

    def get_expanded_node_number_of_visits(self):
        return 0

    def get_action_values(self, win_detected, tie_detected, mcts_game, expansion_move, turn_count, player_num):
        if win_detected or tie_detected:
            return self.get_finished_game_values(win_detected)
        else:
            board_history = mcts_game.get_board_history(turn_count, player_num)
            self.cnn.eval()
            child_probabilities, action_value = self.cnn(board_history.unsqueeze(dim=0))
            expansion_move.child_probabilities = child_probabilities.squeeze()
            return action_value, -action_value
