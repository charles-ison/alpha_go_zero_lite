import copy
import time
import utilities
import monte_carlo_tree as mct


class AlphaGoZero:

    def __init__(self):
        super().__init__()
        self.win_value = 1
        self.lose_value = -1
        self.tie_value = 0

    def get_move(self, turn_count, game, last_move, time_threshold, print_games):
        self.run_monte_carlo_tree_search(turn_count, game, last_move, time_threshold, print_games)
        potential_moves = last_move.children
        if len(potential_moves) == 0:
            print("Bug encountered, no potential AlphaGo Zero Lite moves found. More searches need to be run.")
            game.print_board()
            print(last_move.row)
            print(last_move.column)
            print(last_move.player_num)
            print(turn_count)

        best_move = potential_moves[0]
        for move in potential_moves[1:]:
            if move.num_visits > best_move.num_visits:
                best_move = move
        return best_move

    def run_monte_carlo_tree_search(self, turn_count, game, last_move, time_threshold, print_games):
        mcts_game = copy.deepcopy(game)
        mcts_move = last_move
        expansion_move = None
        mcts_turn_count = turn_count
        num_searches = 0
        time_limit = time.time() + time_threshold
        start_time = time.time()
        while time.time() < time_limit or len(last_move.children) == 0:
            mcts_player_num = utilities.get_player_num(mcts_turn_count)
            next_mcts_move, tree_expanded = self.get_next_mcts_move(mcts_game, mcts_player_num, mcts_move)
            if tree_expanded and expansion_move is None:
                expansion_move = next_mcts_move

            mcts_move = next_mcts_move
            mcts_game.update_board(mcts_move.row, mcts_move.column, mcts_player_num)
            mcts_turn_count += 1

            win_detected, tie_detected = mcts_game.detect_winner(), mcts_game.detect_tie()
            if win_detected or tie_detected or self.should_stop_rollout(tree_expanded):
                current_val, opposing_val = self.get_action_values(win_detected, tie_detected, mcts_game, expansion_move, mcts_turn_count, mcts_player_num)
                action_value_dict = {
                    mcts_player_num: current_val,
                    utilities.get_opposing_player_num(mcts_player_num): opposing_val
                }
                backpropagation_leaf = self.get_backpropagation_leaf(expansion_move, mcts_move)
                self.run_backpropagation(backpropagation_leaf, last_move, action_value_dict)
                mcts_game = copy.deepcopy(game)
                expansion_move = None
                mcts_move = last_move
                mcts_turn_count = turn_count
                num_searches += 1

        stop_time = time.time()
        run_time = int(stop_time - start_time)
        if print_games:
            print("AlphaGo Zero Lite ran " + str(num_searches) + " searches in " + str(run_time) + " seconds.")

    def get_next_mcts_move(self, mcts_game, mcts_player_num, last_mcts_move):
        potential_moves = mcts_game.fetch_potential_moves()
        should_expand_tree = self.should_expand_tree(potential_moves, last_mcts_move.children)
        if should_expand_tree:
            self.expand_tree(potential_moves, last_mcts_move, mcts_game, mcts_player_num)
        return self.get_selection_move(last_mcts_move.children), should_expand_tree

    def run_backpropagation(self, backpropagation_leaf, mcts_root, action_value_dict):
        backprop_move = backpropagation_leaf
        while backprop_move != mcts_root:
            backprop_move.action_value += action_value_dict[backprop_move.player_num]
            backprop_move.num_visits += 1.0
            backprop_move.mean_action_value = backprop_move.action_value / backprop_move.num_visits
            backprop_move = backprop_move.parent

    def get_backpropagation_leaf(self, expansion_move, mcts_move):
        if expansion_move is None:
            return mcts_move
        else:
            return expansion_move

    def should_expand_tree(self, potential_moves, last_mcts_move_children):
        return len(potential_moves) != len(last_mcts_move_children)

    def expand_tree(self, potential_moves, last_mcts_move, mcts_game, mcts_player_num):
        last_mcts_move_children = last_mcts_move.children
        for potential_move_tuple in potential_moves:
            if self.move_unexplored(potential_move_tuple, last_mcts_move_children):
                board_size = mcts_game.board_size
                row = potential_move_tuple[0]
                column = potential_move_tuple[1]
                num_visits = self.get_expanded_node_number_of_visits()
                potential_move = mct.Move(board_size, mcts_player_num,  row, column, last_mcts_move, num_visits)
                last_mcts_move_children.append(potential_move)

    def get_expanded_node_number_of_visits(self):
        raise NotImplementedError("Must override get_expanded_node_number_of_visits().")

    def get_selection_move(self, last_expansion_move_children):
        raise NotImplementedError("Must override get_selection_move().")

    def should_stop_rollout(self, is_simulation):
        raise NotImplementedError("Must override should_not_perform_rollout().")

    def save_game_analysis(self, mcts_game, expansion_move):
        raise NotImplementedError("Must override save_game_analysis().")

    def move_unexplored(self, potential_move_tuple, last_mcts_move_children):
        potential_row = potential_move_tuple[0]
        potential_column = potential_move_tuple[1]
        for child in last_mcts_move_children:
            if child.row == potential_row and child.column == potential_column:
                return False
        return True

    def get_action_values(self, win_detected, tie_detected, mcts_game, expansion_move, turn_count, player_num):
        raise NotImplementedError("Must override get_action_value().")

    def get_finished_game_values(self, win_detected):
        if win_detected:
            return self.win_value, self.lose_value
        else:
            return self.tie_value, self.tie_value