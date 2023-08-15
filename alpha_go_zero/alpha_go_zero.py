import copy
import time
import utilities
import random
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
        while time.time() < time_limit:
            mcts_player_num = utilities.get_player_num(mcts_turn_count)
            next_mcts_move, is_simulation = self.get_next_mcts_move(mcts_game, mcts_player_num, mcts_move)
            if is_simulation and expansion_move is None:
                expansion_move = next_mcts_move
                mcts_move.children.append(expansion_move)

            mcts_move = next_mcts_move
            mcts_game.update_board(mcts_move.row, mcts_move.column, mcts_player_num)
            mcts_turn_count += 1

            win_detected, tie_detected = mcts_game.detect_winner(), mcts_game.detect_tie()
            if win_detected or tie_detected or self.should_stop_rollout(is_simulation):
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
        if self.should_run_selection_move(potential_moves, last_mcts_move.children):
            return self.get_selection_move(last_mcts_move.children), False
        else:
            return self.get_simulation_move(mcts_game, mcts_player_num, potential_moves, last_mcts_move), True

    def run_backpropagation(self, backpropagation_leaf, mcts_root, action_value_dict):
        backprop_move = backpropagation_leaf
        while backprop_move != mcts_root:
            backprop_move.action_value += action_value_dict[backprop_move.player_num]
            backprop_move.num_visits += 1.0
            backprop_move = backprop_move.parent

    def get_backpropagation_leaf(self, expansion_move, mcts_move):
        if expansion_move is None:
            return mcts_move
        else:
            return expansion_move

    def should_run_selection_move(self, potential_moves, last_mcts_move_children):
        return len(potential_moves) == len(last_mcts_move_children)

    def get_selection_move(self, last_expansion_move_children):
        raise NotImplementedError("Must override get_selection_move().")

    def should_stop_rollout(self, is_simulation):
        raise NotImplementedError("Must override should_not_perform_rollout().")

    def save_game_analysis(self, mcts_game, expansion_move):
        raise NotImplementedError("Must override save_game_analysis().")

    def get_simulation_move(self, mcts_game, mcts_player_num, potential_move_tuples, last_mcts_move):
        unexplored_potential_tuples = [tuple for tuple in potential_move_tuples if
                                       self.move_unexplored(tuple, last_mcts_move.children)]
        random_move_tuple = unexplored_potential_tuples[random.randint(0, len(unexplored_potential_tuples) - 1)]
        row = random_move_tuple[0]
        column = random_move_tuple[1]
        return mct.Move(mcts_game.board_size, mcts_player_num, row, column, last_mcts_move)

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