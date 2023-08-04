import copy
import time
import utilities
import random
import monte_carlo_tree as mct


class AlphaGoZeroLite:

    def get_move(self, turn_count, game, last_move, time_threshold):
        self.run_monte_carlo_tree_search(turn_count, game, last_move, time_threshold)
        potential_moves = last_move.children
        if len(potential_moves) == 0:
            print("Bug encountered, no potential AlphaGo Zero Lite moves found. More searches need to be run.")

        best_move = potential_moves[0]
        for move in potential_moves[1:]:
            if move.num_visits > best_move.num_visits:
                best_move = move
        return best_move

    def run_monte_carlo_tree_search(self, turn_count, game, last_move, time_threshold):
        mcts_game = copy.deepcopy(game)
        mcts_move = last_move
        expansion_move = None
        mcts_turn_count = turn_count
        num_searches = 0
        time_limit = time.time() + time_threshold
        start_time = time.time()
        while time.time() < time_limit:
            mcts_player_num = utilities.get_player_num(mcts_turn_count)
            mcts_move, expansion_move = self.get_next_mcts_move(mcts_game, mcts_player_num, mcts_move, expansion_move)
            mcts_game.update_board(mcts_move.row, mcts_move.column, mcts_player_num)
            mcts_turn_count += 1

            win_detected, tie_detected = mcts_game.detect_winner(), mcts_game.detect_tie()
            if win_detected or tie_detected:
                current_player_points, opposing_player_points = self.get_num_points(win_detected)
                points_dict = {
                    mcts_player_num: current_player_points,
                    mcts_move.parent.player_num: opposing_player_points
                }
                backpropagation_leaf = self.get_backpropagation_leaf(expansion_move, mcts_move)
                self.run_backpropagation(backpropagation_leaf, last_move, points_dict)
                mcts_game = copy.deepcopy(game)
                expansion_move = None
                mcts_move = last_move
                mcts_turn_count = turn_count
                num_searches += 1

        stop_time = time.time()
        run_time = int(stop_time - start_time)
        print("AlphaGo Zero Lite ran " + str(num_searches) + " searches in " + str(run_time) + " seconds.")

    def get_next_mcts_move(self, mcts_game, mcts_player_num, last_mcts_move, expansion_move):
        raise NotImplementedError("Must override get_next_mcts_move().")

    def run_backpropagation(self, backpropagation_leaf, mcts_root, points_dict):
        mcts_root.num_visits += 1.0
        backprop_move = backpropagation_leaf
        while backprop_move != mcts_root:
            backprop_move.num_points += points_dict[backprop_move.player_num]
            backprop_move.num_visits += 1.0
            backprop_move = backprop_move.parent

    def get_backpropagation_leaf(self, expansion_move, mcts_move):
        if expansion_move is None:
            return mcts_move
        else:
            return expansion_move

    def should_run_selection_move(self, potential_moves, last_mcts_move_children):
        return len(potential_moves) == len(last_mcts_move_children) and self.all_children_visited(last_mcts_move_children)

    def get_selection_move(self, last_expansion_move_children):
        best_move = last_expansion_move_children[0]
        for move in last_expansion_move_children[1:]:
            if move.get_upper_confidence_bound() > best_move.get_upper_confidence_bound():
                best_move = move
        return best_move

    def get_simulation_move(self, mcts_player_num, potential_move_tuples, last_mcts_move):
        unexplored_potential_tuples = [tuple for tuple in potential_move_tuples if
                                       self.move_unexplored(tuple, last_mcts_move.children)]
        random_move_tuple = unexplored_potential_tuples[random.randint(0, len(unexplored_potential_tuples) - 1)]
        return mct.Move(mcts_player_num, random_move_tuple[0], random_move_tuple[1], last_mcts_move)

    def move_unexplored(self, potential_move_tuple, last_mcts_move_children):
        potential_row = potential_move_tuple[0]
        potential_column = potential_move_tuple[1]
        for child in last_mcts_move_children:
            if child.row == potential_row and child.column == potential_column:
                return False
        return True

    # TODO: Figure out why this check is needed, suggests there is a bug somewhere
    def all_children_visited(self, last_mcts_move_children):
        for child in last_mcts_move_children:
            if child.num_visits == 0.0:
                return False
        return True

    def get_num_points(self, win_detected):
        if win_detected:
            return 3, -3
        else:
            return 1, 1