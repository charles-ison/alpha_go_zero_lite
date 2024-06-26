from players.player import Player
import copy
import time
import utilities
import monte_carlo_tree as mct
import random


class AlphaGoZeroPlayer(Player):

    def __init__(self):
        super().__init__()
        self.win_value = 1
        self.lose_value = -1
        self.tie_value = 0

    def play_move(self, game, player_num, turn_count, last_move, time_limit, add_noise, print_games):
        self.run_mcts(turn_count, game, last_move, time_limit, add_noise, print_games)
        potential_moves = last_move.children
        return self.get_next_move(potential_moves, game, turn_count, add_noise)

    def get_most_visited_potential_move(self, potential_moves):
        best_move = potential_moves[0]
        tie_moves = [best_move]
        is_tie = False
        for move in potential_moves[1:]:
            if move.num_visits > best_move.num_visits:
                best_move = move
                tie_moves = [best_move]
                is_tie = False
            elif move.num_visits == best_move.num_visits:
                tie_moves.append(move)
                is_tie = True

        if is_tie:
            return self.get_random_move(tie_moves)
        return best_move


    def run_mcts(self, turn_count, game, last_move, time_limit, add_noise, print_games):
        mcts_game = copy.deepcopy(game)
        mcts_move = last_move
        mcts_turn_count = turn_count
        searches_count = 0
        expansion_move_performed = False
        start_time = time.time()
        mcts_player_num = utilities.get_player_num(mcts_turn_count)
        self.initialize_run_mcts(mcts_move, mcts_turn_count, mcts_game, mcts_player_num)
        while time.time() < start_time + time_limit or searches_count < 1:
            mcts_player_num = utilities.get_player_num(mcts_turn_count)
            mcts_move = self.get_next_mcts_move(mcts_game, mcts_player_num, mcts_move, expansion_move_performed, add_noise)
            mcts_game.update_board(mcts_move.row, mcts_move.column, mcts_player_num)
            mcts_turn_count += 1

            win_detected = mcts_game.detect_winner()
            tie_detected = mcts_game.detect_tie()
            if mcts_move.num_visits == 0:
                expansion_move_performed = True
            if win_detected or tie_detected or self.should_stop_rollout(expansion_move_performed):
                current_val, opposing_val = self.get_action_values(win_detected, tie_detected, mcts_game, mcts_move, mcts_turn_count, mcts_player_num)
                action_value_dict = {
                    mcts_player_num: current_val,
                    utilities.get_opposing_player_num(mcts_player_num): opposing_val
                }
                self.run_backpropagation(mcts_move, last_move, action_value_dict)
                mcts_game = copy.deepcopy(game)
                mcts_move = last_move
                mcts_turn_count = turn_count
                expansion_move_performed = False
                searches_count += 1

        stop_time = time.time()
        run_time = float(stop_time - start_time)
        rounded_run_time = round(run_time, 3)
        if print_games:
            print("\nAlphaGo Zero Lite ran " + str(searches_count) + " searches in " + str(rounded_run_time) + " seconds.")

    def initialize_run_mcts(self, mcts_move, mcts_turn_count, mcts_game, mcts_player_num):
        raise NotImplementedError("Must override initialize_run_mcts().")

    def get_next_move(self, potential_moves, game, turn_count, add_noise):
        raise NotImplementedError("Must override get_next_move().")

    def get_next_mcts_move(self, mcts_game, mcts_player_num, last_mcts_move, expansion_move_performed, add_noise):
        potential_moves = mcts_game.fetch_potential_moves()
        if self.should_add_new_tree_layer(potential_moves, last_mcts_move.children):
            self.add_new_tree_layer(potential_moves, last_mcts_move, mcts_game, mcts_player_num)
        if expansion_move_performed:
            return self.get_random_move(last_mcts_move.children)
        return self.get_selection_move(last_mcts_move.children, add_noise)

    def run_backpropagation(self, backpropagation_leaf, mcts_root, action_value_dict):
        backprop_move = backpropagation_leaf
        mcts_root.num_visits += 1.0
        while backprop_move != mcts_root:
            backprop_move.action_value += action_value_dict[backprop_move.player_num]
            backprop_move.num_visits += 1.0
            backprop_move = backprop_move.parent

    def should_add_new_tree_layer(self, potential_moves, last_mcts_move_children):
        return len(potential_moves) != len(last_mcts_move_children)

    def should_stop_rollout(self, expansion_move_performed):
        raise NotImplementedError("Must override should_stop_rollout().")

    def add_new_tree_layer(self, potential_moves, last_mcts_move, mcts_game, mcts_player_num):
        last_mcts_move_children = last_mcts_move.children
        for potential_move_tuple in potential_moves:
            if self.move_unexplored(potential_move_tuple, last_mcts_move_children):
                board_size = mcts_game.board_size
                row = potential_move_tuple[0]
                column = potential_move_tuple[1]
                potential_move = mct.Move(board_size, mcts_player_num,  row, column, last_mcts_move, 0)
                last_mcts_move_children.append(potential_move)

    def move_unexplored(self, potential_move_tuple, last_mcts_move_children):
        potential_row = potential_move_tuple[0]
        potential_column = potential_move_tuple[1]
        for child in last_mcts_move_children:
            if child.row == potential_row and child.column == potential_column:
                return False
        return True

    def get_action_values(self, win_detected, tie_detected, mcts_game, mcts_move, turn_count, player_num):
        raise NotImplementedError("Must override get_action_value().")

    def get_finished_game_values(self, win_detected):
        if win_detected:
            return self.win_value, self.lose_value
        else:
            return self.tie_value, self.tie_value

    def get_selection_value(self, move, add_noise):
        raise NotImplementedError("Must override get_selection_value().")

    def get_random_move(self, potential_moves):
        return random.choices(potential_moves)[0]

    def get_selection_move(self, last_expansion_move_children, add_noise):
        best_move = last_expansion_move_children[0]
        tie_moves = [best_move]
        is_tie = False
        for move in last_expansion_move_children[1:]:
            move_selection_value = self.get_selection_value(move, add_noise)
            best_move_selection_value = self.get_selection_value(best_move, add_noise)
            if move_selection_value > best_move_selection_value:
                best_move = move
                tie_moves = [best_move]
                is_tie = False
            elif move_selection_value == best_move_selection_value:
                tie_moves.append(move)
                is_tie = True

        if is_tie:
            return self.get_random_move(tie_moves)
        return best_move
