import utilities
from players.player import Player


class ManualPlayer(Player):
    def play_move(self, game, player_num, turn_count, last_move, num_searches, add_noise, print_games):
        row, column = input("\nPlayer " + str(player_num) + " please enter move coordinates: ").split(",")
        row, column = int(row), int(column)

        if game.is_valid_move(row, column):
            return utilities.append_move(game, player_num, last_move, row, column)
        else:
            print("\nInvalid move. Please try again.")
            return self.play_move(game, player_num, turn_count, last_move, num_searches, add_noise, print_games)
