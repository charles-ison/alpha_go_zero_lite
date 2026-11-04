import utilities
from players.player import Player


class ManualPlayer(Player):
    def play_move(self, game, last_move, time_limit, add_noise, print_games):
        action = game.parse_action(input("\nPlayer " + str(game.current_player) + " please enter move coordinates: "))

        if action is not None:
            return utilities.append_move(game, game.current_player, last_move, action)
        else:
            print("\nInvalid move. Please try again.")
            return self.play_move(game, last_move, time_limit, add_noise, print_games)
