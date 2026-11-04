import argparse
import torch
from games.game_type import GameType, GameFactory
from players.player_factory import PlayerFactory
from players.player_type import PlayerType
from game_runner import play_match


def get_player(player_factory, player_num, game_type, device):
    player_type = get_player_type(player_num)
    return player_factory.get_player(player_type, game_type, device)


def get_player_type(player_num):
    print("Please select Player " + str(player_num) + " type.")
    print("1. AlphaGo Zero Lite with pure Monte Carlo Tree Search")
    print("2. AlphaGo Zero Lite with CNN")
    print("3. AlphaGo Zero Lite with Untrained CNN")
    print("4. AlphaGo Zero Lite with pure probabilities")
    print("5. Manual player")
    return PlayerType(int(input("Please enter your selection: ")))


def get_num_games():
    return int(input("\nHow many games would you like to play? "))


def parse_args():
    parser = argparse.ArgumentParser(description="Play games between AlphaGo Zero Lite players.")
    parser.add_argument("--game", type=GameType, choices=list(GameType), metavar="{" + ",".join(game_type.value for game_type in GameType) + "}")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # About 1-2 seconds is required for pure MCTS to achieve perfect play
    time_limit = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    player_factory = PlayerFactory()
    players = [get_player(player_factory, 1, args.game, device), get_player(player_factory, 2, args.game, device)]
    num_games = get_num_games()
    game_factory = GameFactory()
    game = game_factory.get_game(args.game, device)
    play_match(game, num_games, players, time_limit)
