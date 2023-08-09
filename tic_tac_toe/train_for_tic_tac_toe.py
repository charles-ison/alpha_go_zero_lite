import copy
import torch
import torch.optim as optim
from play_games import play
from games.TicTacToe import TicTacToe
from players.player import Player
from players.player_type import PlayerType
from utilities import get_player_num
from torch.utils.data import DataLoader


class MoveDataSet(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'data': self.data[index], 'label': self.labels[index]}


def get_game_value(result, player_num, alpha_go_zero_lite):
    if result == player_num:
        return alpha_go_zero_lite.win_value
    elif result == 0:
        return alpha_go_zero_lite.tie_value
    else:
        return alpha_go_zero_lite.lose_value

def get_next_move(node):
    for child in node.children:
        if child.was_played:
            return child
    print("Bug encountered, no played move in MCST")

def build_data_loader(results, player_1_roots, player_2_roots, batch_size, games, alpha_go_zero_lite):
    print("\nBuilding data loader. . .")
    data, labels = [], []
    for (result, player_1_node, player_2_node, game) in zip(results, player_1_roots, player_2_roots, games):
        turn_count = 0
        while len(player_1_node.children) != 0 and len(player_2_node.children) != 0:
            print("hi")
            player_num = get_player_num(turn_count)
            game_value = get_game_value(result, player_num, alpha_go_zero_lite)
            board_history = game.board_histories[turn_count]
            data.append(board_history)

            #TODO: Add probabilities here
            labels.append((game_value, []))
            player_1_node = get_next_move(player_1_node)
            player_2_node = get_next_move(player_2_node)
            turn_count += 1
    training_data_set = MoveDataSet(data, labels)
    return DataLoader(dataset=training_data_set, batch_size=batch_size, shuffle=True)


def get_best_player(trained_player, old_player, trained_player_win_count, old_player_win_count):
    if trained_player_win_count > old_player_win_count:
        return trained_player
    else:
        return old_player


def swap_players(players):
    temp = players[0]
    players[0] = players[1]
    players[1] = temp


def run_simulations(game, num_games, trained_player, old_player, time_threshold):
    trained_player_win_count, old_player_win_count, tie_count = 0, 0, 0
    results = []
    player_1_roots = []
    player_2_roots = []
    games = []
    players = [trained_player, old_player]
    print("\nRunning simulations. . .")
    for game_num in range(num_games):
        new_game = copy.deepcopy(game)
        result_num, player_1_root, player_2_root = play(new_game, players, time_threshold, False)
        results.append(result_num)
        player_1_roots.append(player_1_root)
        player_2_roots.append(player_2_root)
        swap_players(players)
        games.append(new_game)
        print("\nGame " + str(game_num + 1) + " finished!")

        if result_num == get_player_num(game_num):
            trained_player_win_count += 1
            print("Trained player won.")
        elif result_num == get_player_num(game_num + 1):
            old_player_win_count += 1
            print("Old player won.")
        elif result_num == 0:
            tie_count += 1
            print("The result was a tie.")

    print("\nNumber of trained player wins: " + str(trained_player_win_count))
    print("Number of old player wins: " + str(old_player_win_count))
    print("Number of ties: " + str(tie_count))

    best_player = get_best_player(trained_player, old_player, trained_player_win_count, old_player_win_count)
    return best_player, results, player_1_roots, player_2_roots, games


def train(model, optimizer, training_loader, device):
    print("\nTraining. . .")
    model.train()
    for batch in training_loader:
        data, labels = batch['data'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        output = model(data)


batch_size = 10
time_threshold = 4
num_games = 2
game = TicTacToe()

trained_player = Player(PlayerType.MCTS_CNN)
old_player = Player(PlayerType.MCTS_CNN)
best_player, results, player_1_roots, player_2_roots, games = run_simulations(game, num_games, trained_player, old_player, time_threshold)
model = best_player.alpha_go_zero_lite.cnn
optimizer = optim.Adam(model.parameters(), lr=0.01)
alpha_go_zero_lite = trained_player.alpha_go_zero_lite
training_loader = build_data_loader(results, player_1_roots, player_2_roots, batch_size, games, alpha_go_zero_lite)
train(model, optimizer, training_loader)

