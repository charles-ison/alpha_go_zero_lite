import copy
import torch
import torch.optim as optim
from play_games import play
from games.TicTacToe import TicTacToe
from players.player import Player
from players.player_type import PlayerType
from utilities import get_player_num
from torch.utils.data import DataLoader
from neural_networks.loss import Loss


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
    print("Bug encountered, no next move found in the MCST")


def get_probabilities(player_1_node, player_2_node, player_num, game):
    player_node = [player_1_node, player_2_node][player_num - 1]
    sum_num_visits = 0.0
    probabilities = torch.zeros(game.board_size, game.board_size)
    for child in player_node.children:
        sum_num_visits += child.num_visits
    for child in player_node.children:
        probabilities[child.row][child.column] = child.num_visits/sum_num_visits
    return probabilities

def build_data_loader(results, player_1_roots, player_2_roots, batch_size, games, alpha_go_zero_lite):
    print("\nBuilding data loader. . .")
    data, labels = [], []
    for (result, player_1_node, player_2_node, game) in zip(results, player_1_roots, player_2_roots, games):
        turn_count = 0
        while len(player_1_node.children) != 0 and len(player_2_node.children) != 0:
            player_num = get_player_num(turn_count)
            board_history = game.get_board_history(turn_count, player_num)
            data.append(board_history)

            probabilities = get_probabilities(player_1_node, player_2_node, player_num, game)
            game_value = get_game_value(result, player_num, alpha_go_zero_lite)
            labels.append((probabilities, game_value))

            player_1_node = get_next_move(player_1_node)
            player_2_node = get_next_move(player_2_node)
            turn_count += 1

    print("Number of moves available for training: ", len(labels))
    training_data_set = MoveDataSet(data, labels)
    return DataLoader(dataset=training_data_set, batch_size=batch_size, shuffle=True)


def get_best_player(trained_player, old_player, trained_player_win_count, old_player_win_count):
    if trained_player_win_count > old_player_win_count:
        print("The trained player had the best performance.")
        return trained_player
    else:
        print("The old player had the best performance.")
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


def train(model, optimizer, data_loader, device):
    print("\nTraining. . .")
    model.train()
    criterion = Loss()
    for batch in data_loader:
        data, prob, value = batch['data'].to(device), batch['label'][0].to(device), batch['label'][1].to(device)
        optimizer.zero_grad()
        predicted_prob, predicted_value = model(data)

        prob = prob.flatten(start_dim=1)
        value = value.float()
        predicted_prob = predicted_prob.flatten(start_dim=1)
        predicted_value = predicted_value.flatten()

        loss = criterion.calculate(predicted_prob, prob, predicted_value, value)
        print("Loss: ", loss.item())
        loss.backward()
        optimizer.step()


batch_size = 2
time_threshold = 4
num_training_steps = 2
num_games = 10
game = TicTacToe()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trained_player = Player(PlayerType.MCTS_CNN)
old_player = Player(PlayerType.MCTS_CNN)

for step in range(num_training_steps):
    print("\nTraining step: ", step)
    best_player, results, player_1_roots, player_2_roots, games = run_simulations(game, num_games, trained_player, old_player, time_threshold)
    trained_player, old_player = copy.deepcopy(best_player), best_player

    model = trained_player.alpha_go_zero_lite.cnn
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    alpha_go_zero_lite = trained_player.alpha_go_zero_lite

    data_loader = build_data_loader(results, player_1_roots, player_2_roots, batch_size, games, alpha_go_zero_lite)
    train(model, optimizer, data_loader, device)

best_player, results, player_1_roots, player_2_roots, games = run_simulations(game, num_games, trained_player, old_player, time_threshold)

