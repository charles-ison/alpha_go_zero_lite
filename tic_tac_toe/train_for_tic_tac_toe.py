import copy
import torch
import torch.optim as optim
from play_games import play
from games.tic_tac_toe import TicTacToe
from players.player import Player
from players.player_type import PlayerType
from utilities import get_player_num
from torch.utils.data import DataLoader
from neural_networks.loss import Loss
from sklearn.model_selection import train_test_split


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


def get_player_node(player_1_node, player_2_node, player_num):
    if player_num == 1:
        return player_2_node
    else:
        return player_1_node


def get_probabilities(player_1_node, player_2_node, player_num, game):
    player_node = get_player_node(player_1_node, player_2_node, player_num)
    probabilities = torch.zeros(game.board_size, game.board_size)
    for child in player_node.children:
        probabilities[child.row][child.column] = child.num_visits/player_node.parent.num_visits
    return probabilities


def get_data_loaders(data, labels):
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size=0.20)
    print("Number of moves available for training: ", len(training_labels))
    print("Number of moves available for testing: ", len(testing_labels))

    training_data_set = MoveDataSet(training_data, training_labels)
    testing_data_set = MoveDataSet(testing_data, testing_labels)

    training_loader = DataLoader(dataset=training_data_set, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(dataset=testing_data_set, batch_size=batch_size, shuffle=True)

    return training_loader, testing_loader

def preprocess_data(results, player_1_roots, player_2_roots, batch_size, games, alpha_go_zero_lite):
    print("\nBuilding data loader. . .")
    data, labels = [], []

    for (result, player_1_node, player_2_node, game) in zip(results, player_1_roots, player_2_roots, games):
        turn_count = 0

        # Skipping roots
        player_1_node = get_next_move(player_1_node)
        player_2_node = get_next_move(player_2_node)

        while len(player_1_node.children) != 0 and len(player_2_node.children) != 0:
            player_num = get_player_num(turn_count)
            board_history = game.get_board_history(turn_count + 1, player_num)
            data.append(board_history)

            probabilities = get_probabilities(player_1_node, player_2_node, player_num, game)
            game_value = get_game_value(result, player_num, alpha_go_zero_lite)
            labels.append((probabilities, game_value))

            player_1_node = get_next_move(player_1_node)
            player_2_node = get_next_move(player_2_node)
            turn_count += 1

    return data, labels


def get_best_player(trained_player, best_player, trained_player_win_count, best_player_win_count, log_player_selection):
    total_num_wins = trained_player_win_count + best_player_win_count
    trained_win_percentage = float(trained_player_win_count) / float(total_num_wins)
    print("Trained player win percentage: ", trained_win_percentage)
    if trained_win_percentage >= 0.55:
        if log_player_selection:
            print("Using the trained player.")
        return trained_player
    else:
        if log_player_selection:
            print("Using the previous best player.")
        return best_player


def swap_players(players):
    temp = players[0]
    players[0] = players[1]
    players[1] = temp


def run_simulations(game, num_simulations, player, time_threshold):
    print("\nRunning simulations. . .")
    results = []
    player_1_roots = []
    player_2_roots = []
    games = []
    players = [player, copy.deepcopy(player)]
    for game_num in range(num_simulations):
        new_game = copy.deepcopy(game)
        result_num, player_1_root, player_2_root = play(new_game, players, time_threshold, False)
        results.append(result_num)
        player_1_roots.append(player_1_root)
        player_2_roots.append(player_2_root)
        games.append(new_game)
    return results, player_1_roots, player_2_roots, games


def evaluate_players(game, num_eval_games, trained_player, best_player, time_threshold, log_player_selection):
    print("\nEvaluating players. . .")
    trained_player_win_count, best_player_win_count, tie_count = 0, 0, 0
    players = [trained_player, best_player]
    for game_num in range(num_eval_games):
        new_game = copy.deepcopy(game)
        result_num, _, _ = play(new_game, players, time_threshold, False)
        swap_players(players)

        if result_num == get_player_num(game_num):
            trained_player_win_count += 1
        elif result_num == get_player_num(game_num + 1):
            best_player_win_count += 1
        elif result_num == 0:
            tie_count += 1

    print("\nNumber of trained player wins: " + str(trained_player_win_count))
    print("Number of previous best player wins: " + str(best_player_win_count))
    print("Number of ties: " + str(tie_count))

    best_player = get_best_player(trained_player, best_player, trained_player_win_count, best_player_win_count, log_player_selection)
    return best_player


def test(model, testing_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    for batch in testing_loader:
        data, prob, value = batch['data'].to(device), batch['label'][0].to(device), batch['label'][1].to(device)
        predicted_prob, predicted_value = model(data)
        prob = prob.flatten(start_dim=1)
        value = value.float()
        predicted_prob = predicted_prob.flatten(start_dim=1)
        predicted_value = predicted_value.flatten()
        loss = criterion.calculate(predicted_prob, prob, predicted_value, value)
        total_loss += loss.item()
    return total_loss


def train_model(model, training_loader, testing_loader, device, criterion, epochs, lr):
    print("\nTraining. . .")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    old_testing_loss = None
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        total_training_loss = 0.0
        old_model = copy.deepcopy(model)
        for batch in training_loader:
            data, prob, value = batch['data'].to(device), batch['label'][0].to(device), batch['label'][1].to(device)
            optimizer.zero_grad()
            predicted_prob, predicted_value = model(data)

            prob = prob.flatten(start_dim=1)
            value = value.float()
            predicted_prob = predicted_prob.flatten(start_dim=1)
            predicted_value = predicted_value.flatten()

            loss = criterion.calculate(predicted_prob, prob, predicted_value, value)
            total_training_loss += loss.item()
            loss.backward()
            optimizer.step()

        testing_loss = test(model, testing_loader, device, criterion)
        print("Training Loss: " + str(total_training_loss) + " and Testing Loss: " + str(testing_loss))
        if old_testing_loss is not None and testing_loss >= old_testing_loss:
            print("Testing loss increasing, using model from one iteration earlier.")
            return old_model
        old_testing_loss = testing_loss
    return model


def save_best_player(best_player):
    torch.save(best_player.alpha_go_zero_lite.cnn.state_dict(), "tic_tac_toe_cnn.pt")


def join_data(all_data, all_labels, data, labels):
    max_data_size = 3000

    all_data.extend(data)
    all_labels.extend(labels)

    if len(all_data) > max_data_size:
        return all_data[-max_data_size:], all_labels[-max_data_size:]
    else:
        return all_data, all_labels


def run_reinforcement(num_checkpoints, game, device, criterion, num_simulations, num_eval_games, epochs, time_threshold, lr):
    print("Running Reinforcement Learning")
    best_player = Player(PlayerType.Untrained_MCTS_CNN)
    untrained_player = Player(PlayerType.Untrained_MCTS_CNN)
    pure_mcts_player = Player(PlayerType.Pure_MCTS)
    all_data, all_labels = [], []

    for step in range(num_checkpoints):
        results, player_1_roots, player_2_roots, games = run_simulations(game, num_simulations, best_player, time_threshold)
        data, labels = preprocess_data(results, player_1_roots, player_2_roots, batch_size, games, best_player.alpha_go_zero_lite)
        all_data, all_labels = join_data(all_data, all_labels, data, labels)
        training_loader, testing_loader = get_data_loaders(all_data, all_labels)
        trained_player = copy.deepcopy(best_player)
        model = trained_player.alpha_go_zero_lite.cnn
        trained_model = train_model(model, training_loader, testing_loader, device, criterion, epochs, lr)
        trained_player.alpha_go_zero_lite.cnn = trained_model
        best_player = evaluate_players(game, num_eval_games, trained_player, best_player, time_threshold, True)
        save_best_player(best_player)

        print("\nCheckpoint number: ", step)
        if step % 10 == 0:
            print("\nComparing performance to untrained CNN")
            evaluate_players(game, num_eval_games, best_player, untrained_player, time_threshold, False)
            print("\nComparing performance to Pure MCTS")
            evaluate_players(game, num_eval_games, best_player, pure_mcts_player, time_threshold, False)


lr = 0.0001
batch_size = 64
time_threshold = 0.05
num_checkpoints = 100
num_simulations = 100
num_eval_games = 50
epochs = 5
game = TicTacToe()
criterion = Loss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_reinforcement(num_checkpoints, game, device, criterion, num_simulations, num_eval_games, epochs, time_threshold, lr)
