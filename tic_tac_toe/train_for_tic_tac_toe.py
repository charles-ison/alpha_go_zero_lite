import copy
import torch
import torch.optim as optim
from play_games import play
from games.tic_tac_toe import TicTacToe
from players.player_factory import PlayerFactory
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


def get_game_value(turn_count, result, player_num, best_player):
    if turn_count == 0:
        return 0
    if result == player_num:
        return best_player.win_value
    elif result == 0:
        return best_player.tie_value
    else:
        return best_player.lose_value


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
        probabilities[child.row][child.column] = child.num_visits / (player_node.num_visits - 1)

    if round(probabilities.sum().item(), 5) != 1.0:
        raise Exception("Probabilities do not sum to 1.")

    return probabilities


def get_data_loaders(data, labels):
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size=0.20, shuffle=True)
    print("Number of moves available for training: ", len(training_labels))
    print("Number of moves available for testing: ", len(testing_labels))

    training_data_set = MoveDataSet(training_data, training_labels)
    testing_data_set = MoveDataSet(testing_data, testing_labels)

    training_loader = DataLoader(dataset=training_data_set, batch_size=batch_size, shuffle=True)
    testing_loader = DataLoader(dataset=testing_data_set, batch_size=batch_size, shuffle=True)
    return training_loader, testing_loader


def preprocess_data(results, player_1_roots, player_2_roots, games, best_player):
    data, labels = [], []

    for (result, player_1_node, player_2_node, game) in zip(results, player_1_roots, player_2_roots, games):
        turn_count = 0

        while len(player_1_node.children) != 0 and len(player_2_node.children) != 0:
            player_num = get_player_num(turn_count - 1)
            board_history = game.get_board_history(turn_count, player_num)
            data.append(board_history)

            probabilities = get_probabilities(player_1_node, player_2_node, player_num, game)
            game_value = get_game_value(turn_count, result, player_num, best_player)
            labels.append((probabilities, game_value))

            player_1_node = get_next_move(player_1_node)
            player_2_node = get_next_move(player_2_node)
            turn_count += 1

    return data, labels


def get_best_player(trained_player, best_player, trained_player_win_count, best_player_win_count, log_player_selection):
    total_num_wins = trained_player_win_count + best_player_win_count

    if total_num_wins == 0:
        print("All ties, using the previous best player.")
        return best_player, False

    trained_win_percentage = float(trained_player_win_count) / float(total_num_wins)
    print("Trained player win percentage: ", trained_win_percentage)
    if trained_win_percentage >= 0.55:
        if log_player_selection:
            print("Using the trained player.")
        save_best_player(trained_player)
        return trained_player, True
    else:
        if log_player_selection:
            print("Using the previous best player.")
        return best_player, False


def run_simulations(game, num_simulations, player, num_searches):
    print("\nRunning simulations. . .")
    results = []
    player_1_roots = []
    player_2_roots = []
    games = []
    players = [player, copy.deepcopy(player)]
    for game_num in range(num_simulations):
        new_game = copy.deepcopy(game)
        result_num, player_1_root, player_2_root = play(new_game, players, num_searches, True, False)
        results.append(result_num)
        player_1_roots.append(player_1_root)
        player_2_roots.append(player_2_root)
        games.append(new_game)
    return results, player_1_roots, player_2_roots, games


def evaluate_players(game, num_eval_games, trained_player, best_player, num_searches, log_player_selection):
    trained_player_win_count, best_player_win_count, tie_count = 0, 0, 0
    players = [trained_player, best_player]
    for game_num in range(num_eval_games):
        new_game = copy.deepcopy(game)
        result_num, _, _ = play(new_game, players, num_searches, False, False)
        players[0], players[1] = players[1], players[0]

        if result_num == get_player_num(game_num):
            trained_player_win_count += 1
        elif result_num == get_player_num(game_num + 1):
            best_player_win_count += 1
        elif result_num == 0:
            tie_count += 1

    print("\nNumber of trained player wins: " + str(trained_player_win_count))
    print("Number of previous best player wins: " + str(best_player_win_count))
    print("Number of ties: " + str(tie_count))

    return get_best_player(trained_player, best_player, trained_player_win_count, best_player_win_count, log_player_selection)


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


def train(model, training_loader, testing_loader, device, criterion, epochs, lr):
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
    torch.save(best_player.cnn.state_dict(), "neural_networks/saved_models/tic_tac_toe_cnn.pt")


def join_data(all_data, all_labels, data, labels, max_data_size):
    all_data.extend(data)
    all_labels.extend(labels)

    if len(all_data) > max_data_size:
        return all_data[-max_data_size:], all_labels[-max_data_size:]
    else:
        return all_data, all_labels


def run_reinforcement(num_checkpoints, num_simulations, num_eval_games, epochs, num_searches, lr, max_data_size, num_checkpoints_before_comparison):
    print("Running Reinforcement Learning")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    game = TicTacToe(device)
    criterion = Loss()
    player_factory = PlayerFactory()
    best_player = player_factory.get_player(PlayerType.Untrained_MCTS_CNN, device)
    pure_mcts_player = player_factory.get_player(PlayerType.Raw_MCTS, device)
    all_data, all_labels = [], []
    old_players = [best_player]

    for checkpoint in range(num_checkpoints + 1):
        results, player_1_roots, player_2_roots, games = run_simulations(game, num_simulations, best_player, num_searches)
        data, labels = preprocess_data(results, player_1_roots, player_2_roots, games, best_player)
        all_data, all_labels = join_data(all_data, all_labels, data, labels, max_data_size)
        training_loader, testing_loader = get_data_loaders(all_data, all_labels)
        trained_player = copy.deepcopy(best_player)
        model = trained_player.cnn
        trained_model = train(model, training_loader, testing_loader, device, criterion, epochs, lr)
        trained_player.cnn = trained_model

        print("\nEvaluating players at checkpoint: ", checkpoint)
        best_player, is_new_best_player = evaluate_players(game, num_eval_games, trained_player, best_player, num_searches, True)
        if is_new_best_player:
            old_players.append(copy.deepcopy(best_player))

        if checkpoint % num_checkpoints_before_comparison == 0 and checkpoint != 0:
            for index, player in enumerate(old_players):
                print("\nComparing performance to older player at index: ", index)
                evaluate_players(game, num_eval_games, best_player, player, num_searches, False)
            print("\nComparing performance to Pure MCTS")
            evaluate_players(game, num_eval_games, best_player, pure_mcts_player, num_searches, False)


lr = 0.0001
batch_size = 32
num_searches = 50
num_checkpoints = 100
num_simulations = 50
num_eval_games = 50
num_checkpoints_before_comparison = 10
epochs = 5
max_data_size = 3000

run_reinforcement(num_checkpoints, num_simulations, num_eval_games, epochs, num_searches, lr, max_data_size, num_checkpoints_before_comparison)
