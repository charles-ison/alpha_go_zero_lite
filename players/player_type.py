from enum import Enum


class PlayerType(Enum):
    Pure_MCTS = 1
    MCTS_CNN = 2
    Untrained_MCTS_CNN = 3
    Manual = 4
