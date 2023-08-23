from enum import Enum


class PlayerType(Enum):
    Raw_MCTS = 1
    MCTS_CNN = 2
    Untrained_MCTS_CNN = 3
    Manual = 4
