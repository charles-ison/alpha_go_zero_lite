class Node:
    def __init__(self, player, row, column, is_root=False):
        self.player = player
        self.row = row
        self.column = column
        self.is_root = is_root
        self.sub_trees = []
        self.visits = 0
        self.wins = 0

    def add_sub_tree(self, sub_tree):
        self.sub_trees.append(sub_tree)