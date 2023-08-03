import torch.nn as nn


class Loss:
    def __init__(self):
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def calculate(self, predicted_val, actual_val, predicted_prob, actual_prob):
        mse = self.mse(predicted_val, actual_val)
        cross_entropy_loss = self.cross_entropy_loss(predicted_prob, actual_prob)
        return mse + cross_entropy_loss
