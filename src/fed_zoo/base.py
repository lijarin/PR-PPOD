import random
import numpy as np


class FedBase:

    def train_step(self):
        raise NotImplementedError

    def validation_step(self):
        raise NotImplementedError

    def fit(self, num_round):
        raise NotImplementedError
