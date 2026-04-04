import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def fit(self, X_train, y_train, opt_name="Adam", epochs=128, batchsize=32, **opt_kwargs):
        self.train() #переход в трежим обучения модели