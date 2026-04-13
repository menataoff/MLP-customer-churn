import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ModelMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.loss_history = []
        self.net = nn.Sequential(nn.Linear(input_dim, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 2),
                                 nn.Sigmoid())
        self.opt_dict = {
            "Adam": optim.Adam,
            "RMSprop": optim.RMSprop,
            "SGD": optim.SGD,
            "Adadelta": optim.Adadelta
        }
    def forward(self, x):
        return self.net(x)
    def fit(self, X_train, y_train, opt_name="Adam", epochs=128, batch_size=32, **opt_kwargs):
        data_train = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = self.opt_dict[opt_name](self.parameters(), **opt_kwargs)
        #self.train() #переход в режим тренировки
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for epoch in tqdm(range(epochs), desc='Fitting model'):
            pass


