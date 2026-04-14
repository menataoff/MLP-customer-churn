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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.loss_history = []
        self.val_loss_history = []
        self.net = nn.Sequential(nn.Linear(input_dim, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1)) #почему лучше без сигмоиды? потому что численно стабильнее
        self.opt_dict = {
            "Adam": optim.Adam,
            "RMSprop": optim.RMSprop,
            "SGD": optim.SGD,
            "Adadelta": optim.Adadelta
        }
    def forward(self, x):
        return self.net(x)
    def fit(self, X_train, y_train, opt_name="Adam", epochs=128, batch_size=32, X_val=None, y_val=None, **opt_kwargs):
        data_train = DataLoader(MyDataset(X_train, y_train),
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)
        data_val = None
        if X_val is not None and y_val is not None:
            data_val = DataLoader(MyDataset(X_val, y_val),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False)
        optimizer = self.opt_dict[opt_name](self.parameters(), **opt_kwargs)
        criterion = nn.BCEWithLogitsLoss()
        self.train()
        for epoch in tqdm(range(epochs)):
            for X_batch, y_batch in data_train:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                self.loss_history.append(loss.item())
                optimizer.step()

            #надо ли сделать self.eval()?
            if data_val is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in data_val:
                        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                        y_val_pred = self.forward(X_val_batch)
                        loss = criterion(y_val_pred, y_val_batch)
                        val_loss += loss.item()
                val_loss /= len(data_val) #зачем?
                self.val_loss_history.append(val_loss)
                self.train()

    def predict(self, X, threshold = 0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).int()


