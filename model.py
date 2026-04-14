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
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(16, 1)
        )
        self.opt_dict = {
            "Adam": optim.Adam,
            "RMSprop": optim.RMSprop,
            "SGD": optim.SGD,
            "Adadelta": optim.Adadelta
        }
    def forward(self, x):
        return self.net(x)
    def fit(self, X_train, y_train, opt_name="Adam", epochs=128, batch_size=32, X_val=None, y_val=None, **opt_kwargs):
        self.to(self.device)
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
        n_pos = y_train.sum().item()  # если класс 1 = 1, класс 0 = 0
        n_neg = len(y_train) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.train()

        best_val_loss = float('inf')
        patience = 64
        patience_counter = 0

        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in data_train:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.forward(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
                n_batches += 1
                epoch_loss += (loss.item() - epoch_loss) / n_batches
            self.loss_history.append(epoch_loss)
            pbar.set_postfix({'loss': f'{epoch_loss:.4f}'})
            #надо ли сделать self.eval()?
            if data_val is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in data_val:
                        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                        y_val_pred = self.forward(X_val_batch)
                        loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
                        val_loss += loss.item()
                val_loss /= len(data_val)
                self.val_loss_history.append(val_loss)
                self.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    def predict_proba(self, X):
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.forward(X)
            return torch.sigmoid(logits).cpu().numpy()

    def predict(self, X, threshold = 0.5):
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).int().cpu()

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.val_loss_history = checkpoint['val_loss_history']


