import torch.nn as nn
import torch

class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, lr=1e-3, weight_decay=1e-5):
        super().__init__()

        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h_size
        layers.append(nn.Linear(in_size, 1))

        self.network = nn.Sequential(*layers)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, x, y):
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss.item()