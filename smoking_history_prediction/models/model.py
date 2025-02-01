import torch.nn as nn

class SmokingPredictionNN(nn.Module):
    def __init__(self, input_dim):
        super(SmokingPredictionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 23),
            nn.ReLU(),
            nn.Linear(23, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
