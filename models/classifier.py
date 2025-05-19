import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        return self.linear(x)
