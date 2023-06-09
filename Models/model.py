import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1280),
            nn.ReLU(),
            nn.Linear(1280, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)