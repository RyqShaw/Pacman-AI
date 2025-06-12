import torch
import torch.nn as nn

import gymnasium as gym
import ale_py

class Network(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.stack = nn.Sequential(
            #Stack Explanation: 4 Input channels for 4 gray scale frames stacked together
            #Starts with large kernel
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            #Takes Output and goes over it again to get more features and go more in depth learning
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            #Should have same amount of features but 3x3 kernel will help with fine grain learning
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,stride=1),
            nn.Flatten(),
            #Input features aftewr flattening
            # the output is (32,20,20) after 1 layer (84-8/4 +1) = 20
            # the output is (64,9,9) after 2 layer (20-4/2 +1) = 9
            # the output is (64,7,7) after 1 layer (9-3/1 +1) = 7
            nn.Linear(in_features=3136, out_features=256),
            nn.ReLU(),
            # Take features, send to output
            nn.Linear(in_features=256, out_features=out_dim),
        )
        
    def forward(self, x):
        return self.stack(x)