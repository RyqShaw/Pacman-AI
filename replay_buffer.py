import torch
import numpy as np
from collections import deque

#Replay Buffer Clas to recall events from
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    #can input exact sars 
    def add(self, state, action, reward, new_state, done):
        self.buffer.append((state, action, reward, new_state, done))

    # get random sars
    def sample(self, batch_size):
        random_sample = np.random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = zip(*random_sample)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float),
            torch.stack(next_states),
            torch.stack(done)
        )

    def __len__(self):
        return len(self.buffer)