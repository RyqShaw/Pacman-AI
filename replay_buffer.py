import torch
import random
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
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool)
        )

    def __len__(self):
        return len(self.buffer)