import torch
import numpy as np
import torch.optim as optim
import gymnasium as gym
import ale_py
from network import Network
from replay_buffer import ReplayBuffer

#Setup
gym.register_envs(ale_py)
env = gym.make('ALE/MsPacman-v5',  obs_type="grayscale")
# Resizing to 84x84, found to be a sweet spot for atari games
env = gym.wrappers.ResizeObservation(env, (84, 84))
obs, info = env.reset()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Running on Device: {device}")

# Runs for certain amount of episodes
# Inputs are: 
#   - batch_size: samples processed together
#   - gamma: Multiplier for future rewards
#   - epsilon: Epsilon Greedy Strategy value to determine random actions
#   - decay_rate: amount epsilon is decayed by every iteration
def train(batch_size, max_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.999, min_epsilon=0.1):
    # Deep Q Learning Setup
    policy_nn = Network(env.action_space.n).to(device)
    target_nn = Network(env.action_space.n).to(device)
    buffer = ReplayBuffer(10000)
    optimizer = optimizer = optim.Adam(policy_nn.parameters, lr=0.0001)
    total_reward = 0
    
    for episode in max_episodes:
        obs, reward, terminated, truncated, info = env.reset()
        
        done = False
        while not done:
            action = 0
            
            # Epsilon Greedy: either random action
            if epsilon > np.random.rand():
                action = np.random.randint(6)
            
            # Or Optimal Action
            else:
                # Makes observation into a tensor
                # sends it to nn
                # Gets best possible action
                with torch.no_grad():
                    state = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    q_vals = policy_nn.forward(state)
                    action = torch.argmax(q_vals).item()

            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # Punish Lack of Action
            if action == 0:
                reward -= 0.01
            
            # add to buffer
            new_obs = torch.tensor(new_obs, dtype=torch.float32).flatten().to(device)
            buffer.add(obs, action, reward, new_obs, done)
            total_reward += reward
            
            # Do Deep Q Learning at Batch Size
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Process all values
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.BoolTensor(bool).to(device)
                
                q_vals = policy_nn.forward(states).gather(1, actions)
                
                with torch.no_grad():
                    # Q Learning, Getting the max from the first state and making that the target_q
                    new_q_values = policy_nn(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * new_q_values
                
                loss_val = 0 #mse implement
            
            done = terminated or truncated

        epsilon = max(min_epsilon, epsilon * decay_rate)

train()
