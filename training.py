import torch
import numpy as np
import torch.optim as optim
import gymnasium as gym
import ale_py
from network import Network
from replay_buffer import ReplayBuffer
import os
import time

#Setup
gym.register_envs(ale_py)
env = gym.make('ALE/MsPacman-v5',  obs_type="grayscale")
# Resizing to 84x84, found to be a sweet spot for atari games
env = gym.wrappers.ResizeObservation(env, (84, 84))

# 4 Frames as input
env = gym.wrappers.FrameStack(env, 4)
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
def train(batch_size=64, max_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.999, min_epsilon=0.1, load_checkpoint=False):
    # Deep Q Learning Setup
    policy_nn = Network(env.action_space.n).to(device)
    target_nn = Network(env.action_space.n).to(device)
    min_replay_size = 10000
    buffer = ReplayBuffer(min_replay_size)
    optimizer = optim.Adam(policy_nn.parameters(), lr=0.0001)
    mse_loss_nn = torch.nn.MSELoss()
    episodes_done = 0
    total_steps = 0
    #Load Checkpoint if needed
    if os.path.exists("checkpoint.pth") and load_checkpoint:
        print("Loading Checkpoint")
        checkpoint = torch.load("checkpoint.pth")
        episodes_done = checkpoint['episode']
        policy_nn.load_state_dict(checkpoint['policy_state_dict'])
        target_nn.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
    
    for episode in range(episodes_done, max_episodes):
        obs, reward, terminated, truncated, info = env.reset()
        
        done = False
        while not done:
            # obs on scale of 0 to 1 for all pixels
            normalized_obs = obs.astype(np.float32) / 255.0
            action = 0
            
            # Epsilon Greedy: either random action
            if epsilon > np.random.rand():
                action = np.random.randint(env.action_space.n)
            
            # Or Optimal Action
            else:
                # Makes observation into a tensor
                # sends it to nn
                # Gets best possible action
                with torch.no_grad():
                    state = torch.FloatTensor(normalized_obs).unsqueeze(1).to(device)
                    q_vals = policy_nn.forward(state)
                    action = torch.argmax(q_vals).item()

            new_obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            # Scales reward
            clipped_reward = np.clip(reward, -1, 1)
            
            # add to buffer
            new_obs = torch.tensor(new_obs, dtype=torch.float32).to(device)
            buffer.add(normalized_obs, action, clipped_reward, new_obs, done)
            
            # Do Deep Q Learning at Batch Size
            if len(buffer) >= min_replay_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Process all values
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.BoolTensor(dones).to(device)
                
                q_vals = policy_nn.forward(states).gather(1, actions)
                
                with torch.no_grad():
                    # Q Learning, Getting the max from the first state and making that the target_q
                    new_q_values = target_nn(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * new_q_values
                
                # MSE loss 
                loss = mse_loss_nn(q_vals, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent instability in training
                torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), max_norm=10.0)
                optimizer.step()
                
                # Update target every 1000 steps AI takes
                if total_steps % 1000 == 0:
                    target_nn.load_state_dict(policy_nn.state_dict())
                    
                # update observation for next round
                obs = new_obs
            
            done = terminated or truncated

        epsilon = max(min_epsilon, epsilon * decay_rate)
        
        # Update Target NN + checkpoint
        if episode % 100 == 0 and episode != 0: 
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy_nn.state_dict(),
                'target_state_dict': target_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }
        torch.save(checkpoint, "checkpoint.pth")
    
    torch.save(target_nn.state_dict(), "nn.path")

train(max_episodes=1000)
