import torch
import numpy as np
import torch.optim as optim
from torch.amp import autocast, GradScaler
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
env = gym.wrappers.FrameStackObservation(env, 4)
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
def train(batch_size=256, max_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.999, min_epsilon=0.1, max_episode_steps=300, load_checkpoint=False):
    # Deep Q Learning Setup
    policy_nn = Network(env.action_space.n).to(device)
    target_nn = Network(env.action_space.n).to(device)
    min_replay_size = 5000
    buffer = ReplayBuffer(50000)
    optimizer = optim.Adam(policy_nn.parameters(), lr=0.0001)
    mse_loss_nn = torch.nn.MSELoss()
    episodes_done = 0
    total_steps = 0
    #Load Checkpoint if needed
    if os.path.exists("checkpoint.path") and load_checkpoint:
        print("Loading Checkpoint")
        checkpoint = torch.load("checkpoint.path")
        episodes_done = checkpoint['episode']
        policy_nn.load_state_dict(checkpoint['policy_state_dict'])
        target_nn.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epsilon = checkpoint['epsilon']
    
    for episode in range(episodes_done, max_episodes):
        episode_steps = 0
        if episode % 100 == 0:
            print(f"Episode: {episode} / {max_episodes}")
        
        obs, info = env.reset()
        
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
                    state = torch.FloatTensor(normalized_obs).unsqueeze(0).to(device)
                    q_vals = policy_nn.forward(state)
                    action = torch.argmax(q_vals).item()

            new_obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            episode_steps += 1
            # Scales reward
            clipped_reward = np.clip(reward, -1, 1)
            
            # add to buffer
            new_obs_array = np.array(new_obs)
            normalized_new_obs = new_obs_array.astype(np.float32) / 255.0
            done = terminated or truncated  # Calculate done first
            buffer.add(normalized_obs, action, clipped_reward, normalized_new_obs, done)
            
            # Do Deep Q Learning at Batch Size, update every 4 steps
            if len(buffer) >= min_replay_size and total_steps % 4 == 0:
                scaler = GradScaler()
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                
                # Process all values
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.BoolTensor(dones).to(device)
                
                # Memory saving for gpu training
                with autocast():
                    q_vals = policy_nn.forward(states).gather(1, actions.unsqueeze(1))
                    
                    with torch.no_grad():
                        # Q Learning, Getting the max from the first state and making that the target_q
                        next_q_values = target_nn(next_states).max(1)[0]
                        target_q = rewards + gamma * next_q_values * ~dones 

                    # Loss Calculations
                    loss = mse_loss_nn(q_vals.squeeze(), target_q)
                
                # Optimizer to update gradients
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Gradient clipping to prevent instability in training
                torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), max_norm=10.0)
                
                # Free some memory now
                del states, next_states, q_vals, target_q, loss
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Free some memory now
                torch.cuda.empty_cache()
                
                # Update target every 1000 steps AI takes
                if total_steps % 1000 == 0:
                    target_nn.load_state_dict(policy_nn.state_dict())
            
            # update observation for next round
            obs = new_obs
            
            # Episode Limit
            if episode_steps >= max_episode_steps:
                done = True

        epsilon = max(min_epsilon, epsilon * decay_rate)
        
        # Update Target NN + checkpoint
        if episode % 100 == 0 and episode != 0:
            print("Checkpoint!")
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy_nn.state_dict(),
                'target_state_dict': target_nn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
            }
            torch.save(checkpoint, "checkpoint.path")
    
    torch.save(target_nn.state_dict(), "nn.path")

start_time = time.time()
train(load_checkpoint=False)
end_time = time.time()
print(f"Total Time in Training: {end_time-start_time}")