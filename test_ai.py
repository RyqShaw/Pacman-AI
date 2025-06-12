import ale_py
import gymnasium as gym
import torch

#Set Rendermode
visual_render = True
rendering = ""

if visual_render:
    rendering = "human"
else:
    rendering = "rgb_array"
    
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

# Gymnasium Env
gym.register_envs(ale_py)

env = gym.make('ALE/MsPacman-v5', render_mode=rendering, obs_type="grayscale")
obs, info = env.reset()

# Game Loop
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated

env.close()