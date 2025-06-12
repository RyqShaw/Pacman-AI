import ale_py
import gymnasium as gym
import torch
from network import Network

#Set Rendermode
visual_render = True
rendering = ""

if visual_render:
    rendering = "human"
else:
    rendering = "rgb_array"

# Gymnasium Env
gym.register_envs(ale_py)

env = gym.make('ALE/MsPacman-v5', render_mode=rendering, obs_type="grayscale")

env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.FrameStackObservation(env, 4)

obs, info = env.reset()

#Setup network
cnn = Network(env.action_space.n)
cnn.load_state_dict(torch.load("nn.path"))
cnn.eval()

# Game Loop
done = False
while not done:
    action = 0
    with torch.no_grad():
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        q_values = cnn.forward(state_tensor)
        action = torch.argmax(q_values).item()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()