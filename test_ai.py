import ale_py
import gymnasium as gym

#Set Rendermode
visual_render = False
rendering = ""

if visual_render:
    rendering = "human"
else:
    rendering = "rgb_array"

# Gymnasium Env
gym.register_envs(ale_py)

env = gym.make('ALE/Pacman-v5', render_mode=rendering)
obs, info = env.reset()

# Game Loop
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
env.close()