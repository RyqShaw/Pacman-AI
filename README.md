# Pacman AI:
## Teaching myself what goes into making an DQN

Recently having taken a class about Artifical Intelligence, this projects goal was to:

- Learn how a Convulution Neural Network is built and what factors determine stuff such as the layers
- How to use Deep Q Learning to train a Neural Network
- a Better understanding of PyTorch, as well as the important tools needed to run optimized programs
- Make an AI that can play Ms Pacman(specifically) better than I can

## How it works

As Said prior, I am using a CNN `network.py`, using the combination of PyTorch's Documentation, as well as a resource from @liyinxuan0213, I was able to decide upon my Conv layers being:

- Taking an Input of 4 Frames(Frame buffer often used for Atari Games) and out put 32 features at a large kernel size and stride
- Taking those output features, getting 64 more features out of them using a half sized kernel and stride in comparison to the last
- Finally with those inputs, using a fine grain kernel and stride to get outputs of 64, for the next steps in the Neural Network

It is important to note that the frames i am pulling are scaled down to 84x84, and in graysclale. This reduces training time, and the color features are not incredibly important for the scope of this project.

From here I perfrorm 2 layers of Linear Regression on the flattened data, so that the features interpretted can be determined as weights for actions.

Next I made a replay buffer(`replay_buffer.py`) to store experience in training, Essentially acting as a wrapper for the Python Deque. I implemented an add and sample function so that I can store and interpret these important factors:

- State: Where the AI starts
- Action: what it does
- Reward: How Beneficial it was
- Next State: where did it end up
- Done: did the AI die prematurely

These became important to access when we get into the training.

Training at a high level looks like this:

- Instance a Policy Network to Train on
- Instance a Target Network to Reference from and update every X steps
- Instance a Replay Buffer
- Instance an Optimizer and a Loss function so that the Neural Network, so that adjustments can be made based on the Error found
- Start looping through X many episodes
- In each episode, after a certain amount of steps and the the Replay buffer has a certain amount of experiences,  Train Using Q Learning Strategy and Neural Network tools
- Update Target Network after a certain amount of steps
- Return Ideal Network

Other Important Notes about how this works:

- I am using an epsilon greedy strategy: where over time and iterations the likelihood I take more and more random actions goes up(Limited to an extent)
- I limit Episode length, as theoretically a Game could go on forever, making training time miserably long
- Most of code is optimized to be ran on a Nvidia GPU using CUDA, so the way I handle certain things(Specifically Training) may look different than standard PyTorch code just being trained on a CPU

## Acknowledgements
Without these software documentions, and helpful info about Atari games and building DQN, this project would not have been possible

- [Arcade Learning Enviorment](https://ale.farama.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://docs.pytorch.org/docs/stable/)
- [Source for Refernce](https://medium.com/@liyinxuan0213/step-by-step-double-deep-q-networks-double-dqn-tutorial-from-atari-games-to-bioengineering-dec7e6373896)