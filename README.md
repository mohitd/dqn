# Deep Q-Networks

Implementation of Deep Q-Networks in Pytorch. Along with the base implementation with the target network and experience replay, Dueling DQNs and double DQNs are also implemented.

## Features
- Base DQN (target network, frame skipping, experience replay)
- Double DQN (DDQN)
- Dueling DQN

## Getting Started

Install the following prerequisites on your system

- pytorch
- torchvision
- opencv
- gym
- gym[atari]

To execute a DQN, run the `main.py` file.

```
python main.py
```

All of the DQN training and optimizer parameters are at the top of `main.py` so feel free to modify these to suit your configuration.

There are some parameter configurations on the command line. More will be added!

## Todos
- Implement Prioritized Experience Replay
- More command line configurations, e.g., enable/disable dueling DQN/DDQN, set number of timesteps, etc.
- Train for a few days and post results

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Berkeley Deep RL course for DeepMind Atari wrappers and program structure (https://github.com/berkeleydeeprlcourse/homework)
* Idea on how to better organize the main training loop (https://github.com/transedward/pytorch-dqn)
