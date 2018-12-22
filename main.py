"""
Main file to run
"""

import gym
import random
import argparse
import torch.optim as optim

from utils.gym_envs import get_env, get_wrapper_by_name
from utils.schedules import PiecewiseSchedule
from dqn import DQN, DuelingDQN

from learn import OptimizerSpec, dqn_learn

ENVS = ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4']

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env', help='Atari environment', choices=ENVS, default=ENVS[0])

# optimizer params
LEARNING_RATE = 0.001

# DQN training params
REPLAY_BUFFER_SIZE = 1_000_000_000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_STARTS = 50_000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
GRAD_NORM_CLIPPING = 10
DOUBLE_Q = True

def atari_learn(env, num_timesteps):
    """Trains DQN on 
    
    Parameters
    ----------
    env : gym.Env
        OpenAI Gymgym environment
    num_timesteps : int
        maximum number of time steps
    """

    optimizer = OptimizerSpec(constructor=optim.Adam, kwargs={'lr': LEARNING_RATE})
    exploration_schedule = PiecewiseSchedule([
            (0, 1.0),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )

    def stopping_criterion(env, t):
        """Determine when to stop DQN training
        
        Parameters
        ----------
        env : gym.Env
            gym environment
        t : int
            timestep of environment
        
        Returns
        -------
        bool
            True if we can stop training, False otherwise
        """

        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    dqn_learn(env=env,
        q_func=DuelingDQN,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ,
        grad_norm_clipping=GRAD_NORM_CLIPPING,
        double_q=DOUBLE_Q)

if __name__ == '__main__':
    args = parser.parse_args()

    # create environment and generate random seed
    task = gym.make(args.env)
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)

    # wrap environment in the same style as DeepMind
    env = get_env(task, seed)

    atari_learn(env, num_timesteps=1e8)
