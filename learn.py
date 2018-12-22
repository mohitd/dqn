"""
Training algorithm
"""
import sys
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer
from utils.gym_envs import get_wrapper_by_name

LOG_FREQ = 1_000

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# if CUDA, use it
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def dqn_learn(env,
        q_func,
        optimizer_spec,
        exploration,
        stopping_criterion,
        replay_buffer_size,
        batch_size,
        gamma,
        learning_starts,
        learning_freq,
        frame_history_len,
        target_update_freq,
        grad_norm_clipping,
        double_q):
    """Implements DQN training
    
    Parameters
    ----------
    env : gym.Env
        OpenAI gym environment
    q_func : torch.nn.Module
        DQN that computes q-values for each action: (state) -> (q-value, action)
    optimizer_spec : OptimizerSpec
        parameters for the optimizer
    exploration : Schedule
        schedule for epsilon-greedy exploration
    stopping_criterion : func
        when to stop training: (env, num_timesteps) -> bool
    replay_buffer_size : int
        experience replay memory size
    batch_size : int
        batch size to sample from replay memory
    gamma : float
        discount factor
    learning_starts : int
        number of environment steps before starting the training process
    learning_freq : int
        number of environment steps between updating DQN weights
    frame_history_len : int
        number of previous frames to include as DQN input
    target_update_freq : int
        number of experience replay steps to update the target network
    grad_norm_clipping : float
        maximum size of gradients to clip to
    double_q : bool
        enable double DQN learning
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    def select_action(dqn, obs, t):
        """Implements epsilon-greedy exploration
        
        Parameters
        ----------
        dqn : torch.nn.Module
            DQN model
        obs : np.ndarray
            Stacked input frames to evaluate
        t : int
            Current time step
        
        Returns
        -------
        nd.array (1,1)
            action to take
        """
        threshold = exploration.value(t)
        if random.random() > threshold:
            # take optimal action
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            # DQN returns (q-value, action)
            q_values = dqn(obs)
            # returns (max, argmax) of q-values (max q-value, action which produces max q-value)
            _, action = q_values.data.max(1)
        else:
            # take a random action
            action = torch.IntTensor([random.randrange(num_actions)])
        return action

    # get input sizes and num actions
    img_h, img_w, img_c = env.observation_space.shape
    in_channels = frame_history_len * img_c
    input_shape = (img_h, img_w, in_channels)
    num_actions = env.action_space.n

    # construct online and target DQNs
    online_DQN = q_func(in_channels=in_channels, num_actions=num_actions)
    target_DQN = q_func(in_channels=in_channels, num_actions=num_actions)

    # construct optimizer
    optimizer = optimizer_spec.constructor(online_DQN.parameters(), **optimizer_spec.kwargs)

    # construct replay memory
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # initialize main loop variables
    num_param_updates = 0
    avg_episode_reward = float('-inf')
    best_avg_episode_reward = float('-inf')
    cumulative_avg_episode_reward = float('-inf')
    prev_obs = env.reset()

    # main training loop
    for t in count():
        # check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break
        
        # store transition and concatenate last frames
        last_idx = replay_buffer.store_frame(prev_obs)

        # stack previous frames into a tensor to give to DQN
        stacked_obs = replay_buffer.encode_recent_observation()

        # take random actions until we've officially started training
        if t > learning_starts:
            # select action according to epsilon-greedy
            action = select_action(online_DQN, stacked_obs, t)[0]
        else:
            # take a random action
            action = random.randrange(num_actions)
        
        # step environment
        obs, reward, done, _ = env.step(action)
        # clip reward
        reward = min(-1.0, max(reward, 1.0))
        # store effect of taking action in prev_obs into replay memory
        replay_buffer.store_effect(last_idx, action, reward, done)

        # if game is finished, reset environment
        if done:
            obs = env.reset()
        prev_obs = obs

        # experience replay
        if t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size):

            # sample batches
            obs_batch, action_batch, reward_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            obs_batch = torch.from_numpy(obs_batch).type(dtype) / 255.0
            action_batch = torch.from_numpy(action_batch).long()
            reward_batch = torch.from_numpy(reward_batch)
            next_obs_batch = torch.from_numpy(next_obs_batch).type(dtype) / 255.0
            not_done_mask = torch.from_numpy(1 - done_mask).type(dtype)

            if torch.cuda.is_available():
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
            
            # Compute current q-values: Q(s, a)
            # Select q-values based on actions we would have taken for each state
            # shape: (BATCH_SIZE, 1)
            current_q_values = online_DQN(obs_batch).gather(1, action_batch.unsqueeze(1))

            # double DQN or vanilla DQN
            if double_q:
                # compute which actions to take according to online network: argmax_a Q(s', a)
                greedy_actions = online_DQN(next_obs_batch).detach().max(1)[1]
                # compute q-values of those actions using target network: Q_hat(s', argmax_a Q(s', a))
                next_q_values = target_DQN(next_obs_batch).gather(1, greedy_actions.unsqueeze(1))
            else:
                # Compute next q-values using target network
                next_q_values = target_DQN(next_obs_batch).detach().max(1)[0]
                next_q_values = next_q_values.unsqueeze(1)
            
            # apply mask to retain q-values
            next_q_values = not_done_mask.unsqueeze(1) * next_q_values
            
            """
            Compute the target q-values (BATCH_SIZE, 1)
            y_j = r_j + gamma * max_a' Q(s', a')                for vanilla DQN
            y_j = r_j + gamma * Q_hat(s', argmax_a Q(s', a))    for double DQN
            """
            target_q_values = reward_batch + (gamma * next_q_values)

            """
            Use the huber loss instead of clipping the TD error.
            Huber loss intuitively means we assign a much larger loss where the error is large (quadratic)
            Smaller errors equate to smaller losses (linear)
            """
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            loss.backward()

            # clip gradients
            nn.utils.clip_grad_norm_(online_DQN.parameters(), grad_norm_clipping)

            # update weights of dqn
            optimizer.step()
            num_param_updates += 1

            # update target network weights
            if num_param_updates % target_update_freq == 0:
                target_DQN.load_state_dict(online_DQN.state_dict())

        # end experience replay

        # log progress so far by averaging last 100 episodes
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            avg_episode_reward = np.mean(episode_rewards[-100:])
            cumulative_avg_episode_reward = np.mean(episode_rewards)
        if len(episode_rewards) > 100:
            best_avg_episode_reward = max(best_avg_episode_reward, avg_episode_reward)

        if t % LOG_FREQ == 0 and t > learning_starts:
            print('-' * 64)
            print('Timestep {}'.format(t))
            print('Average reward (100 episodes): {}'.format(avg_episode_reward))
            print('Best average reward: {}'.format(best_avg_episode_reward))
            print('Cumulative average reward: {}'.format(cumulative_avg_episode_reward))
            print('Episode {}'.format(len(episode_rewards)))
            print('Exploration {}'.format(exploration.value(t)))
            print('\n')
            sys.stdout.flush()

    # end main training loop
# end function
