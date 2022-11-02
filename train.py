import numpy as np
import argparse

import numpy.random

from TD3 import TD3
from utils import create_directory, plot_learning_curve, scale_action
from my_env import Env

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=2000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/TD3/')
parser.add_argument('--figure_file', type=str, default='./output_images/reward.png')
parser.add_argument('--max_steps', type=int, default=30)
args = parser.parse_args()


def main():
    env = Env(1)
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=env.state_dim,
                action_dim=env.action_dim, actor_fc1_dim=400, actor_fc2_dim=300,
                critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir=args.ckpt_dir, gamma=0.99,
                tau=0.005, action_noise=0.01, policy_noise=0.2, policy_noise_clip=0.5,
                delay_time=2, max_size=1000000, batch_size=256)
    create_directory(path=args.ckpt_dir, sub_path_list=['Actor', 'Critic1', 'Critic2', 'Target_actor',
                                                        'Target_critic1', 'Target_critic2'])

    total_reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        total_reward = 0
        state = env.reset()
        for i in range(args.max_steps):
            action = agent.choose_action(state, train=True)
            # action_ = scale_action(action, env.action_low_space, env.action_high_space)
            state_, reward, done = env.next_state(action)
            agent.remember(state, action, reward, state_, done)
            agent.learn()
            total_reward += reward
            state = state_
        total_reward_history.append(total_reward)
        avg_reward = np.mean(total_reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {} Reward: {} AvgReward: {}'.format(episode+1, total_reward, avg_reward))

        if (episode + 1) % 200 == 0:
            agent.save_models(episode+1)

    episodes = [i+1 for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_reward_history, title='AvgReward', ylabel='reward',
                        figure_file=args.figure_file)


if __name__ == '__main__':
    main()
