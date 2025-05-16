"""
Multi-Agent DQN Training and Evaluation Script

This script handles the training and evaluation of the SMADQN algorithm
in the multi-agent exploration environment. It provides:
- Training and evaluation modes
- Performance tracking and statistics
- Result visualization
- Model checkpointing
"""

from pretrain_ts_env import ENV
from pretrain_ts_alg import SMADQN
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def run(mode, ep_num, save_path):
    """
    Run training or evaluation of the SMADQN algorithm.
    
    Args:
        mode (str): 'train' or 'test' mode
        ep_num (int): Number of episodes to run
        save_path (str): Directory to save results
        
    The function tracks and reports:
    - Episode rewards and exploration ratios
    - Best and worst episode performance
    - Training time and statistics
    - Generates performance plots (in training mode)
    """
    step_total = 0
    episode = 0
    rewardSum_temp = 0
    exploration_sum_temp = 0
    meanReward_list = []
    exploration_ratio_list = []
    timeCostSum_temp = 0
    meanTime_list = []
    timeStart = time.time()
    
    # Track best and worst episodes
    best_episode = {'reward': float('-inf'), 'episode': 0, 'exploration': 0, 'steps': 0}
    worst_episode = {'reward': float('inf'), 'episode': 0, 'exploration': 0, 'steps': 0}
    best_episode_overall = {'reward': float('-inf'), 'episode': 0, 'exploration': 0, 'steps': 0}
    worst_episode_overall = {'reward': float('inf'), 'episode': 0, 'exploration': 0, 'steps': 0}

    for ep in range(ep_num):
        episode += 1
        observation = env.reset()
        ep_reward = 0  # Track total reward for this episode
        ep_timeCost = 0

        # Show episode info at start
        env.show_episode_info(episode, 0)

        for step in range(MAX_EP_STEPS):
            env.render()
            
            # Choose actions for all agents
            actions = np.zeros(agentNum, dtype=np.int32)
            for i in range(agentNum):
                actions[i] = RL.choose_action(observation[i])

            # Execute actions
            observation_, reward, done, exploration_ratio = env.step(actions)

            # Update episode info
            env.show_episode_info(episode, exploration_ratio)

            if mode == 'train':  # Store transitions for all agents
                for i in range(agentNum):
                    RL.store_transition(observation[i], actions[i], reward[i], observation_[i], done)

            if mode == 'train':  # Learn
                if (step_total > 200) and (step_total % 5 == 0):
                    RL.learn()

            observation = observation_
            ep_reward += reward[0]  # Accumulate reward through episode (agents get same reward)
            step_total += 1
            ep_timeCost += 1

            if done or step == MAX_EP_STEPS - 1:
                # Update best/worst episodes
                current_stats = {
                    'reward': ep_reward,
                    'episode': episode,
                    'exploration': exploration_ratio,
                    'steps': ep_timeCost
                }
                
                # Update for 100-episode window
                if ep_reward > best_episode['reward']:
                    best_episode = current_stats.copy()
                if ep_reward < worst_episode['reward']:
                    worst_episode = current_stats.copy()
                    
                # Update overall best/worst
                if ep_reward > best_episode_overall['reward']:
                    best_episode_overall = current_stats.copy()
                if ep_reward < worst_episode_overall['reward']:
                    worst_episode_overall = current_stats.copy()
                
                rewardSum_temp += ep_reward
                exploration_sum_temp += exploration_ratio
                timeCostSum_temp += ep_timeCost
                break

        if mode == 'train':
            if episode % 100 == 0:
                mean_reward = rewardSum_temp / 100
                mean_exploration = exploration_sum_temp / 100
                mean_time = timeCostSum_temp / 100
                
                print(f"\n~~~~~~~  Episode {episode} ~~~~~~~~")
                print(f"Average exploration ratio: {mean_exploration:.2%}")
                print(f"Average episode reward: {mean_reward:.4f}")
                print(f"Average steps per episode: {mean_time:.1f}")
                print(f"Last episode reward: {ep_reward:.4f}")
                
                print("\nBest episode in last 100:")
                print(f"Episode {best_episode['episode']}")
                print(f"Reward: {best_episode['reward']:.4f}")
                print(f"Exploration: {best_episode['exploration']:.2%}")
                print(f"Steps: {best_episode['steps']}")
                
                print("\nWorst episode in last 100:")
                print(f"Episode {worst_episode['episode']}")
                print(f"Reward: {worst_episode['reward']:.4f}")
                print(f"Exploration: {worst_episode['exploration']:.2%}")
                print(f"Steps: {worst_episode['steps']}")
                
                meanReward_list.append(mean_reward)
                exploration_ratio_list.append(mean_exploration)
                meanTime_list.append(mean_time)
                
                # Reset accumulators and episode trackers
                rewardSum_temp = 0
                exploration_sum_temp = 0
                timeCostSum_temp = 0
                best_episode = {'reward': float('-inf'), 'episode': 0, 'exploration': 0, 'steps': 0}
                worst_episode = {'reward': float('inf'), 'episode': 0, 'exploration': 0, 'steps': 0}

    # Print overall statistics at the end
    print("\n~~~~~~~  Overall Training Statistics ~~~~~~~~")
    print(f"Total episodes: {episode}")
    print(f"Total steps: {step_total}")
    print(f"Training time: {time.time() - timeStart:.1f} seconds")
    
    print("\nBest episode overall:")
    print(f"Episode {best_episode_overall['episode']}")
    print(f"Reward: {best_episode_overall['reward']:.4f}")
    print(f"Exploration: {best_episode_overall['exploration']:.2%}")
    print(f"Steps: {best_episode_overall['steps']}")
    
    print("\nWorst episode overall:")
    print(f"Episode {worst_episode_overall['episode']}")
    print(f"Reward: {worst_episode_overall['reward']:.4f}")
    print(f"Exploration: {worst_episode_overall['exploration']:.2%}")
    print(f"Steps: {worst_episode_overall['steps']}")

    if mode == 'train':
        RL.save_parameters()
        np.savetxt(save_path+'meanReward.txt', meanReward_list)
        np.savetxt(save_path+'timeCost.txt', meanTime_list)
        np.savetxt(save_path+'exploration_ratio.txt', exploration_ratio_list)
        
        # Save best and worst episode stats
        best_stats = np.array([
            best_episode_overall['episode'],
            best_episode_overall['reward'],
            best_episode_overall['exploration'],
            best_episode_overall['steps']
        ])
        worst_stats = np.array([
            worst_episode_overall['episode'],
            worst_episode_overall['reward'],
            worst_episode_overall['exploration'],
            worst_episode_overall['steps']
        ])
        np.savetxt(save_path+'best_episode.txt', best_stats)
        np.savetxt(save_path+'worst_episode.txt', worst_stats)
        
        # Plot results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(meanReward_list))*100, meanReward_list, 'b.-')
        plt.ylabel('Average episode reward')
        plt.xlabel('Number of training episodes')
        
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(exploration_ratio_list))*100, exploration_ratio_list, 'r.-')
        plt.ylabel('Average exploration ratio')
        plt.xlabel('Number of training episodes')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or evaluate SMADQN algorithm')
    parser.add_argument("--agent_num", type=int, default=4, help="Number of agents")
    parser.add_argument("--mode", type=str, default='train', help="'train' or 'test' mode")
    parser.add_argument("--save_path", type=str, default='./result', help="Path to save results")
    args = parser.parse_args()
    
    # Set up environment and parameters
    mode = args.mode
    if mode == 'train':
        ep_num = 10000  # Number of training episodes
        np.random.seed(5)
    else:
        ep_num = 1000  # Number of evaluation episodes
        np.random.seed(2)
    
    # Initialize environment
    agentNum = args.agent_num
    env = ENV(agentNum)
    envSize = env.ENV_H
    MAX_EP_STEPS = envSize * 2  # Maximum steps per episode
    envSize -= 0.99
    historyStep = env.historyStep
    
    # Create save directory
    print(f'N={env.agentNum}')
    save_path = f"{args.save_path}/N{env.agentNum}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path
    
    # Initialize SMADQN agent
    RL = SMADQN(env.n_actions, env.n_features, args.mode,
                learning_rate=0.01,
                reward_decay=0.9,
                replace_target_iter=300,
                memory_size=1500,
                batch_size=32,
                model_path=save_path
                )
    
    # Start training/evaluation
    env.after(100, run(args.mode, ep_num, save_path))
    env.mainloop()
