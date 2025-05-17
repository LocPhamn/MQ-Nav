import numpy as np
import math
import time
from env import ENV
from algo import HIST_Alg
import os
import argparse
from deep_q_algo import Deep_Q_Algo
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--agent_num", type=int, default=0)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--pretrain_path", type=str, default='./pretrain/result')
parser.add_argument("--save_path", type=str, default='./result')
parser.add_argument("--load_path", type=str, default='./result')
parser.add_argument("--render", type=bool, default=False)  # Add render flag
args = parser.parse_args()

mode = args.mode
if mode == 'train':
    ep_num = 2
    np.random.seed(1)
else:
    ep_num = 1000
    np.random.seed(2)

agentNum = args.agent_num
env = ENV(agentNum)
agentNum = env.agentNum
envSize = env.ENV_H
obsNum = env.obsNum
detecDirNum = env.n_states_ca
MAX_EP_STEPS = envSize * 3
envSize_ = envSize - 0.99
historyStep = env.historyStep
s_dim_dqn = env.n_states_ts
s_dim_ddpg = env.n_states_ca + 2   
s_dim = s_dim_dqn + env.n_states_ca  
n_actions = env.n_actions_ts
a_dim = env.n_actions_ca
a_bound = env.max_torque
model_path_dqn = f"{args.pretrain_path}/N{agentNum}/"

if mode == 'train':
    path = f"{args.save_path}/N{agentNum}/"
    if not os.path.exists(path):
        os.makedirs(path)
else:
    path = f"{args.load_path}/N{agentNum}/"

RL = HIST_Alg(a_dim, n_actions, s_dim, s_dim_ddpg, s_dim_dqn, a_bound, envSize, model_path_dqn, path, mode)
RL_DQN = Deep_Q_Algo(2*agentNum, n_actions)

# Initialize arrays
max_torque = env.max_torque
agentSize = env.agentSize
agentPositionArray = np.zeros((agentNum, 2))
agentPositionArray0 = np.zeros((agentNum, 2))
tarPositionArray = np.zeros((agentNum, 2))
tarPositionArray0 = np.zeros((agentNum, 2))
obstacleArray = np.zeros((obsNum, 2))
agentObstacleDisTS = np.ones((agentNum, detecDirNum))

# Training statistics
step_total = 0
episode = 0
temp_rewardSum = 0
totalReward_list = []
meanReward_list = []
timeCostSum_temp = 0
meanTime_list = []
collision_num = 0
collision_obs_num = 0
collision_wall_num = 0
collisionNum_list = []
success_num = 0
successNum_list = []
found_targets_num = 0
found_targets_list = []  # List to store targets found per episode
total_targets_found = 0  # Track total targets found across all episodes

# Performance optimization
BATCH_SIZE = 128  # Increased batch size
RENDER_INTERVAL = 10  # Render every N steps
timeStart = time.time()

for ep in range(ep_num):
    episode += 1
    # Generate obs, agent, tar
    locationArea = np.random.choice(envSize, agentNum * 2 + obsNum, replace=False)
    for i in range(obsNum):
        obstacleArray[i] = [locationArea[i] // 5 * 5 + 2.5, locationArea[i] % 5 * 5 + 2.5] + 1 * np.random.rand(2)
    for i in range(agentNum):
        tarPositionArray0[i] = [locationArea[obsNum + i] // 5 * 5 + 2, locationArea[obsNum + i] % 5 * 5 + 2] + 2 * np.random.rand(2)
        agentPositionArray0[i] = [locationArea[obsNum + agentNum + i] // 5 * 5 + 2, locationArea[obsNum + agentNum + i] % 5 * 5 + 2] + 2 * np.random.rand(2)
    
    # Vectorized sorting
    sortTar_index = np.argsort(tarPositionArray0[:, 0])
    tarPositionArray = tarPositionArray0[sortTar_index]
    sortAgent_index = np.argsort(agentPositionArray0[:, 0])
    agentPositionArray = agentPositionArray0[sortAgent_index]
    
    obstacleSize = np.random.rand(obsNum) * 1.3 + 0.5

    observation = env.reset(agentPositionArray, tarPositionArray, obstacleArray, obstacleSize)
    ep_timeCost = 0
    agentObstacleDis = np.ones((agentNum, detecDirNum))
    agentExistObstacle_Target = np.zeros(agentNum)
    agentExistObstacle_Target_old = np.zeros(agentNum)
    tarAngle = np.zeros(agentNum)
    observationCA = np.hstack((observation[:, :2], agentObstacleDis))
    action = np.zeros(agentNum, dtype=int)
    action_ = np.zeros(agentNum, dtype=int)
    action_h = -np.ones(agentNum, dtype=np.int)
    move = np.zeros((agentNum, 2))
    agentDone = np.zeros(agentNum, dtype=int)
    ep_reward = np.zeros(agentNum)
    observation_h = np.tile(observation[:, -(agentNum - 1) * 2:], historyStep)
    observation_h_temp = observation_h
    collision_cross = np.zeros(agentNum)
    
    # Reset episode statistics
    ep_collision_agent = 0
    ep_collision_obs = 0
    ep_collision_wall = 0

    for step in range(MAX_EP_STEPS):
        # Render only if enabled and at specified interval
        if args.render and step % RENDER_INTERVAL == 0:
            env.render()
            
        # Vectorized action selection
        otherTarCoordi = np.zeros((agentNum, 2))
        action_ddpg = np.zeros(agentNum)
        
        # Batch process observations
        observationCA = np.zeros((agentNum, 9))  # 2 for target coords + 7 for obstacle distances
        for i in range(agentNum):
            observationCA[i] = np.hstack((
                observation[i, action[i]*2: action[i]*2+2],
                agentObstacleDis[i]
            ))
        
        # Calculate actions for each agent
        action_ddpg = np.zeros(agentNum)
        for i in range(agentNum):
            # Get the target-related state components (2 dimensions per target)
            target_state = observation[i, :2*agentNum]
            action_ddpg[i] = RL_DQN.make_decision(target_state)
            
        move = np.column_stack((
            np.sin(tarAngle + action_ddpg) * env.stepLength,
            -np.cos(tarAngle + action_ddpg) * env.stepLength
        ))

        observation_, reward, done, agentDone, collision_cross, collision_agent, collision_obs, success, arriveSame, agentPositionArray, collision_wall = \
            env.move(move, agentExistObstacle_Target, otherTarCoordi, action, action_h, 0)

        # Update statistics
        ep_collision_agent += collision_agent
        ep_collision_obs += collision_obs
        ep_collision_wall += collision_wall
        ep_reward += reward
        ep_timeCost += 1

        # Update total statistics
        collision_num += collision_agent
        collision_obs_num += collision_obs
        collision_wall_num += collision_wall
        success_num += success
        targets_found = np.sum(env.founded_targets)
        total_targets_found += targets_found
        found_targets_list.append(targets_found)  # Store targets found in this episode
        temp_rewardSum += min(ep_reward)

        # Update history
        action_h = action.copy()
        observation_h_temp = np.zeros_like(observation_h)
        observation_h_temp[:, 2 * (agentNum - 1):] = observation_h[:, :-2 * (agentNum - 1)]
        observation_h_temp[:, :2 * (agentNum - 1)] = observation[:, -2 * (agentNum - 1):]
        observation_h = observation_h_temp.copy()
        observation = observation_.copy()
        step_total += 1

        # Train DQN with larger batch size
        if mode == 'train' and len(RL_DQN.replay_buffer) > BATCH_SIZE:
            loss = RL_DQN.train_main_network(batch_size=BATCH_SIZE)
            # Update epsilon after training
            RL_DQN.epsilon = max(RL_DQN.epsilon_min, RL_DQN.epsilon * RL_DQN.epsilon_decay)
        else:
            loss = 0.0

        agentExistObstacle_Target_old = agentExistObstacle_Target.copy()
        otherTarCoordi = np.zeros((agentNum, 2))

        # Vectorized action selection for next step
        for i in range(agentNum):
            if collision_cross[i] != 1:
                action_[i] = RL.choose_action_dqn(np.hstack((
                    observation_[i], 
                    observation_h_temp[i], 
                    observation_[i, -2*(agentNum-1):]
                )))
            else:
                action_[i] = action[i]
                
            # Vectorized target detection
            tarAgentCoordi = (observation_[i, 2 * action_[i]: 2 * action_[i] + 2]) * envSize
            tarAgentDis = np.linalg.norm(tarAgentCoordi)
            
            if tarAgentDis >= 1:
                # Calculate distances to other targets
                other_targets = np.zeros((agentNum, 2))
                for j in range(agentNum):
                    other_targets[j] = observation_[i, 2*j:2*j+2] * envSize
                
                other_distances = np.linalg.norm(other_targets, axis=1)
                valid_targets = (other_distances > 1) & (other_distances < 2.5)
                if np.any(valid_targets):
                    otherTarCoordi[i] = -other_targets[np.where(valid_targets)[0][0]]
                
                tarAgentDirCoordi = tarAgentCoordi / tarAgentDis
                agentNextDDPG, agentObstacleDis[i, 0] = env.detect_obstacle(tarAgentDirCoordi, i, otherTarCoordi[i])
                
                if agentNextDDPG == 1:
                    tarAngle[i] = math.asin(tarAgentCoordi[0] / tarAgentDis)
                    if tarAgentCoordi[1] >= 0:
                        if tarAgentCoordi[0] >= 0:
                            tarAngle[i] = np.pi - tarAngle[i]
                        if tarAgentCoordi[0] < 0:
                            tarAngle[i] = -np.pi - tarAngle[i]
                            
                    # Vectorized angle calculations
                    angles = np.array([tarAngle[i] + (j+1) * np.pi / 6 for j in range(3)])
                    negative_angles = np.array([tarAngle[i] - (j+1) * np.pi / 6 for j in range(3)])
                    angles = np.concatenate([angles, negative_angles])
                    polar_coords = np.column_stack((np.sin(angles), -np.cos(angles)))
                    
                    for j, coord in enumerate(polar_coords):
                        _, agentObstacleDis[i, j + 1] = env.detect_obstacle(coord, i, otherTarCoordi[i])

            observation_All = np.hstack((observation_[i], observation_h_temp[i], observation_[i, -2*(agentNum-1):], agentObstacleDis[i]))
            if mode == 'train' and agentExistObstacle_Target[i] == 1:
                RL_DQN.save_experience(observationCA[i], action_ddpg[i], reward[i], observation_All, done[i])

        if sum(done) or step == MAX_EP_STEPS - 1:
            if success == 0:
                ep_timeCost = MAX_EP_STEPS
            timeCostSum_temp += ep_timeCost
            
            # Cập nhật thông số cho đồ thị
            if mode == 'train':
                RL_DQN.update_metrics(
                    episode_reward=min(ep_reward),
                    episode_loss=loss,  # Use actual loss from training
                    episode_steps=ep_timeCost,
                    success_rate=1.0 if success else 0.0,
                    found_targets=np.sum(env.founded_targets)
                )
            
            # Display per-episode statistics
            exploration_ratio = np.sum(env.grid_map) / (env.ENV_H * env.ENV_H)
            print(f"\nEpisode {episode} | Steps: {ep_timeCost} | Reward: {np.around(min(ep_reward), decimals=3)} | Success: {'Yes' if success else 'No'} | Found Targets: {np.sum(env.founded_targets)}/{env.agentNum} | Explored: {exploration_ratio:.1%} | Agent Collisions: {ep_collision_agent} | Obstacle Collisions: {ep_collision_obs} | Wall Collisions: {ep_collision_wall}")
            break

    # Display statistics every 100 episodes during training
    if mode == 'train' and episode % 100 == 0:
        print("\n" + "="*100)
        print(f"Training Statistics - Episode {episode}")
        print("="*100)
        avg_exploration = np.sum(env.grid_map) / (env.ENV_H * env.ENV_H)
        # Calculate average of last 100 episodes for targets found
        recent_targets = found_targets_list[-100:] if len(found_targets_list) >= 100 else found_targets_list
        avg_targets = np.mean(recent_targets)
        print(f"Success Rate: {success_num/100:.2%} | Avg Targets (Last 100): {avg_targets:.2f} | Avg Exploration: {avg_exploration:.1%} | Avg Agent Collisions: {collision_num/100:.2f} | Avg Obstacle Collisions: {collision_obs_num/100:.2f} | Avg Wall Collisions: {collision_wall_num/100:.2f} | Avg Reward: {temp_rewardSum/100:.2f}", end='')
        if success_num >= 3:
            mean_time = timeCostSum_temp / success_num
            print(f" | Avg Steps to Success: {mean_time:.2f}")
        else:
            print()
        print("="*100 + "\n")
        
        # Reset statistics for next 100 episodes
        meanReward_list.append(temp_rewardSum / 100)
        if success_num >= 3:
            meanTime_list.append(mean_time)
        temp_rewardSum, timeCostSum_temp = 0, 0
        collisionNum_list.append(collision_num)
        collision_num = 0
        collision_obs_num = 0
        collision_wall_num = 0
        found_targets_num = 0
        successNum_list.append(success_num)
        success_num = 0

# Display final statistics
if mode == 'train':
    print("\n" + "="*100)
    print("Final Training Results")
    print("="*100)
    # Calculate final average using last 100 episodes
    final_avg_targets = np.mean(found_targets_list[-100:]) if len(found_targets_list) >= 100 else np.mean(found_targets_list)
    print(f"Total Episodes: {ep_num} | Success Rate: {success_num/ep_num:.2%} | Avg Targets (Last 100): {final_avg_targets:.2f} | Avg Agent Collisions: {collision_num/ep_num:.2f} | Avg Obstacle Collisions: {collision_obs_num/ep_num:.2f} | Avg Wall Collisions: {collision_wall_num/ep_num:.2f} | Avg Steps: {timeCostSum_temp/ep_num:.2f} | Normalized Time: {np.around(timeCostSum_temp/ep_num/MAX_EP_STEPS, decimals=3)}")
    print("="*100 + "\n")
    
    RL.save_Parameters()
    np.savetxt(path+'meanReward.txt', meanReward_list)
    
    # Vẽ đồ thị các thông số huấn luyện
    print("\nVẽ đồ thị các thông số huấn luyện...")
    RL_DQN.plot_training_metrics(save_path=path+'training_metrics.png')
else:
    print("\n" + "="*100)
    print("Final Evaluation Results")
    print("="*100)
    print(f"Total Episodes: {ep_num} | Success Rate: {success_num/ep_num:.2%} | Avg Targets: {found_targets_num/ep_num:.2f} | Avg Agent Collisions: {collision_num/ep_num:.2f} | Avg Obstacle Collisions: {collision_obs_num/ep_num:.2f} | Avg Wall Collisions: {collision_wall_num/ep_num:.2f} | Avg Steps: {timeCostSum_temp/ep_num:.2f} | Normalized Time: {np.around(timeCostSum_temp/ep_num/MAX_EP_STEPS, decimals=3)}")
    print("="*100 + "\n")

print(f"Finished! Running time: {time.time() - timeStart}")
