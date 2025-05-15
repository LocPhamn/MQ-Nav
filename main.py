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
args = parser.parse_args()
mode = args.mode
if mode == 'train':
    ep_num = 10000
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
RL_DQN = Deep_Q_Algo(s_dim,n_actions)
max_torque = env.max_torque
agentSize = env.agentSize
agentPositionArray = np.zeros((agentNum, 2))
agentPositionArray0 = np.zeros((agentNum, 2))
tarPositionArray = np.zeros((agentNum, 2))
tarPositionArray0 = np.zeros((agentNum, 2))
obstacleArray = np.zeros((obsNum, 2))
agentObstacleDisTS = np.ones((agentNum, detecDirNum))
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
# conflict_num = 0  # Comment out conflict counter
found_targets_num = 0
timeStart = time.time()
# ep_list = []
# reward_list = []

for ep in range(ep_num):
    episode += 1
    # Generate obs, agent, tar
    locationArea = np.random.choice(envSize, agentNum * 2 + obsNum, replace=False)
    for i in range(obsNum):
        obstacleArray[i] = [locationArea[i] // 5 * 5 + 2.5, locationArea[i] % 5 * 5 + 2.5] + 1 * np.random.rand(2)
    for i in range(agentNum):
        tarPositionArray0[i] = [locationArea[obsNum + i] // 5 * 5 + 2, locationArea[obsNum + i] % 5 * 5 + 2] + 2 * np.random.rand(2)
        agentPositionArray0[i] = [locationArea[obsNum + agentNum + i] // 5 * 5 + 2, locationArea[obsNum + agentNum + i] % 5 * 5 + 2] + 2 * np.random.rand(2)
    sortTar_index = np.argsort(tarPositionArray0[:, 0])
    for i in range(agentNum):
        tarPositionArray[i, :] = tarPositionArray0[sortTar_index[i], :]
    sortAgent_index = np.argsort(agentPositionArray0[:, 0])
    for i in range(agentNum):
        agentPositionArray[i, :] = agentPositionArray0[sortAgent_index[i], :]
    obstacleSize = np.random.rand(obsNum) * 1.3 + 0.5

    observation = env.reset(agentPositionArray, tarPositionArray, obstacleArray, obstacleSize)
    ep_timeCost = 0
    agentObstacleDis = np.ones((agentNum, detecDirNum))
    agentExistObstacle_Target = np.zeros(agentNum)
    agentExistObstacle_Target_old = np.zeros(agentNum)
    tarAngle = np.zeros(agentNum)
    observationCA = np.hstack((observation[:, :2], agentObstacleDis))
    action = [int(val) for val in np.zeros(agentNum)]
    action_ = [int(val) for val in np.zeros(agentNum)]
    action_h = - np.ones(agentNum, dtype=np.int)
    move = np.zeros((agentNum, 2))
    agentDone = [int(val) for val in np.zeros(agentNum)]
    ep_reward = np.zeros(agentNum)
    observation_h = np.tile(observation[:, -(agentNum - 1) * 2:], historyStep)
    observation_h_temp = observation_h
    collision_cross = np.zeros(agentNum)
    
    # Reset episode statistics
    ep_collision_agent = 0
    ep_collision_obs = 0
    ep_collision_wall = 0
    # ep_conflict = 0  # Comment out conflict update

    for step in range(MAX_EP_STEPS):
        env.render()
        otherTarCoordi = np.zeros((agentNum, 2))
        action_ddpg = np.zeros(agentNum)
        for i in range(agentNum):
            agentExistObstacle_Target[i] = 0
            observationCA[i] = np.hstack((observation[i, action[i]*2: action[i]*2+2], agentObstacleDis[i]))
            action_ddpg[i] = RL_DQN.make_decision(observation[i, action[i]*2: action[i]*2+2])
            actionMove = tarAngle[i] + action_ddpg[i]
            move[i, 0], move[i, 1] = env.step_ddpg(actionMove)

        observation_, reward, done, agentDone, collision_cross, collision_agent, collision_obs, success, conflict, agentPositionArray, collision_wall = \
            env.move(move, agentExistObstacle_Target, otherTarCoordi, action, action_h, 0)

        # Update episode statistics
        ep_collision_agent += collision_agent
        ep_collision_obs += collision_obs
        ep_collision_wall += collision_wall
        # ep_conflict += conflict  # Comment out conflict update
        ep_reward += reward
        ep_timeCost += 1

        # Update total statistics
        collision_num += collision_agent
        collision_obs_num += collision_obs
        collision_wall_num += collision_wall
        success_num += success
        # conflict_num += conflict  # Comment out conflict update
        found_targets_num += np.sum(env.founded_targets)
        temp_rewardSum += min(ep_reward)

        action_h = action
        observation_h_temp[:, 2 * (agentNum - 1):] = observation_h[:, :-2 * (agentNum - 1)]
        observation_h_temp[:, :2 * (agentNum - 1)] = observation[:, -2 * (agentNum - 1):]
        step_total += 1

        if mode == 'train':
            if len(RL_DQN.replay_buffer) > 64:
                RL_DQN.train_main_network(batch_size=64)

        agentExistObstacle_Target_old = agentExistObstacle_Target
        otherTarCoordi = np.zeros((agentNum, 2))

        # Store transitions
        for i in range(agentNum):
            if collision_cross[i] != 1:
                action_[i] = RL.choose_action_dqn(np.hstack((observation_[i], observation_h_temp[i], observation_[i, -2*(agentNum-1):])))
            else:
                action_[i] = action[i]
            agentNextDDPG = 0
            tarAgentCoordi = (observation_[i, 2 * action_[i]: 2 * action_[i] + 2]) * envSize
            tarAgentDis = np.linalg.norm(tarAgentCoordi)
            if tarAgentDis >= 1:
                for k in range(agentNum):
                    if k != action_[i] and 1 < np.linalg.norm(observation_[i, k*2:k*2+2]) * envSize < 2.5:
                        otherTarCoordi[i] = -(observation_[i, k*2:k*2+2]) * envSize
                        break
                tarAgentDirCoordi = tarAgentCoordi / tarAgentDis
                agentNextDDPG, agentObstacleDis[i, 0] = env.detect_obstacle(tarAgentDirCoordi, i, otherTarCoordi[i])
                if agentNextDDPG == 1:
                    tarAngle[i] = math.asin(tarAgentCoordi[0] / tarAgentDis)
                    if tarAgentCoordi[1] >= 0:
                        if tarAgentCoordi[0] >= 0:
                            tarAngle[i] = np.pi - tarAngle[i]
                        if tarAgentCoordi[0] < 0:
                            tarAngle[i] = -np.pi - tarAngle[i]
                    for interval in range(3):
                        tarAngleAround = tarAngle[i] + (interval+1) * np.pi / 6
                        tarAngleAround_PolarCoordi = np.array([np.sin(tarAngleAround), -np.cos(tarAngleAround)])
                        temp, agentObstacleDis[i, interval + 1] = env.detect_obstacle(tarAngleAround_PolarCoordi, i, otherTarCoordi[i])
                    for interval in range(3):
                        tarAngleAround = tarAngle[i] - (interval+1) * np.pi / 6
                        tarAngleAround_PolarCoordi = np.array([np.sin(tarAngleAround), -np.cos(tarAngleAround)])
                        temp, agentObstacleDis[i, interval + 4] = env.detect_obstacle(tarAngleAround_PolarCoordi, i, otherTarCoordi[i])
            observation_All = np.hstack((observation_[i], observation_h_temp[i], observation_[i, -2*(agentNum-1):], agentObstacleDis[i]))
            if mode == 'train' and agentExistObstacle_Target[i] == 1:
                RL_DQN.save_experience(observationCA[i], action_ddpg[i], reward[i], observation_All, done[i])

        observation_h = observation_h_temp
        observation = observation_

        if sum(done) or step == MAX_EP_STEPS - 1:
            if success == 0:
                ep_timeCost = MAX_EP_STEPS
            timeCostSum_temp += ep_timeCost
            
            # Display per-episode statistics
            print(f"\nEpisode {episode} | Steps: {ep_timeCost} | Reward: {np.around(min(ep_reward), decimals=3)} | Success: {'Yes' if success else 'No'} | Found Targets: {np.sum(env.founded_targets)}/{env.agentNum} | Agent Collisions: {ep_collision_agent} | Obstacle Collisions: {ep_collision_obs} | Wall Collisions: {ep_collision_wall}")  # Removed conflict from display
            
            break

    # Display statistics every 100 episodes during training
    if mode == 'train' and episode % 100 == 0:
        print("\n" + "="*100)
        print(f"Training Statistics - Episode {episode}")
        print("="*100)
        print(f"Success Rate: {success_num/100:.2%} | Avg Targets: {found_targets_num/100:.2f} | Avg Agent Collisions: {collision_num/100:.2f} | Avg Obstacle Collisions: {collision_obs_num/100:.2f} | Avg Wall Collisions: {collision_wall_num/100:.2f} | Avg Reward: {temp_rewardSum/100:.2f}", end='')  # Removed conflict from display
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
        # conflict_num = 0  # Comment out conflict reset

# Display final statistics
if mode == 'train':
    print("\n" + "="*100)
    print("Final Training Results")
    print("="*100)
    print(f"Total Episodes: {ep_num} | Success Rate: {success_num/ep_num:.2%} | Avg Targets: {found_targets_num/ep_num:.2f} | Avg Agent Collisions: {collision_num/ep_num:.2f} | Avg Obstacle Collisions: {collision_obs_num/ep_num:.2f} | Avg Wall Collisions: {collision_wall_num/ep_num:.2f} | Avg Steps: {timeCostSum_temp/ep_num:.2f} | Normalized Time: {np.around(timeCostSum_temp/ep_num/MAX_EP_STEPS, decimals=3)}")  # Removed conflict from display
    print("="*100 + "\n")
    
    RL.save_Parameters()
    np.savetxt(path+'meanReward.txt', meanReward_list)
else:
    print("\n" + "="*100)
    print("Final Evaluation Results")
    print("="*100)
    print(f"Total Episodes: {ep_num} | Success Rate: {success_num/ep_num:.2%} | Avg Targets: {found_targets_num/ep_num:.2f} | Avg Agent Collisions: {collision_num/ep_num:.2f} | Avg Obstacle Collisions: {collision_obs_num/ep_num:.2f} | Avg Wall Collisions: {collision_wall_num/ep_num:.2f} | Avg Steps: {timeCostSum_temp/ep_num:.2f} | Normalized Time: {np.around(timeCostSum_temp/ep_num/MAX_EP_STEPS, decimals=3)}")  # Removed conflict from display
    print("="*100 + "\n")

print(f"Finished! Running time: {time.time() - timeStart}")
