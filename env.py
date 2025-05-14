import numpy as np
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 20  # grid size
ENV_H = 36  # env height
ENV_W = ENV_H  # env width
halfUnit = UNIT / 2
obsNum = 10
MAX_EP_STEPS = ENV_H * 3  # Maximum steps per episode
color = ['lightblue', 'pink', 'royalblue', 'pink', 'lightblue', 'lightblue']
class ENV(tk.Tk, object):
    def __init__(self, agentNum):
        super(ENV, self).__init__()
        # Khởi tạo danh sách lưu vị trí trước đó của agent để tính vận tốc
        self.prev_positions = np.zeros((agentNum, 2))
        self.agentNum = agentNum
        self.ENV_H = ENV_H
        self.obsNum = obsNum
        self.n_actions_ts = self.agentNum
        self.historyStep = 1
        self.n_states_ts = 2 * (2 * self.agentNum-1) + 2*(self.agentNum-1)*self.historyStep*2
        self.n_states_ca = 7  # number of detecting directions
        self.n_actions_ca = 1
        self.max_torque = np.pi / 2
        self.stepLength = UNIT*0.5
        self.stepLengthFree = UNIT*1
        self.observeRange = 4  # detection radius
        self.observeTimes = 400
        self.agent_all = [None] * self.agentNum
        self.target_all = [None] * self.agentNum
        self.obstacle_all = [None] * obsNum
        self.agentSize = 0.25*UNIT
        self.tarSize = 0.25*UNIT
        self.obsSize = np.ones(obsNum)
        self.agent_center = np.zeros((self.agentNum, 2))
        self.tar_center = np.zeros((self.agentNum, 2))
        self.obs_center = np.zeros((obsNum, 2))
        self.geometry('{0}x{1}'.format(ENV_H * UNIT, ENV_H * UNIT))
        self.current_step = 0  # Add step counter
        self.MAX_EP_STEPS = MAX_EP_STEPS  # Add MAX_EP_STEPS to class

        #new parameters
        self.grid_map = np.zeros((ENV_H, ENV_H),dtype=int)
        self.new_cell_point = [[] for _ in range(self.agentNum)]  # Initialize empty lists for each agent
        self.targets = []
        self.founded_targets = np.zeros(agentNum)
        self.target_rewards_given = np.zeros(agentNum)  # Track which targets have been rewarded
        self._build_env()

    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=ENV_H * UNIT,
                                width=ENV_W * UNIT)
        # Create grid
        for i in range(ENV_H):
            # Vertical lines
            self.canvas.create_line(i * UNIT, 0, i * UNIT, ENV_H * UNIT, fill='lightgray')
            # Horizontal lines
            self.canvas.create_line(0, i * UNIT, ENV_W * UNIT, i * UNIT, fill='lightgray')
            
        # Create outer walls (thicker)
        self.canvas.create_rectangle(0, 0, UNIT * ENV_W, UNIT * ENV_H, width=3)

        self.origin = np.array([halfUnit, halfUnit])
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            self.targets.append(self.tar_center[i])
            self.agent_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, 
                self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, 
                self.agent_center[i, 1] + self.agentSize,
                fill='blue')
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize, 
                self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize, 
                self.tar_center[i, 1] + self.tarSize,
                fill='red')
        for i in range(int(obsNum/2)):  # Square obstacles
            self.obs_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H/2)
            self.obstacle_all[i] = self.canvas.create_rectangle(
                self.obs_center[i, 0] - self.obsSize[i], 
                self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], 
                self.obs_center[i, 1] + self.obsSize[i],
                fill='grey')
        for i in range(int(obsNum/2), obsNum):  # Round obstacles
            self.obs_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H/2)
            self.obstacle_all[i] = self.canvas.create_oval(
                self.obs_center[i, 0] - self.obsSize[i], 
                self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], 
                self.obs_center[i, 1] + self.obsSize[i],
                fill='grey')
        self.canvas.pack()

    def show_episode_info(self, episode, exploration_ratio):
        """
        Display episode information on the canvas.
        
        Args:
            episode: Current episode number
            exploration_ratio: Current exploration progress
        """
        # Clear previous info if exists
        if hasattr(self, 'info_text'):
            self.canvas.delete(self.info_text)
        
        # Show episode number and exploration ratio
        self.info_text = self.canvas.create_text(
            ENV_W * UNIT / 2, 10,
            text=f'Episode: {episode} | Explored: {exploration_ratio:.1%}',
            fill='black',
            font=('Helvetica', 10, 'bold')
        )

    def draw_grid(self, grid_matrix, unit):
        """
        Draw explored areas on the grid.
        
        Args:
            grid_matrix: Matrix representing explored areas
            unit: Size of each grid cell
        """
        for row in range(ENV_H):
            for col in range(ENV_W):
                if grid_matrix[row, col] == 1:  # If cell is explored
                    x0, y0 = col * UNIT, row * UNIT
                    x1, y1 = x0 + UNIT, y0 + UNIT
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, 
                        fill="lightgreen", 
                        width=0,
                        tags="squares"
                    )
                    self.canvas.tag_lower("squares")  # Put explored area behind grid lines

    def reset(self, agentPositionArray, tarPositionArray, obsArray, obsSize):
        self.update()
        self.grid_map = np.zeros((ENV_H, ENV_H),dtype=int)
        self.canvas.delete("squares")
        self.canvas.delete("founded_target")
        self.current_step = 0  # Reset step counter
        self.target_rewards_given = np.zeros(self.agentNum)  # Reset target rewards tracking

        sATAA = np.zeros((self.agentNum, 2 * (2 * self.agentNum - 1)))
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))
        self.founded_targets = np.zeros(self.agentNum)
        for i in range(self.agentNum):
            self.canvas.delete(self.agent_all[i])
            self.canvas.delete(self.target_all[i])
        for i in range(obsNum):
            self.canvas.delete(self.obstacle_all[i])
        self.agentPositionArray = agentPositionArray
        self.tarPositionArray = tarPositionArray
        self.obsArray = obsArray
        self.obsSize = obsSize * UNIT
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * self.tarPositionArray[i]
            # self.targets.append(self.tar_center[i])
            self.agent_center[i] = self.origin + UNIT * self.agentPositionArray[i]
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize, self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize, self.tar_center[i, 1] + self.tarSize,
                fill='red')
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, self.agent_center[i, 1] + self.agentSize,
                fill='blue')
        for i in range(int(obsNum/2)):  # Square obstacles
            self.obs_center[i] = self.origin + UNIT * self.obsArray[i]
            self.obstacle_all[i] = self.canvas.create_rectangle(
                self.obs_center[i, 0] - self.obsSize[i], self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], self.obs_center[i, 1] + self.obsSize[i],
                fill='lightgrey', outline='lightgrey')
        for i in range(int(obsNum/2), obsNum):  # Round obstacles
            self.obs_center[i] = self.origin + UNIT * self.obsArray[i]
            self.obstacle_all[i] = self.canvas.create_oval(
                self.obs_center[i, 0] - self.obsSize[i], self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i], self.obs_center[i, 1] + self.obsSize[i],
                fill='lightgrey', outline='lightgrey')
        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + np.array([self.tarSize, self.tarSize])
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])

        for i in range(self.agentNum):
            for k in range(self.agentNum):  # distances between agents and targets
                sATAA[i, 2*k: 2*(k+1)] = (tar_coordi[k] - agent_coordi[i]) / (ENV_H * UNIT)
            for j in range(self.agentNum):
                if j > i:
                    sATAA[i, 2*(self.agentNum + j - 1): 2*(self.agentNum + j)] = (agent_coordi[j] - agent_coordi[i]) / (ENV_H * UNIT)
                elif j < i:
                    sATAA[i, 2*(self.agentNum + j): 2*(self.agentNum + j)+2] = - sATAA[j, 2*(self.agentNum + i - 1): 2*(self.agentNum + i)]
        return sATAA

    def detect_obstacle(self, tarAgentDirCoordi, i, otherTarCoordi):
        obstacleExist = 0
        obstacleDistance = 1
        for k in range(self.observeTimes + 1):
            observeDistance = k / self.observeTimes * self.observeRange
            observeCoordi = observeDistance * tarAgentDirCoordi
            A_coordi = self.canvas.coords(self.agent_all[i]) + UNIT * np.hstack((np.hstack(observeCoordi), np.hstack(observeCoordi)))
            for j in range(0, obsNum+1):
                if j < int(obsNum/2):  # Detecting square obstacles
                    Ob_coordi = self.canvas.coords(self.obstacle_all[j])
                    agentNonObstacle = A_coordi[2] <= Ob_coordi[0] or A_coordi[0] >= Ob_coordi[2] or A_coordi[3] <= Ob_coordi[1] or A_coordi[1] >= Ob_coordi[3]
                    if agentNonObstacle == 0:
                        obstacleExist = 1
                        obstacleDistance = observeDistance / self.observeRange
                        break
                elif j < obsNum:  # Detecting round obstacles
                    Ob_coordi = self.canvas.coords(self.obstacle_all[j])
                    agentNonObstacle = np.linalg.norm(Ob_coordi[:2]+self.obsSize[j]-A_coordi[:2]-self.agentSize) > (self.obsSize[j] + self.agentSize)
                    if agentNonObstacle == 0:
                        obstacleExist = 1
                        obstacleDistance = observeDistance / self.observeRange
                        break
                elif j == obsNum:  # Detecting targets
                    if np.linalg.norm(otherTarCoordi) != 0:
                        agentNonObstacle = np.linalg.norm(otherTarCoordi + observeCoordi)*UNIT > (self.tarSize + self.agentSize)
                        if agentNonObstacle == 0:
                            obstacleExist = 1
                            obstacleDistance = observeDistance / self.observeRange
                            break
            if agentNonObstacle == 0:
                break
        return obstacleExist, obstacleDistance

    def mark_detection_area(self,grid_matrix, agent_pos, observe_range, unit):
        # Chuyển đổi tọa độ tác nhân sang tọa độ ô lưới
        agent_row, agent_col = int(agent_pos[1] // unit), int(agent_pos[0] // unit)
        new_cells = 0

        # Bán kính phát hiện tính theo số ô lưới
        detection_radius = int(observe_range * UNIT)

        # Duyệt qua các ô nằm trong phạm vi bán kính
        for r in range(-detection_radius, detection_radius + 1):
            for c in range(-detection_radius, detection_radius + 1):
                # Tính tọa độ lưới của ô hiện tại
                row = agent_row + r
                col = agent_col + c

                # Kiểm tra xem ô có nằm trong lưới không
                if 0 <= row < grid_matrix.shape[0] and 0 <= col < grid_matrix.shape[1]:
                    # Kiểm tra ô có nằm trong vùng tròn phát hiện
                    distance = (r) ** 2 + (c) ** 2
                    if distance <= observe_range ** 2:
                        if grid_matrix[row, col] == 0:
                            new_cells += 1
                            grid_matrix[row, col] = 1 # Đánh dấu vùng phát hiện bằng giá trị 1
        return grid_matrix,new_cells

    def detect_targets(self,grid_matrix,tar_pos,observe_range,unit):
        tar_row, tar_col = int(tar_pos[1] // unit), int(tar_pos[0] // unit)

        # Nếu ô của target nằm ngoài lưới thì trả về False
        if tar_row < 0 or tar_row >= grid_matrix.shape[0] or tar_col < 0 or tar_col >= grid_matrix.shape[1]:
            return 0

        # Kiểm tra giá trị của ô trong grid_map
        if grid_matrix[tar_row, tar_col] == 1:
            return 1
        else:
            return 0

    def mark_target(self, grid_matrix, observe_range, unit):
        check = 0
        for idx in range(len(self.targets)):
            tar_pos = self.targets[idx]
            if self.detect_targets(grid_matrix, tar_pos, observe_range, unit) and self.founded_targets[idx] == 0:
                print(f"target {idx} đã được tìm thấy tại tọa độ: {tar_pos}")
                self.canvas.create_rectangle(
                    tar_pos[0] - self.tarSize, tar_pos[1] - self.tarSize,
                    tar_pos[0] + self.tarSize, tar_pos[1] + self.tarSize,
                    fill='pink',tags= "founded_target")
                self.founded_targets[idx] = 1
                check = 1
        return self.founded_targets, check

    def step_ddpg(self, action):
        base_actionA = np.array([0.0, 0.0])
        base_actionA[0] += np.sin(action) * self.stepLength
        base_actionA[1] -= np.cos(action) * self.stepLength
        return base_actionA[0], base_actionA[1]

    def step_dqn(self, action, observation, agentiDone):
        base_actionA = np.array([0.0, 0.0])
        if agentiDone != action + 1:
            base_actionA += observation[action*2: (action+1)*2]/np.linalg.norm(observation[action*2:(action+1)*2])*self.stepLengthFree
        return base_actionA[0], base_actionA[1]

    def move(self, move, agentExistObstacle_Target, otherTarCoordi, action, action_h, drawTrajectory):
        # Initialize arrays
        done_collision_cross = np.zeros(self.agentNum)
        done_collision_agent = 0
        done_collision_obs = 0
        success = 0
        arriveSame = 0
        done = np.zeros(self.agentNum)
        reward = -1 * np.ones(self.agentNum)/ENV_H
        agentNewPosition = np.zeros((self.agentNum, 2))
        searcher = []
        sATAA = np.zeros((self.agentNum, 2*(2*self.agentNum - 1)))
        agentDone = np.zeros(self.agentNum, dtype=int)

        # Update step counter
        self.current_step += 1

        # Pre-calculate all coordinates once
        agent_coords = np.array([self.canvas.coords(agent) for agent in self.agent_all])
        agent_centers = np.column_stack((
            (agent_coords[:, 0] + agent_coords[:, 2]) / 2,
            (agent_coords[:, 1] + agent_coords[:, 3]) / 2
        ))
        
        tar_coords = np.array([self.canvas.coords(target) for target in self.target_all])
        tar_centers = np.column_stack((
            (tar_coords[:, 0] + tar_coords[:, 2]) / 2 + self.tarSize,
            (tar_coords[:, 1] + tar_coords[:, 3]) / 2 + self.tarSize
        ))

        # Pre-calculate obstacle coordinates and types
        obs_coords = np.array([self.canvas.coords(obs) for obs in self.obstacle_all])
        obs_centers = np.column_stack((
            (obs_coords[:, 0] + obs_coords[:, 2]) / 2,
            (obs_coords[:, 1] + obs_coords[:, 3]) / 2
        ))
        square_obs_mask = np.arange(obsNum) < int(obsNum/2)
        round_obs_mask = ~square_obs_mask

        # Calculate new positions for all agents at once
        new_positions = agent_centers + move
        new_grid_positions = new_positions / UNIT

        # Check wall collisions for all agents at once
        wall_collision = (
            (new_grid_positions[:, 0] <= 0) | 
            (new_grid_positions[:, 0] >= ENV_W-1) |
            (new_grid_positions[:, 1] <= 0) | 
            (new_grid_positions[:, 1] >= ENV_H-1)
        )
        move[wall_collision] = [0, 0]
        reward[wall_collision] -= 1.0

        # Process each agent for obstacle collisions and exploration
        for i in range(self.agentNum):
            if wall_collision[i]:
                continue

            new_x, new_y = new_positions[i]
            new_agent_coords = [
                new_x - self.agentSize,
                new_y - self.agentSize,
                new_x + self.agentSize,
                new_y + self.agentSize
            ]

            # Check square obstacles using vectorized operations
            square_obs_coords = obs_coords[square_obs_mask]
            square_collisions = ~(
                (new_agent_coords[2] <= square_obs_coords[:, 0]) |
                (new_agent_coords[0] >= square_obs_coords[:, 2]) |
                (new_agent_coords[3] <= square_obs_coords[:, 1]) |
                (new_agent_coords[1] >= square_obs_coords[:, 3])
            )
            if np.any(square_collisions):
                move[i] = [0, 0]
                reward[i] -= 1.0
                continue

            # Check round obstacles using vectorized distance calculation
            round_obs_centers = obs_centers[round_obs_mask]
            round_obs_sizes = self.obsSize[round_obs_mask]
            distances = np.sqrt(
                (new_x - round_obs_centers[:, 0])**2 + 
                (new_y - round_obs_centers[:, 1])**2
            )
            if np.any(distances < (self.agentSize + round_obs_sizes)):
                move[i] = [0, 0]
                reward[i] -= 1.0
                continue

            # Update exploration
            agent_center = (int(new_x), int(new_y))
            self.grid_map, new_cells = self.mark_detection_area(self.grid_map, agent_center, self.observeRange, UNIT)
            new_cell_reward = (new_cells*100)/(ENV_H*ENV_W)
            self.new_cell_point[i].append(new_cell_reward)

            # Check target detection
            founded_targets_move, check = self.mark_target(self.grid_map, self.observeRange, UNIT)
            if check == 1:
                searcher.append(i)

            # Move agent
            self.canvas.move(self.agent_all[i], move[i, 0], move[i, 1])

        # Update grid visualization once
        self.draw_grid(self.grid_map, UNIT)
        self.canvas.tag_lower("squares")

        # Calculate sATAA using vectorized operations
        agent_coords = np.array([self.canvas.coords(agent) for agent in self.agent_all])
        agent_centers = np.column_stack((
            (agent_coords[:, 0] + agent_coords[:, 2]) / 2 + self.agentSize,
            (agent_coords[:, 1] + agent_coords[:, 3]) / 2 + self.agentSize
        ))

        # Vectorized calculation of distances between agents and targets
        for i in range(self.agentNum):
            # Calculate distances to all targets
            for k in range(self.agentNum):
                sATAA[i, 2*k:2*(k+1)] = (tar_centers[k] - agent_centers[i]) / (ENV_H * UNIT)
            
            # Calculate distances to other agents
            for j in range(self.agentNum):
                if j > i:
                    sATAA[i, 2*(self.agentNum + j - 1):2*(self.agentNum + j)] = (agent_centers[j] - agent_centers[i]) / (ENV_H * UNIT)
                elif j < i:
                    sATAA[i, 2*(self.agentNum + j):2*(self.agentNum + j)+2] = -sATAA[j, 2*(self.agentNum + i - 1):2*(self.agentNum + i)]

        # Check if agents reached their targets using vectorized operations
        target_distances = np.linalg.norm(agent_centers - tar_centers[action], axis=1)
        reached_targets = target_distances < self.stepLengthFree
        
        # Update agentDone for agents that reached their targets
        for i in range(self.agentNum):
            if reached_targets[i]:
                agentDone[i] = action[i] + 1
                
        if np.sum(agentDone > 0) == self.agentNum:
            success = 1
            done = np.ones(self.agentNum)

        # Calculate rewards
        for i in searcher:
            if not self.target_rewards_given[i]:
                reward[i] += 200
                self.target_rewards_given[i] = 1

        # Add time penalty
        time_penalty = -0.5 * (self.current_step / self.MAX_EP_STEPS)
        reward += time_penalty

        # Add exploration rewards
        for i in range(self.agentNum):
            reward[i] += sum(self.new_cell_point[i])

        # Add movement reward
        reward += 0.1

        # Calculate exploration ratio once
        exploration_ratio = np.sum(self.grid_map) / (ENV_H * ENV_W)
        
        # Apply exploration multipliers
        if exploration_ratio >= 0.5 and exploration_ratio < 0.75:
            reward *= 1.2
        elif exploration_ratio >= 0.75 and exploration_ratio < 1.0:
            reward *= 1.5
        elif exploration_ratio == 1.0:
            reward *= 2.0
            success = 1
            done = np.ones(self.agentNum)

        # Calculate final agent positions
        agentNewPosition = agent_centers / UNIT - self.origin / UNIT

        return sATAA, reward, done, agentDone, done_collision_cross, done_collision_agent, done_collision_obs, success, arriveSame, agentNewPosition

    def render(self):
        self.update()
