import numpy as np
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 20  # grid size
ENV_H = 30  # env height
ENV_W = ENV_H  # env width
halfUnit = UNIT / 2
obsNum = 10
MAX_EP_STEPS = ENV_H * 3  # Maximum steps per episode
color = ['lightblue', 'pink', 'royalblue', 'pink', 'lightblue', 'lightblue']

class ENV(tk.Tk, object):
    def __init__(self, agentNum):
        super(ENV, self).__init__()
        # Initialize arrays
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
        self.current_step = 0
        self.MAX_EP_STEPS = MAX_EP_STEPS

        # Cache for frequently used values
        self.grid_map = np.zeros((ENV_H, ENV_H), dtype=int)
        self.new_cell_point = [[] for _ in range(self.agentNum)]
        self.targets = []
        self.founded_targets = np.zeros(agentNum)
        self.target_rewards_given = np.zeros(agentNum)
        
        # Pre-compute detection angles
        self.detection_angles = np.array([(i+1) * np.pi / 6 for i in range(3)])
        self.detection_angles = np.concatenate([self.detection_angles, -self.detection_angles])
        
        self._build_env()

    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='white',
                              height=ENV_H * UNIT,
                              width=ENV_W * UNIT)
        
        # Create grid using vectorized operations
        x_coords = np.arange(ENV_H) * UNIT
        y_coords = np.arange(ENV_W) * UNIT
        
        # Create vertical lines
        for x in x_coords:
            self.canvas.create_line(x, 0, x, ENV_H * UNIT, fill='lightgray')
        
        # Create horizontal lines
        for y in y_coords:
            self.canvas.create_line(0, y, ENV_W * UNIT, y, fill='lightgray')
        
        # Create outer walls
        self.canvas.create_rectangle(0, 0, UNIT * ENV_W, UNIT * ENV_H, width=3)

        self.origin = np.array([halfUnit, halfUnit])
        
        # Initialize agents and targets
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            self.targets.append(self.tar_center[i])
            self.agent_center[i] = self.origin + UNIT * np.random.rand(2) * (ENV_H - 1 + 0.01)
            
            # Create agent
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize,
                self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize,
                self.agent_center[i, 1] + self.agentSize,
                fill='blue')
            
            # Create target
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize,
                self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize,
                self.tar_center[i, 1] + self.tarSize,
                fill='red')
        
        # Create obstacles
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

    def reset(self, agentPositionArray, tarPositionArray, obsArray, obsSize):
        self.update()
        self.grid_map = np.zeros((ENV_H, ENV_H), dtype=int)
        self.canvas.delete("squares")
        self.canvas.delete("founded_target")
        self.current_step = 0
        self.target_rewards_given = np.zeros(self.agentNum)
        
        # Update statistics display
        self.show_stats()

        # Initialize arrays
        sATAA = np.zeros((self.agentNum, 2 * (2 * self.agentNum - 1)))
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))
        self.founded_targets = np.zeros(self.agentNum)
        
        # Clear existing objects
        for i in range(self.agentNum):
            self.canvas.delete(self.agent_all[i])
            self.canvas.delete(self.target_all[i])
        for i in range(obsNum):
            self.canvas.delete(self.obstacle_all[i])
        
        # Update positions
        self.agentPositionArray = agentPositionArray
        self.tarPositionArray = tarPositionArray
        self.obsArray = obsArray
        self.obsSize = obsSize * UNIT
        
        # Create new agents and targets
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * self.tarPositionArray[i]
            self.agent_center[i] = self.origin + UNIT * self.agentPositionArray[i]
            
            # Create target
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize,
                self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize,
                self.tar_center[i, 1] + self.tarSize,
                fill='red')
            
            # Create agent
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize,
                self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize,
                self.agent_center[i, 1] + self.agentSize,
                fill='blue')
        
        # Create obstacles
        for i in range(int(obsNum/2)):  # Square obstacles
            self.obs_center[i] = self.origin + UNIT * self.obsArray[i]
            self.obstacle_all[i] = self.canvas.create_rectangle(
                self.obs_center[i, 0] - self.obsSize[i],
                self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i],
                self.obs_center[i, 1] + self.obsSize[i],
                fill='lightgrey', outline='lightgrey')
        
        for i in range(int(obsNum/2), obsNum):  # Round obstacles
            self.obs_center[i] = self.origin + UNIT * self.obsArray[i]
            self.obstacle_all[i] = self.canvas.create_oval(
                self.obs_center[i, 0] - self.obsSize[i],
                self.obs_center[i, 1] - self.obsSize[i],
                self.obs_center[i, 0] + self.obsSize[i],
                self.obs_center[i, 1] + self.obsSize[i],
                fill='lightgrey', outline='lightgrey')
        
        # Calculate coordinates
        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + np.array([self.tarSize, self.tarSize])
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])

        # Vectorized calculation of distances
        for i in range(self.agentNum):
            # Calculate distances to all targets
            sATAA[i, :2*self.agentNum] = (tar_coordi - agent_coordi[i]).flatten() / (ENV_H * UNIT)
            
            # Calculate distances to other agents
            for j in range(self.agentNum):
                if j > i:
                    sATAA[i, 2*(self.agentNum + j - 1):2*(self.agentNum + j)] = (agent_coordi[j] - agent_coordi[i]) / (ENV_H * UNIT)
                elif j < i:
                    sATAA[i, 2*(self.agentNum + j):2*(self.agentNum + j)+2] = -sATAA[j, 2*(self.agentNum + i - 1):2*(self.agentNum + i)]
        
        return sATAA

    def detect_obstacle(self, tarAgentDirCoordi, i, otherTarCoordi):
        obstacleExist = 0
        obstacleDistance = 1
        
        # Pre-calculate observation points
        observe_distances = np.linspace(0, self.observeRange, self.observeTimes + 1)
        observe_coords = np.outer(observe_distances, tarAgentDirCoordi)
        
        # Get agent coordinates
        agent_coords = np.array(self.canvas.coords(self.agent_all[i]))
        agent_center = np.array([(agent_coords[0] + agent_coords[2])/2, (agent_coords[1] + agent_coords[3])/2])
        
        # Check each observation point
        for k, observe_coord in enumerate(observe_coords):
            A_coordi = agent_center + observe_coord
            
            # Check square obstacles
            for j in range(int(self.obsNum/2)):
                Ob_coordi = self.canvas.coords(self.obstacle_all[j])
                if not (A_coordi[0] + self.agentSize <= Ob_coordi[0] or
                       A_coordi[0] - self.agentSize >= Ob_coordi[2] or
                       A_coordi[1] + self.agentSize <= Ob_coordi[1] or
                       A_coordi[1] - self.agentSize >= Ob_coordi[3]):
                    obstacleExist = 1
                    obstacleDistance = observe_distances[k] / self.observeRange
                    return obstacleExist, obstacleDistance
            
            # Check round obstacles
            for j in range(int(self.obsNum/2), self.obsNum):
                Ob_coordi = np.array(self.canvas.coords(self.obstacle_all[j])[:2]) + self.obsSize[j]
                if np.linalg.norm(Ob_coordi - A_coordi) <= (self.obsSize[j] + self.agentSize):
                    obstacleExist = 1
                    obstacleDistance = observe_distances[k] / self.observeRange
                    return obstacleExist, obstacleDistance
            
            # Check other targets
            if np.linalg.norm(otherTarCoordi) != 0:
                if np.linalg.norm(otherTarCoordi + observe_coord) * UNIT <= (self.tarSize + self.agentSize):
                    obstacleExist = 1
                    obstacleDistance = observe_distances[k] / self.observeRange
                    return obstacleExist, obstacleDistance
        
        return obstacleExist, obstacleDistance

    def mark_detection_area(self, grid_matrix, agent_pos, observe_range, unit):
        # Convert agent position to grid coordinates
        agent_row, agent_col = int(agent_pos[1] // unit), int(agent_pos[0] // unit)
        new_cells = 0
        
        # Fixed detection radius of 4 cells
        detection_radius = 4
        
        # Create grid of coordinates to check
        rows = np.arange(-detection_radius, detection_radius + 1)
        cols = np.arange(-detection_radius, detection_radius + 1)
        row_grid, col_grid = np.meshgrid(rows, cols)
        
        # Calculate Manhattan distances
        manhattan_distances = np.abs(row_grid) + np.abs(col_grid)
        valid_cells = manhattan_distances <= detection_radius
        
        # Get valid coordinates
        valid_rows = agent_row + row_grid[valid_cells]
        valid_cols = agent_col + col_grid[valid_cells]
        
        # Filter coordinates within grid bounds
        valid_mask = (valid_rows >= 0) & (valid_rows < grid_matrix.shape[0]) & \
                    (valid_cols >= 0) & (valid_cols < grid_matrix.shape[1])
        
        valid_rows = valid_rows[valid_mask]
        valid_cols = valid_cols[valid_mask]
        
        # Update grid
        for row, col in zip(valid_rows, valid_cols):
            if grid_matrix[row, col] == 0:
                new_cells += 1
                grid_matrix[row, col] = 1
        
        return grid_matrix, new_cells

    def detect_targets(self, grid_matrix, tar_pos, observe_range, unit):
        tar_row, tar_col = int(tar_pos[1] // unit), int(tar_pos[0] // unit)
        
        # Check if target is within grid bounds
        if not (0 <= tar_row < grid_matrix.shape[0] and 0 <= tar_col < grid_matrix.shape[1]):
            return 0
        
        return int(grid_matrix[tar_row, tar_col] == 1)

    def mark_target(self, grid_matrix, observe_range, unit):
        check = 0
        for idx in range(len(self.targets)):
            tar_pos = self.targets[idx]
            if self.detect_targets(grid_matrix, tar_pos, observe_range, unit) and self.founded_targets[idx] == 0:
                self.canvas.create_rectangle(
                    tar_pos[0] - self.tarSize,
                    tar_pos[1] - self.tarSize,
                    tar_pos[0] + self.tarSize,
                    tar_pos[1] + self.tarSize,
                    fill='pink',
                    tags="founded_target")
                self.founded_targets[idx] = 1
                check = 1
        return self.founded_targets, check

    def step_ddpg(self, action):
        return np.array([
            np.sin(action) * self.stepLength,
            -np.cos(action) * self.stepLength
        ])

    def step_dqn(self, action, observation, agentiDone):
        if agentiDone != action + 1:
            direction = observation[action*2:(action+1)*2]
            norm = np.linalg.norm(direction)
            if norm > 0:
                return direction / norm * self.stepLengthFree
        return np.zeros(2)

    def show_stats(self):
        """Display current statistics on the canvas"""
        if hasattr(self, 'stats_text'):
            self.canvas.delete(self.stats_text)
        
        exploration_ratio = np.sum(self.grid_map) / (ENV_H * ENV_W)
        found_targets = np.sum(self.founded_targets)
        
        stats = f'Step: {self.current_step}/{self.MAX_EP_STEPS} | Explored: {exploration_ratio:.1%} | Found Targets: {found_targets}/{self.agentNum}'
        
        self.stats_text = self.canvas.create_text(
            ENV_W * UNIT / 2, 20,
            text=stats,
            fill='black',
            font=('Helvetica', 10, 'bold')
        )
        self.canvas.tag_raise(self.stats_text)

    def draw_grid(self, grid_matrix, unit):
        """
        Draw explored areas on the grid.
        
        Args:
            grid_matrix: Matrix representing explored areas
            unit: Size of each grid cell
        """
        # Clear previous grid visualization
        self.canvas.delete("squares")
        
        # Get coordinates of explored cells
        explored_cells = np.where(grid_matrix == 1)
        rows, cols = explored_cells
        
        # Draw explored cells
        for row, col in zip(rows, cols):
            x0, y0 = col * UNIT, row * UNIT
            x1, y1 = x0 + UNIT, y0 + UNIT
            self.canvas.create_rectangle(
                x0, y0, x1, y1, 
                fill="lightgreen", 
                width=0,
                tags="squares"
            )
        
        # Put explored area behind grid lines
        self.canvas.tag_lower("squares")

    def move(self, move, agentExistObstacle_Target, otherTarCoordi, action, action_h, drawTrajectory):
        # Initialize arrays
        done_collision_cross = np.zeros(self.agentNum)
        done_collision_agent = 0
        done_collision_obs = 0
        done_collision_wall = 0
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
        self.show_stats()

        # Pre-calculate all coordinates
        agent_coords = np.array([self.canvas.coords(agent) for agent in self.agent_all])
        agent_centers = np.column_stack((
            (agent_coords[:, 0] + agent_coords[:, 2]) / 2,
            (agent_coords[:, 1] + agent_coords[:, 3]) / 2
        ))
        
        # Check if agents are staying in the same position
        same_position = np.all(agent_centers == self.prev_positions, axis=1)
        reward[same_position] -= 0.5
        
        # Update previous positions
        self.prev_positions = agent_centers.copy()
        
        # Calculate new positions
        new_positions = agent_centers + move
        
        # Check wall collisions
        wall_collision = (
            (new_positions[:, 0] - self.agentSize <= 0) |
            (new_positions[:, 0] + self.agentSize >= ENV_W * UNIT) |
            (new_positions[:, 1] - self.agentSize <= 0) |
            (new_positions[:, 1] + self.agentSize >= ENV_H * UNIT)
        )
        
        # Handle wall collisions
        move[wall_collision] = 0
        reward[wall_collision] -= 1.0
        done_collision_wall = np.sum(wall_collision)

        # Process each agent
        for i in range(self.agentNum):
            if wall_collision[i]:
                continue

            new_pos = new_positions[i]
            new_agent_coords = [
                new_pos[0] - self.agentSize,
                new_pos[1] - self.agentSize,
                new_pos[0] + self.agentSize,
                new_pos[1] + self.agentSize
            ]

            # Check square obstacles
            square_obs_coords = np.array([self.canvas.coords(obs) for obs in self.obstacle_all[:int(self.obsNum/2)]])
            square_collisions = ~(
                (new_agent_coords[2] <= square_obs_coords[:, 0]) |
                (new_agent_coords[0] >= square_obs_coords[:, 2]) |
                (new_agent_coords[3] <= square_obs_coords[:, 1]) |
                (new_agent_coords[1] >= square_obs_coords[:, 3])
            )
            
            if np.any(square_collisions):
                move[i] = 0
                reward[i] -= 1.0
                done_collision_obs += 1
                continue

            # Check round obstacles
            round_obs_coords = np.array([self.canvas.coords(obs) for obs in self.obstacle_all[int(self.obsNum/2):]])
            round_obs_centers = np.column_stack((
                (round_obs_coords[:, 0] + round_obs_coords[:, 2]) / 2,
                (round_obs_coords[:, 1] + round_obs_coords[:, 3]) / 2
            ))
            
            distances = np.linalg.norm(new_pos - round_obs_centers, axis=1)
            if np.any(distances < (self.agentSize + self.obsSize[int(self.obsNum/2):])):
                move[i] = 0
                reward[i] -= 1.0
                done_collision_obs += 1
                continue

            # Check agent collisions
            other_agents = np.delete(agent_centers, i, axis=0)
            distances = np.linalg.norm(new_pos - other_agents, axis=1)
            if np.any(distances < (2 * self.agentSize)):
                move[i] = 0
                reward[i] -= 1.0
                reward[np.where(distances < (2 * self.agentSize))[0] + (i if i == 0 else 0)] -= 1.0
                done_collision_agent += 1
                continue

            # Update exploration
            agent_center = (int(new_pos[0]), int(new_pos[1]))
            self.grid_map, new_cells = self.mark_detection_area(self.grid_map, agent_center, self.observeRange, UNIT)
            new_cell_reward = (new_cells*100)/(ENV_H*ENV_W)
            self.new_cell_point[i].append(new_cell_reward)

            # Check target detection
            founded_targets_move, check = self.mark_target(self.grid_map, self.observeRange, UNIT)
            if check == 1:
                searcher.append(i)

            # Move agent
            self.canvas.move(self.agent_all[i], move[i, 0], move[i, 1])

        # Update grid visualization
        self.draw_grid(self.grid_map, UNIT)
        self.canvas.tag_lower("squares")

        # Calculate new coordinates
        agent_coords = np.array([self.canvas.coords(agent) for agent in self.agent_all])
        agent_centers = np.column_stack((
            (agent_coords[:, 0] + agent_coords[:, 2]) / 2 + self.agentSize,
            (agent_coords[:, 1] + agent_coords[:, 3]) / 2 + self.agentSize
        ))

        # Calculate distances to targets and other agents
        tar_coords = np.array([self.canvas.coords(target) for target in self.target_all])
        tar_centers = np.column_stack((
            (tar_coords[:, 0] + tar_coords[:, 2]) / 2 + self.tarSize,
            (tar_coords[:, 1] + tar_coords[:, 3]) / 2 + self.tarSize
        ))

        # Vectorized calculation of sATAA
        for i in range(self.agentNum):
            # Calculate distances to all targets
            sATAA[i, :2*self.agentNum] = (tar_centers - agent_centers[i]).flatten() / (ENV_H * UNIT)
            
            # Calculate distances to other agents
            for j in range(self.agentNum):
                if j > i:
                    sATAA[i, 2*(self.agentNum + j - 1):2*(self.agentNum + j)] = (agent_centers[j] - agent_centers[i]) / (ENV_H * UNIT)
                elif j < i:
                    sATAA[i, 2*(self.agentNum + j):2*(self.agentNum + j)+2] = -sATAA[j, 2*(self.agentNum + i - 1):2*(self.agentNum + i)]

        # Check if agents reached their targets
        target_distances = np.linalg.norm(agent_centers - tar_centers[action], axis=1)
        reached_targets = target_distances < self.stepLengthFree
        agentDone[reached_targets] = action[reached_targets] + 1

        # Check success conditions
        found_targets_count = np.sum(self.founded_targets)
        if found_targets_count >= 3:
            success = 1
            done = np.ones(self.agentNum)
            remaining_steps = self.MAX_EP_STEPS - self.current_step
            step_ratio = remaining_steps / self.MAX_EP_STEPS
            bonus_reward = 1000 * step_ratio  # Increased base bonus
            reward += bonus_reward
        elif np.sum(agentDone > 0) == self.agentNum:
            success = 1
            done = np.ones(self.agentNum)
            remaining_steps = self.MAX_EP_STEPS - self.current_step
            step_ratio = remaining_steps / self.MAX_EP_STEPS
            bonus_reward = 2000 * step_ratio  # Even larger bonus for finding all targets
            reward += bonus_reward

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

        # Calculate exploration ratio
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

        # Calculate final positions
        agentNewPosition = agent_centers / UNIT - self.origin / UNIT

        return sATAA, reward, done, agentDone, done_collision_cross, done_collision_agent, done_collision_obs, success, arriveSame, agentNewPosition, done_collision_wall

    def render(self):
        self.update()
