"""
Multi-Agent Deep Q-Network Environment for Area Exploration

This module implements a custom environment for multiple agents exploring an unknown area.
The environment is built using Tkinter for visualization and provides:
- A grid-based world where multiple agents can move and explore
- Detection range mechanics for each agent
- Reward system based on exploration progress and agent cooperation
- Visualization of explored areas and agent movements

Key Parameters:
    ENV_H: Environment height (grid cells)
    ENV_W: Environment width (grid cells)
    UNIT: Size of each grid cell in pixels
    MAX_EP_STEPS: Maximum steps per episode
    DETECTION_RANGE: Manhattan distance for agent detection
    OVERLAP_PENALTY: Penalty factor for overlapping detection ranges
"""

import numpy as np
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import time

# Environment configuration constants
ENV_H = 15  # env height in grid cells
ENV_W = ENV_H  # env width in grid cells
UNIT = 20  # pixel size of each grid cell
HalfUnit = UNIT / 2
MAX_EP_STEPS = ENV_H * 2  # Maximum steps allowed per episode
DETECTION_RANGE = 2  # Manhattan distance for detection range
OVERLAP_PENALTY = 0.2  # Penalty factor for overlapping detection ranges

class ENV(tk.Tk, object):
    """
    Multi-agent exploration environment class.
    
    This class creates and manages the environment where agents explore an unknown area.
    It handles agent movements, exploration tracking, and reward calculations.
    """
    
    def __init__(self, agentNum):
        """
        Initialize the environment with specified number of agents.
        
        Args:
            agentNum: Number of agents in the environment
        """
        super(ENV, self).__init__()
        self.ENV_H = ENV_H
        self.agentNum = agentNum
        self.n_actions = 4  # up, down, left, right
        self.historyStep = 1
        self.n_features = 2 + ENV_H * ENV_W  # agent position + explored map
        self.agent_all = [None] * self.agentNum
        self.explored_cells = []  # Store explored cell rectangles
        self.agentSize = 0.25 * UNIT
        self.agent_center = np.zeros((self.agentNum, 2))
        self.explored_map = np.zeros((ENV_H, ENV_W))  # 0: unexplored, 1: explored
        self.geometry('{0}x{1}'.format(ENV_H * UNIT, ENV_H * UNIT))
        self._build_env()

    def _build_env(self):
        """
        Build and initialize the environment visualization.
        Creates the canvas, grid lines, walls, and randomly places agents.
        """
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

        # Initialize agents at random positions (away from walls)
        self.origin = np.array([HalfUnit, HalfUnit])
        for i in range(self.agentNum):
            # Random position (1 to ENV_H-2 to avoid walls)
            pos = np.random.randint(1, ENV_H-1, size=2)
            self.agent_center[i] = self.origin + UNIT * pos
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, 
                self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, 
                self.agent_center[i, 1] + self.agentSize,
                fill='blue')
        
        self.canvas.pack()

    def reset(self):
        """
        Reset the environment to initial state.
        
        Clears exploration history, resets agents to random positions,
        and initializes tracking variables for the new episode.
        
        Returns:
            numpy.ndarray: Initial state for each agent
        """
        self.update()
        self.explored_map = np.zeros((ENV_H, ENV_W))
        self.current_step = 0  # Reset step counter
        self.previous_explored_count = 0  # Track previous explored count
        
        # Clear previous explored cells
        for cell in self.explored_cells:
            self.canvas.delete(cell)
        self.explored_cells = []
        
        # Reset agents to random positions
        for i in range(self.agentNum):
            self.canvas.delete(self.agent_all[i])
            pos = np.random.randint(1, ENV_H-1, size=2)
            self.agent_center[i] = self.origin + UNIT * pos
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, 
                self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, 
                self.agent_center[i, 1] + self.agentSize,
                fill='blue')
            
            # Mark initial explored areas
            self._update_explored_area(i)
        
        # Store initial exploration count
        self.previous_explored_count = np.sum(self.explored_map)
        
        # Calculate initial exploration ratio
        exploration_ratio = np.sum(self.explored_map) / (ENV_H * ENV_W)
        
        # Show initial state
        self.show_episode_info(0, exploration_ratio)
            
        # Create state for each agent
        state = np.zeros((self.agentNum, self.n_features))
        for i in range(self.agentNum):
            state[i] = self._get_state(i)
            
        return state

    def _get_state(self, agent_idx):
        """
        Get the current state for a specific agent.
        
        Args:
            agent_idx: Index of the agent
            
        Returns:
            numpy.ndarray: State vector containing agent position and explored map
        """
        # Get agent position and flattened explored map
        pos = (self.agent_center[agent_idx] - self.origin) / UNIT
        return np.concatenate([pos, self.explored_map.flatten()])

    def _update_explored_area(self, agent_idx):
        """
        Update the explored area for a specific agent.
        Marks cells within the agent's detection range as explored
        and updates visualization.
        
        Args:
            agent_idx: Index of the agent
        """
        pos = (self.agent_center[agent_idx] - self.origin) / UNIT
        x, y = int(pos[0]), int(pos[1])
        
        # Update explored area within detection range (Manhattan distance)
        for dx in range(-DETECTION_RANGE, DETECTION_RANGE + 1):
            for dy in range(-DETECTION_RANGE + abs(dx), DETECTION_RANGE - abs(dx) + 1):
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < ENV_W and 0 <= new_y < ENV_H:
                    if self.explored_map[new_y, new_x] == 0:  # If not already explored
                        self.explored_map[new_y, new_x] = 1
                        # Create visual representation of explored area
                        cell = self.canvas.create_rectangle(
                            new_x * UNIT, new_y * UNIT,
                            (new_x + 1) * UNIT, (new_y + 1) * UNIT,
                            fill='lightgreen', width=0)
                        self.canvas.tag_lower(cell)  # Put explored area behind grid lines
                        self.explored_cells.append(cell)

    def _calculate_overlap_penalty(self):
        """
        Calculate penalties for overlapping detection ranges between agents.
        Encourages agents to explore different areas.
        
        Returns:
            numpy.ndarray: Array of overlap penalties for each agent
        """
        overlap_count = np.zeros(self.agentNum)
        
        # For each agent pair, check if their detection ranges overlap
        for i in range(self.agentNum):
            pos_i = (self.agent_center[i] - self.origin) / UNIT
            x_i, y_i = int(pos_i[0]), int(pos_i[1])
            
            # Get detection range cells for agent i
            cells_i = set()
            for dx in range(-DETECTION_RANGE, DETECTION_RANGE + 1):
                for dy in range(-DETECTION_RANGE + abs(dx), DETECTION_RANGE - abs(dx) + 1):
                    new_x, new_y = x_i + dx, y_i + dy
                    if 0 <= new_x < ENV_W and 0 <= new_y < ENV_H:
                        cells_i.add((new_x, new_y))
            
            # Compare with other agents
            for j in range(i + 1, self.agentNum):
                pos_j = (self.agent_center[j] - self.origin) / UNIT
                x_j, y_j = int(pos_j[0]), int(pos_j[1])
                
                # Get detection range cells for agent j
                cells_j = set()
                for dx in range(-DETECTION_RANGE, DETECTION_RANGE + 1):
                    for dy in range(-DETECTION_RANGE + abs(dx), DETECTION_RANGE - abs(dx) + 1):
                        new_x, new_y = x_j + dx, y_j + dy
                        if 0 <= new_x < ENV_W and 0 <= new_y < ENV_H:
                            cells_j.add((new_x, new_y))
                
                # Calculate overlap
                overlap = len(cells_i.intersection(cells_j))
                if overlap > 0:
                    overlap_count[i] += overlap
                    overlap_count[j] += overlap
        
        return overlap_count

    def step(self, actions):
        """
        Execute one time step in the environment.
        
        Args:
            actions: Array of actions for each agent
            
        Returns:
            tuple: (next_state, reward, done, exploration_ratio)
                - next_state: New state after actions
                - reward: Reward for each agent
                - done: Whether episode is finished
                - exploration_ratio: Current exploration progress
        """
        collision_with_wall = np.zeros(self.agentNum)
        move = np.zeros((self.agentNum, 2))
        
        # Calculate moves for all agents
        for i in range(self.agentNum):
            if actions[i] == 0:  # up
                move[i] = [0, -UNIT]
            elif actions[i] == 1:  # down
                move[i] = [0, UNIT]
            elif actions[i] == 2:  # left
                move[i] = [-UNIT, 0]
            else:  # right
                move[i] = [UNIT, 0]
                
            # Check wall collision
            new_pos = self.agent_center[i] + move[i]
            grid_pos = (new_pos - self.origin) / UNIT
            if (grid_pos[0] <= 0 or grid_pos[0] >= ENV_W-1 or 
                grid_pos[1] <= 0 or grid_pos[1] >= ENV_H-1):
                collision_with_wall[i] = 1
                move[i] = [0, 0]  # Don't move if colliding with wall

        # Move agents and update exploration
        for i in range(self.agentNum):
            self.canvas.move(self.agent_all[i], move[i, 0], move[i, 1])
            if not collision_with_wall[i]:
                self.agent_center[i] += move[i]
            self._update_explored_area(i)
            self.canvas.tag_raise(self.agent_all[i])  # Keep agents on top

        # Update step counter
        if not hasattr(self, 'current_step'):
            self.current_step = 0
        self.current_step += 1
        
        # Calculate exploration progress
        current_explored_count = np.sum(self.explored_map)
        exploration_difference = current_explored_count - self.previous_explored_count
        self.previous_explored_count = current_explored_count
        
        # Calculate exploration ratio
        exploration_ratio = current_explored_count / (ENV_H * ENV_W)
        
        # Calculate time-based scaling factor (1x to 3x)
        time_scale = 1.0 + (2.0 * self.current_step / MAX_EP_STEPS)
        
        # Initialize done flag
        done = False
        
        # Calculate base reward
        if exploration_difference > 0:
            # Positive reward for new area, scaled by time
            reward_value = exploration_difference * time_scale
        else:
            # Negative reward (penalty) for no new area
            reward_value = exploration_difference  # No time scaling for penalties
        
        # Calculate overlap penalties (reduced from 0.2 to 0.1)
        overlap_counts = self._calculate_overlap_penalty()
        overlap_penalties = 0.1 * overlap_counts
        
        # Apply wall collision penalty (-1.0 for each collision) and overlap penalties
        reward = reward_value * np.ones(self.agentNum)  # All agents get same base reward
        reward[collision_with_wall == 1] -= 1.0  # Increased penalty for agents that hit walls
        reward -= overlap_penalties  # Apply overlap penalties
        
        # Add constant reward for movement to encourage continuous exploration
        reward += 0.1  # Small constant reward for each step
        
        # Add progressive reward multipliers based on exploration milestones
        if exploration_ratio >= 0.5 and exploration_ratio < 0.75:
            reward *= 1.2  # 1.2x multiplier for reaching 50% exploration
        elif exploration_ratio >= 0.75 and exploration_ratio < 1.0:
            reward *= 1.5  # 1.5x multiplier for reaching 75% exploration
        elif exploration_ratio == 1.0:
            reward *= 2.0  # 2.0x multiplier for achieving full exploration
            done = True
        elif self.current_step >= MAX_EP_STEPS:
            done = True

        # Multiply final reward by exploration ratio when episode ends
        if done:
            reward *= exploration_ratio

        # Get new state
        state = np.zeros((self.agentNum, self.n_features))
        for i in range(self.agentNum):
            state[i] = self._get_state(i)

        return state, reward, done, exploration_ratio

    def render(self):
        """Update the environment visualization."""
        self.update()

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
