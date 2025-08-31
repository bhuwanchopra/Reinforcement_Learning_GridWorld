import numpy as np

class GridWorld:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.fail_pos = None
        self.reward_grid = None
        self.total_reward = 0

    def generate_grid(self):
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.reward_grid = np.full((self.rows, self.cols), -0.1)  # Small negative reward for each step
        
    def add_obstacles(self, positions):
        """Add obstacles to the grid at specified positions."""
        if self.grid is None:
            raise ValueError("Grid not generated. Call generate_grid() first.")
        for pos in positions:
            row, col = pos
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.grid[row, col] = 1  # Mark as obstacle
                self.reward_grid[row, col] = -1.0  # Set negative reward for obstacles
            else:
                raise ValueError(f"Position {pos} is outside the grid boundaries.")

    def set_positions(self, start=None, goal=None, fail=None):
        """Set agent, goal, and fail positions."""
        # Use default positions based on grid size if not provided
        if start is None:
            start = (self.rows // 2, self.cols // 2)
        if goal is None:
            goal = (0, self.cols - 1)
        if fail is None:
            fail = (self.rows - 1, 0)
            
        # Validate positions
        for name, pos in [("start", start), ("goal", goal), ("fail", fail)]:
            r, c = pos
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"{name} position {pos} is outside the grid boundaries.")
        
        self.agent_pos = list(start)
        self.goal_pos = list(goal)
        self.fail_pos = list(fail)
        # Set rewards for specific positions
        self.reward_grid[goal[0], goal[1]] = 1.0  # Positive reward for reaching goal
        self.reward_grid[fail[0], fail[1]] = -1.0  # Negative reward for fail state
        
    def set_custom_reward(self, position, reward):
        """Set a custom reward at a specific position in the grid."""
        if self.reward_grid is None:
            raise ValueError("Reward grid not generated. Call generate_grid() first.")
        row, col = position
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.reward_grid[row, col] = reward
        else:
            raise ValueError(f"Position {position} is outside the grid boundaries.")

    def move(self, action):
        """
        Move agent in the grid.
        Actions: 'up', 'down', 'left', 'right'
        Returns:
            reward: The reward received after the move
        """
        if self.agent_pos is None:
            raise ValueError("Agent position not set. Call set_positions() first.")
        
        old_pos = list(self.agent_pos)  # Save old position
        
        if action == 'up' and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 'down' and self.agent_pos[0] < self.rows - 1:
            self.agent_pos[0] += 1
        elif action == 'left' and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 'right' and self.agent_pos[1] < self.cols - 1:
            self.agent_pos[1] += 1
        # else: invalid move, agent stays in place
        
        # Check if the new position is an obstacle
        row, col = self.agent_pos
        if self.grid[row, col] == 1:  # If obstacle
            self.agent_pos = old_pos  # Move back to previous position
            return -1.0  # Penalty for hitting obstacle
        
        # Get reward for the new position
        reward = self.reward_grid[self.agent_pos[0], self.agent_pos[1]]
        self.total_reward += reward
        
        return reward

    def is_goal_reached(self):
        if self.agent_pos is None or self.goal_pos is None:
            return False
        return tuple(self.agent_pos) == tuple(self.goal_pos)

    def is_fail_reached(self):
        if self.agent_pos is None or self.fail_pos is None:
            return False
        return tuple(self.agent_pos) == tuple(self.fail_pos)

    def get_agent_position(self):
        """Get the current position of the agent as a tuple."""
        if self.agent_pos is None:
            raise ValueError("Agent position not set. Call set_positions() first.")
        return tuple(self.agent_pos)

    def render(self):
        if self.grid is None:
            raise ValueError("Grid not generated. Call generate_grid() first.")
        grid = np.copy(self.grid)
        if self.goal_pos is not None:
            grid[self.goal_pos[0], self.goal_pos[1]] = 7  # Mark goal with 7
        if self.fail_pos is not None:
            grid[self.fail_pos[0], self.fail_pos[1]] = 3  # Mark fail with 3
        if self.agent_pos is not None:
            grid[self.agent_pos[0], self.agent_pos[1]] = 9  # Mark agent with 9
        print(grid)
        
    def show_rewards(self):
        """Display the reward grid."""
        if self.reward_grid is None:
            raise ValueError("Reward grid not generated. Call generate_grid() first.")
        print("Reward Grid:")
        print(np.round(self.reward_grid, 2))
        
    def is_valid_position(self, position):
        """Check if a position is valid (within grid and not an obstacle)."""
        if self.grid is None:
            raise ValueError("Grid not generated. Call generate_grid() first.")
        row, col = position
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col] != 1  # Not an obstacle
        return False
        
    def get_total_reward(self):
        """Get the total accumulated reward."""
        return self.total_reward


# Example usage if running this file directly
if __name__ == "__main__":
    env = GridWorld(5, 5)
    env.generate_grid()
    env.set_positions(start=(2, 2), goal=(0, 4), fail=(4, 0))
    
    # Add some custom rewards to make certain paths more favorable
    env.set_custom_reward((1, 4), 0.5)  # Bonus reward on the path to the goal
    env.set_custom_reward((4, 1), -0.5)  # Extra penalty on the path to failure
    
    env.render()
    env.show_rewards()
    
    actions = ['down', 'down', 'left', 'left', 'left', 'left']
    print("\nStarting the agent's journey:")
    for action in actions:
        reward = env.move(action)
        env.render()
        print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {env.get_total_reward():.2f}")
        if env.is_goal_reached():
            print("Goal reached!")
            break
        if env.is_fail_reached():
            print("Fail state reached!")
            break
