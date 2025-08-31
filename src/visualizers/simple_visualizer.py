import numpy as np
import matplotlib.pyplot as plt
import time
import random
from src.environments.gridworld import GridWorld
from src.environments.stochastic_gridworld import StochasticGridWorld
from src.agents.q_learning_agent import QLearningAgent

class SimpleGridWorldVisualizer:
    """A simple step-by-step visualizer for GridWorld environments."""
    
    def __init__(self, grid_size=5, obstacles=None, noise=0.0):
        """
        Initialize the visualizer.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            obstacles: List of obstacle positions
            noise: Probability of actions not executing as intended (0.0 to 1.0)
        """
        # Create environment
        if noise > 0.0:
            self.env = StochasticGridWorld(grid_size, grid_size, noise=noise)
        else:
            self.env = GridWorld(grid_size, grid_size)
        
        self.env.generate_grid()
        
        # Store obstacles
        self.obstacles = obstacles if obstacles else []
        
        # Add obstacles if specified
        if obstacles:
            self.env.add_obstacles(obstacles)
        
        # Set positions
        self.env.set_positions(
            start=(grid_size//2, grid_size//2), 
            goal=(0, grid_size-1), 
            fail=(grid_size-1, 0)
        )
        
        # Create and train agent
        self.agent = QLearningAgent(self.env, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    def train_agent(self, episodes=500):
        """Train the agent."""
        print("Training agent...")
        self.agent.train(episodes=episodes, max_steps=self.env.rows*4, decay_epsilon=True)
        print("Training complete!")
    
    def visualize_policy(self):
        """Visualize the learned policy."""
        # Get the policy grid from the agent
        policy_grid = np.zeros((self.env.rows, self.env.cols), dtype=object)
        
        # Fill the grid with the best action for each state
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                q_values = self.agent.q_table[row, col, :]
                best_action_idx = np.argmax(q_values)
                
                # Convert action index to arrow symbol
                if self.agent.actions[best_action_idx] == 'up':
                    policy_grid[row, col] = '↑'
                elif self.agent.actions[best_action_idx] == 'down':
                    policy_grid[row, col] = '↓'
                elif self.agent.actions[best_action_idx] == 'left':
                    policy_grid[row, col] = '←'
                elif self.agent.actions[best_action_idx] == 'right':
                    policy_grid[row, col] = '→'
        
        # Mark special cells
        if self.env.goal_pos is not None:
            policy_grid[self.env.goal_pos[0], self.env.goal_pos[1]] = 'G'
        if self.env.fail_pos is not None:
            policy_grid[self.env.fail_pos[0], self.env.fail_pos[1]] = 'F'
        
        # Mark obstacles
        if self.obstacles:
            for r, c in self.obstacles:
                policy_grid[r, c] = '■'
        
        return policy_grid
    
    def visualize_step_by_step(self, max_steps=20, delay=0.5):
        """
        Visualize the agent following the learned policy step by step.
        
        Args:
            max_steps: Maximum number of steps to take
            delay: Delay in seconds between steps
        """
        # Reset environment
        self.env.generate_grid()
        if self.obstacles:
            self.env.add_obstacles(self.obstacles)
        
        start = (self.env.rows//2, self.env.cols//2)
        goal = (0, self.env.cols-1)
        fail = (self.env.rows-1, 0)
        self.env.set_positions(start=start, goal=goal, fail=fail)
        
        # Get the policy grid
        policy = self.visualize_policy()
        
        # Create a grid to show agent's position
        position_grid = np.zeros((self.env.rows, self.env.cols), dtype=object)
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                position_grid[r, c] = ' '
        
        # Get initial state
        state = self.env.get_agent_position()
        total_reward = 0
        
        print("Starting policy visualization...")
        
        # Show initial state
        position_grid[state[0], state[1]] = 'A'
        self._display_grids(policy, position_grid)
        time.sleep(delay)
        
        # Follow policy step by step
        for step in range(max_steps):
            # Choose best action (no exploration)
            row, col = state
            q_values = self.agent.q_table[row, col, :]
            best_action_idx = np.argmax(q_values)
            action = self.agent.actions[best_action_idx]
            
            # Take action
            reward = self.env.move(action)
            next_state = self.env.get_agent_position()
            total_reward += reward
            
            # Check if position actually changed
            if next_state == state:
                print(f"\nStep {step+1}: Action = {action}, Result: Agent hit a wall or obstacle. Trying different action.")
                
                # Try a different action
                q_values_copy = q_values.copy()
                q_values_copy[best_action_idx] = -np.inf  # Exclude the current best action
                
                if np.max(q_values_copy) > -np.inf:  # If there's another valid action
                    next_best_action_idx = np.argmax(q_values_copy)
                    action = self.agent.actions[next_best_action_idx]
                    
                    # Try the new action
                    reward = self.env.move(action)
                    next_state = self.env.get_agent_position()
                    total_reward += reward
                
                # If still stuck, pick a random action
                if next_state == state:
                    action = random.choice(self.agent.actions)
                    reward = self.env.move(action)
                    next_state = self.env.get_agent_position()
                    total_reward += reward
                    print(f"Still stuck, trying random action: {action}")
            
            # Update position grid
            position_grid[state[0], state[1]] = ' '
            position_grid[next_state[0], next_state[1]] = 'A'
            
            # Display current state
            print(f"\nStep {step+1}: Action = {action}, Reward = {reward:.2f}")
            self._display_grids(policy, position_grid)
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if self.env.is_goal_reached():
                print("\nGoal reached!")
                break
            elif self.env.is_fail_reached():
                print("\nFail state reached!")
                break
            
            # Delay between steps
            time.sleep(delay)
        
        print(f"\nVisualization complete: Total steps = {step+1}, Total reward = {total_reward:.2f}")
    
    def _display_grids(self, policy_grid, position_grid):
        """Display policy and position grids side by side."""
        # Print header
        print("\nLearned Policy" + " "*20 + "Agent Position")
        print("-"*25 + " " + "-"*25)
        
        # Print grids side by side
        for r in range(self.env.rows):
            # Policy grid
            policy_row = " ".join(policy_grid[r])
            # Position grid
            position_row = " ".join(position_grid[r])
            
            print(f"{policy_row}          {position_row}")
            
        # Display obstacles for clarity
        obstacles_str = ", ".join([f"({r},{c})" for r, c in self.obstacles]) if self.obstacles else "None"
        print(f"\nObstacles: {obstacles_str}")
        goal_pos = tuple(self.env.goal_pos) if self.env.goal_pos is not None else None
        fail_pos = tuple(self.env.fail_pos) if self.env.fail_pos is not None else None
        print(f"Goal: {goal_pos}, Fail: {fail_pos}")


# Example usage
if __name__ == "__main__":
    # This code will run when simple_visualizer.py is executed directly
    # Create and train the visualizer with obstacles
    visualizer = SimpleGridWorldVisualizer(
        grid_size=5, 
        obstacles=[(1, 1), (1, 2), (3, 2), (3, 3)],
        noise=0.0  # Change to 0.2 for stochastic environment
    )
    
    # Train the agent
    visualizer.train_agent(episodes=500)
    
    # Visualize the policy step by step
    visualizer.visualize_step_by_step(max_steps=20, delay=1.0)
