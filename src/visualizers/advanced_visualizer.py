import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import time
import random
from src.environments.gridworld import GridWorld
from src.environments.stochastic_gridworld import StochasticGridWorld
from src.agents.q_learning_agent import QLearningAgent

class GridWorldVisualizer:
    """A visualizer for GridWorld environments with colored grid cells."""
    
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
        
        # For tracking the agent's path
        self.path = []
        self.visited_states = set()
    
    def train_agent(self, episodes=500):
        """Train the agent."""
        print("Training agent...")
        self.agent.train(episodes=episodes, max_steps=self.env.rows*4, decay_epsilon=True)
        print("Training complete!")
    
    def show_policy_grid(self, save_path='policy_visualization.png'):
        """
        Show the policy grid with colored cells.
        
        Args:
            save_path: Path to save the visualization image
        """
        # Get policy
        policy_grid = self._get_policy_grid()
        
        # Setup the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create color grid for visualization
        grid_values = np.zeros((self.env.rows, self.env.cols))
        
        # Mark special cells
        goal_pos = tuple(self.env.goal_pos) if self.env.goal_pos is not None else None
        fail_pos = tuple(self.env.fail_pos) if self.env.fail_pos is not None else None
        
        # Create colored grid
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                pos = (r, c)
                if pos == goal_pos:
                    grid_values[r, c] = 1  # Goal state
                elif pos == fail_pos:
                    grid_values[r, c] = 2  # Fail state
                elif pos in self.obstacles:
                    grid_values[r, c] = 3  # Obstacle
                elif pos in self.visited_states:
                    grid_values[r, c] = 4  # Visited state
                else:
                    grid_values[r, c] = 0  # Regular cell
        
        # Create colormap
        cmap = mcolors.ListedColormap(['white', 'green', 'red', 'gray', 'lightblue'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Draw grid
        ax.imshow(grid_values, cmap=cmap, norm=norm)
        
        # Add policy arrows to each cell
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                cell_value = grid_values[r, c]
                text = policy_grid[r, c]
                
                # Choose text color based on cell type
                if cell_value == 1:  # Goal
                    text_color = 'white'
                    text = 'G'
                elif cell_value == 2:  # Fail
                    text_color = 'white'
                    text = 'F'
                elif cell_value == 3:  # Obstacle
                    text_color = 'white'
                    text = 'X'
                else:
                    text_color = 'black'
                
                # Add text to grid cell
                ax.text(c, r, text, ha='center', va='center', fontsize=14, color=text_color)
        
        # Add grid lines
        ax.grid(True, color='black', linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(-.5, self.env.cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.env.rows, 1), minor=True)
        ax.set_xticks(np.arange(0, self.env.cols, 1))
        ax.set_yticks(np.arange(0, self.env.rows, 1))
        ax.tick_params(which="minor", bottom=False, left=False)
        
        # Remove axis ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='white', label='Regular Cell'),
            mpatches.Patch(color='green', label='Goal'),
            mpatches.Patch(color='red', label='Fail State'),
            mpatches.Patch(color='gray', label='Obstacle'),
            mpatches.Patch(color='lightblue', label='Visited')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Add title
        plt.title('Learned Policy (→ ← ↑ ↓)')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path)
        plt.close()
        
        print(f"Policy visualization saved to {save_path}")
    
    def _get_policy_grid(self):
        """Get the policy grid with arrows."""
        # Create a grid to visualize the policy
        policy_grid = np.zeros((self.env.rows, self.env.cols), dtype=object)
        
        # Fill the grid with the best action for each state
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                if (row, col) in self.obstacles:
                    policy_grid[row, col] = 'X'
                    continue
                    
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
        
        return policy_grid
    
    def visualize_policy(self, max_steps=20, delay=0.5, save_path='policy_visualization.png'):
        """
        Visualize the agent following the learned policy step by step.
        Includes better handling of obstacles and avoiding cycles.
        
        Args:
            max_steps: Maximum number of steps to take
            delay: Delay in seconds between steps
            save_path: Path to save the visualization image
        
        Returns:
            total_reward: Total reward accumulated
            steps: Number of steps taken
        """
        # Reset environment and tracking variables
        self.env.generate_grid()
        if self.obstacles:
            self.env.add_obstacles(self.obstacles)
        
        self.env.set_positions(
            start=(self.env.rows//2, self.env.cols//2), 
            goal=(0, self.env.cols-1), 
            fail=(self.env.rows-1, 0)
        )
        
        # Reset path tracking
        self.path = []
        self.visited_states = set()
        
        # Get initial state
        state = self.env.get_agent_position()
        self.path.append(state)
        self.visited_states.add(state)
        
        total_reward = 0
        
        print("Starting policy visualization...")
        print(f"Agent starting at position: {state}")
        
        goal_pos_str = tuple(self.env.goal_pos) if self.env.goal_pos is not None else None
        fail_pos_str = tuple(self.env.fail_pos) if self.env.fail_pos is not None else None
        print(f"Goal at: {goal_pos_str}, Fail at: {fail_pos_str}")
        print(f"Obstacles: {self.obstacles}")
        
        # Map from action to movement direction
        action_to_dir = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # For tracking consecutive repeated states
        last_positions = []
        
        # Follow policy step by step
        for step in range(max_steps):
            # Get current state and Q-values
            row, col = state
            q_values = self.agent.q_table[row, col, :].copy()
            
            # Choose an action, with special handling for cycles
            if len(last_positions) >= 3 and len(set(last_positions[-3:])) <= 1:
                # Detected cycling behavior - force exploration
                for action_idx, action in enumerate(self.agent.actions):
                    dr, dc = action_to_dir[action]
                    next_r, next_c = row + dr, col + dc
                    
                    # Check if this action would lead to a new state
                    if (0 <= next_r < self.env.rows and 
                        0 <= next_c < self.env.cols and 
                        (next_r, next_c) not in self.visited_states and
                        (next_r, next_c) not in self.obstacles):
                        
                        # Prioritize this unexplored action
                        action_idx = action_idx
                        action = self.agent.actions[action_idx]
                        print(f"Breaking cycle by exploring new state with action: {action}")
                        break
                else:
                    # If no unexplored states, choose randomly but avoid obstacles
                    valid_actions = []
                    for action_idx, action in enumerate(self.agent.actions):
                        dr, dc = action_to_dir[action]
                        next_r, next_c = row + dr, col + dc
                        
                        if (0 <= next_r < self.env.rows and 
                            0 <= next_c < self.env.cols and 
                            (next_r, next_c) not in self.obstacles):
                            valid_actions.append(action)
                    
                    action = random.choice(valid_actions) if valid_actions else self.agent.actions[np.argmax(q_values)]
                    print(f"Breaking cycle with random valid action: {action}")
            else:
                # Normal policy following
                action_idx = np.argmax(q_values)
                action = self.agent.actions[action_idx]
            
            # Take action
            old_state = state
            reward = self.env.move(action)
            state = self.env.get_agent_position()
            
            # Update tracking variables
            total_reward += reward
            self.path.append(state)
            self.visited_states.add(state)
            
            # Check if we actually moved
            if state == old_state:
                print(f"Step {step+1}: Action {action} failed (hit wall/obstacle)")
            else:
                print(f"Step {step+1}: Action {action}, Reward: {reward:.2f}, New Position: {state}")
            
            # Update last positions for cycle detection
            last_positions.append(state)
            if len(last_positions) > 5:
                last_positions.pop(0)
            
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
        
        # Show the path on the grid
        self.show_policy_grid(save_path=save_path)
        
        return total_reward, step+1


# Example usage
if __name__ == "__main__":
    # This code will run when advanced_visualizer.py is executed directly
    # Create visualizer with obstacles
    visualizer = GridWorldVisualizer(
        grid_size=5, 
        obstacles=[(1, 1), (1, 2), (3, 2), (3, 3)],
        noise=0.0  # Change to 0.2 for stochastic environment
    )
    
    # Train the agent
    visualizer.train_agent(episodes=500)
    
    # Visualize the policy step by step
    visualizer.visualize_policy(max_steps=20, delay=0.5)
