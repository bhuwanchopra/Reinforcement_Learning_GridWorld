import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import time
from src.environments.gridworld import GridWorld
from src.environments.stochastic_gridworld import StochasticGridWorld
from src.agents.q_learning_agent import QLearningAgent

class AnimatedGridWorldVisualizer:
    """Visualizer for GridWorld environments with animation capabilities."""
    
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
        self.env.set_positions(start=(grid_size//2, grid_size//2), 
                              goal=(0, grid_size-1), 
                              fail=(grid_size-1, 0))
        
        # Create and train agent
        self.agent = QLearningAgent(self.env, alpha=0.1, gamma=0.9, epsilon=0.1)
        
        # Setup visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.cmap = ListedColormap(['white', 'gray', 'red', 'green', 'blue'])
        self.img = None
        self.text = self.ax.text(0.02, 0.02, "", transform=self.ax.transAxes)
        
        # For animation
        self.frames = []
        self.pause_time = 0.5  # seconds between steps
    
    def train_agent(self, episodes=500):
        """Train the agent."""
        print("Training agent...")
        self.agent.train(episodes=episodes, max_steps=self.env.rows*4, decay_epsilon=True)
        print("Training complete!")
    
    def visualize_grid(self):
        """Visualize the current state of the grid."""
        # Create a visualization grid
        grid = np.copy(self.env.grid)
        
        # Mark special cells
        if self.env.goal_pos is not None:
            grid[self.env.goal_pos[0], self.env.goal_pos[1]] = 3  # Goal (green)
        if self.env.fail_pos is not None:
            grid[self.env.fail_pos[0], self.env.fail_pos[1]] = 2  # Fail (red)
        if self.env.agent_pos is not None:
            grid[self.env.agent_pos[0], self.env.agent_pos[1]] = 4  # Agent (blue)
        
        # Update or create the image
        if self.img is None:
            self.img = self.ax.imshow(grid, cmap=self.cmap, vmin=0, vmax=4)
            self.ax.grid(True, which='both', color='black', linewidth=1)
            self.ax.set_xticks(np.arange(-.5, self.env.cols, 1), minor=True)
            self.ax.set_yticks(np.arange(-.5, self.env.rows, 1), minor=True)
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
        else:
            self.img.set_data(grid)
        
        # Add policy arrows
        # Clear existing texts except the status text
        for txt in self.ax.texts:
            if txt != self.text:
                txt.remove()
        
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                # Skip agent, goal, fail positions and obstacles
                if (r, c) == tuple(self.env.agent_pos) or \
                   (r, c) == tuple(self.env.goal_pos) or \
                   (r, c) == tuple(self.env.fail_pos) or \
                   self.env.grid[r, c] == 1:
                    continue
                
                # Get best action for this state
                action_idx = np.argmax(self.agent.q_table[r, c, :])
                action = self.agent.actions[action_idx]
                
                # Convert action to arrow
                if action == 'up':
                    arrow = '↑'
                elif action == 'down':
                    arrow = '↓'
                elif action == 'left':
                    arrow = '←'
                elif action == 'right':
                    arrow = '→'
                
                # Add text arrow
                self.ax.text(c, r, arrow, ha='center', va='center', fontsize=14)
        
        # Update status text
        reward = self.env.reward_grid[self.env.agent_pos[0], self.env.agent_pos[1]]
        status = f"Position: {tuple(self.env.agent_pos)}, Reward: {reward:.2f}"
        self.text.set_text(status)
        
        return self.img,
    
    def animate_policy(self, max_steps=20):
        """Animate the agent following the learned policy."""
        # Reset environment
        self.env.generate_grid()
        if hasattr(self, 'obstacles') and self.obstacles:
            self.env.add_obstacles(self.obstacles)
        self.env.set_positions(start=(self.env.rows//2, self.env.cols//2), 
                              goal=(0, self.env.cols-1), 
                              fail=(self.env.rows-1, 0))
        
        # Create initial visualization
        self.visualize_grid()
        
        # Create animation with delay between frames
        def update(_):
            if len(self.frames) > 0:
                frame = self.frames.pop(0)
                self.img.set_data(frame)
            return self.img,
        
        # Get initial state
        state = self.env.get_agent_position()
        total_reward = 0
        
        print("Starting policy visualization...")
        
        # Generate frames
        self.frames = []
        for step in range(max_steps):
            # Choose best action (no exploration)
            row, col = state
            action_idx = np.argmax(self.agent.q_table[row, col, :])
            action = self.agent.actions[action_idx]
            
            # Take action
            reward = self.env.move(action)
            state = self.env.get_agent_position()
            total_reward += reward
            
            # Create a frame for the current state
            grid = np.copy(self.env.grid)
            if self.env.goal_pos is not None:
                grid[self.env.goal_pos[0], self.env.goal_pos[1]] = 3  # Goal (green)
            if self.env.fail_pos is not None:
                grid[self.env.fail_pos[0], self.env.fail_pos[1]] = 2  # Fail (red)
            grid[self.env.agent_pos[0], self.env.agent_pos[1]] = 4  # Agent (blue)
            
            # Add frame multiple times to slow down animation
            for _ in range(5):  # Adjust for desired speed
                self.frames.append(grid)
            
            # Check if episode is done
            if self.env.is_goal_reached() or self.env.is_fail_reached():
                if self.env.is_goal_reached():
                    print("Goal reached!")
                else:
                    print("Fail state reached!")
                break
        
        print(f"Animation complete: Total steps = {step+1}, Total reward = {total_reward:.2f}")
        
        # Create animation
        ani = animation.FuncAnimation(self.fig, update, frames=len(self.frames), 
                                     interval=200, blit=True)
        
        # Create data directory if it doesn't exist
        import os
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save animation to data folder
        output_path = os.path.join(data_dir, 'gridworld_visualization.gif')
        ani.save(output_path, writer='pillow', fps=5)
        print(f"Animation saved as '{output_path}'")
        
        # Display the animation in the notebook
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create and train the visualizer with obstacles
    visualizer = AnimatedGridWorldVisualizer(
        grid_size=5, 
        obstacles=[(1, 1), (1, 2), (3, 2), (3, 3)],
        noise=0.0  # Change to 0.2 for stochastic environment
    )
    
    # Train the agent
    visualizer.train_agent(episodes=500)
    
    # Visualize the policy
    visualizer.animate_policy(max_steps=20)
