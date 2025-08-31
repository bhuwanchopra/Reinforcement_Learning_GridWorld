import random
import numpy as np
import matplotlib.pyplot as plt
from src.environments.gridworld import GridWorld

class StochasticGridWorld(GridWorld):
    """
    A stochastic version of GridWorld where actions may not always result
    in the intended movement.
    """
    
    def __init__(self, rows, cols, noise=0.2):
        """
        Initialize StochasticGridWorld.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            noise: Probability of action not being executed as intended (0 to 1)
        """
        super().__init__(rows, cols)
        self.noise = noise
    
    def move(self, action):
        """
        Move agent in the grid with probability of random outcomes.
        
        Args:
            action: The intended action ('up', 'down', 'left', 'right')
            
        Returns:
            reward: The reward received after the move
        """
        if self.agent_pos is None:
            raise ValueError("Agent position not set. Call set_positions() first.")
        
        # With probability (1-noise), perform the intended action
        # With probability noise, perform a random action
        if random.random() < self.noise:
            # Choose a random action (could be the same as intended)
            actual_action = random.choice(['up', 'down', 'left', 'right'])
            print(f"Oops! Action slipped: {action} -> {actual_action}")
        else:
            actual_action = action
        
        old_pos = list(self.agent_pos)  # Save old position
        
        # Execute the actual action
        if actual_action == 'up' and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif actual_action == 'down' and self.agent_pos[0] < self.rows - 1:
            self.agent_pos[0] += 1
        elif actual_action == 'left' and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif actual_action == 'right' and self.agent_pos[1] < self.cols - 1:
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


# Example usage
if __name__ == "__main__":
    # This code will run when stochastic_gridworld.py is executed directly
    from src.agents.q_learning_agent import QLearningAgent
    
    # Parameters
    grid_size = 5
    num_episodes = 100
    noise_levels = [0.0, 0.2]
    
    # Results storage
    results = {f"Noise {noise:.1f}": {"rewards": [], "steps": []} for noise in noise_levels}
    
    for noise in noise_levels:
        print(f"\nTesting with noise level = {noise}")
        
        # Create environment (deterministic or stochastic)
        if noise == 0.0:
            env = GridWorld(grid_size, grid_size)
        else:
            env = StochasticGridWorld(grid_size, grid_size, noise=noise)
        
        env.generate_grid()
        
        # Add some obstacles
        env.add_obstacles([(1, 1), (1, 2), (3, 2), (3, 3)])
        
        # Set positions
        env.set_positions(start=(2, 2), goal=(0, 4), fail=(4, 0))
        
        # Create and train agent
        agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
        agent.train(episodes=num_episodes, max_steps=50, decay_epsilon=True)
        
        # Display the learned policy
        policy = agent.visualize_policy()
        print("\nLearned policy with noise =", noise)
        for row in policy:
            print(' '.join(row))
        
        # Demonstrate the learned policy
        if noise > 0:
            print("\nDemonstrating policy with stochastic actions:")
        else:
            print("\nDemonstrating policy with deterministic actions:")
        agent.demonstrate_policy(max_steps=10)
