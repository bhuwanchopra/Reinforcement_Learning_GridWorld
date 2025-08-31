import numpy as np
import random
import matplotlib.pyplot as plt
from src.environments.gridworld import GridWorld

class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize SARSA agent with hyperparameters.
        
        Args:
            env: GridWorld environment
            alpha: Learning rate (0 to 1)
            gamma: Discount factor (0 to 1)
            epsilon: Exploration rate (0 to 1)
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.rows, env.cols, 4))  # 4 actions: up, down, left, right
        self.actions = ['up', 'down', 'left', 'right']
        self.action_idx = {a: i for i, a in enumerate(self.actions)}
        
        # For tracking learning progress
        self.episode_rewards = []
        self.steps_per_episode = []
        
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            # Exploration: choose a random action
            return random.choice(self.actions)
        else:
            # Exploitation: choose the best action based on Q-values
            row, col = state
            q_values = self.q_table[row, col, :]
            # If multiple actions have the same max value, choose randomly among them
            max_q = np.max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_table(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-table using the SARSA update rule.
        """
        row, col = state
        next_row, next_col = next_state
        action_idx = self.action_idx[action]
        
        # Current Q-value
        current_q = self.q_table[row, col, action_idx]
        
        # Next Q-value (SARSA uses the actual next action, not the max)
        if done:
            next_q = 0
        else:
            next_action_idx = self.action_idx[next_action]
            next_q = self.q_table[next_row, next_col, next_action_idx]
        
        # SARSA update formula: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[row, col, action_idx] = new_q
    
    def train(self, episodes=1000, max_steps=100, decay_epsilon=True):
        """
        Train the agent using SARSA.
        
        Args:
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            decay_epsilon: Whether to decay epsilon over time
        """
        initial_epsilon = self.epsilon
        
        for episode in range(episodes):
            # Reset environment
            self.env.generate_grid()
            
            # Set positions based on environment defaults
            start_pos = (self.env.rows//2, self.env.cols//2)
            goal_pos = (0, self.env.cols-1)
            fail_pos = (self.env.rows-1, 0)
            self.env.set_positions(start=start_pos, goal=goal_pos, fail=fail_pos)
            
            # Decay epsilon over time if enabled
            if decay_epsilon:
                # Use an exponential decay for smoother exploration reduction
                self.epsilon = initial_epsilon * np.exp(-3.0 * episode / episodes)
            
            # Get initial state and choose action
            state = self.env.get_agent_position()
            action = self.choose_action(state)
            
            total_reward = 0
            done = False
            steps = 0
            
            for step in range(max_steps):
                # Take action and observe reward and next state
                reward = self.env.move(action)
                next_state = self.env.get_agent_position()
                
                # Check if episode is done
                if self.env.is_goal_reached() or self.env.is_fail_reached():
                    done = True
                    next_action = None  # No next action if done
                else:
                    # Choose next action based on next state
                    next_action = self.choose_action(next_state)
                
                # Update Q-table
                if not done:
                    self.update_q_table(state, action, reward, next_state, next_action, done)
                else:
                    # Use a simpler update for terminal states
                    row, col = state
                    action_idx = self.action_idx[action]
                    current_q = self.q_table[row, col, action_idx]
                    new_q = current_q + self.alpha * (reward - current_q)
                    self.q_table[row, col, action_idx] = new_q
                
                # Update state and action
                state = next_state
                action = next_action  # This is the key difference from Q-learning
                
                # Update total reward and steps
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Store episode stats
            self.episode_rewards.append(total_reward)
            self.steps_per_episode.append(steps)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.steps_per_episode[-100:])
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {self.epsilon:.2f}")
    
    def visualize_learning(self, save_path='data/sarsa_learning_progress.png'):
        """
        Visualize the learning progress.
        
        Args:
            save_path: Path to save the learning progress plot
        """
        # Create data directory if it doesn't exist
        import os
        data_dir = os.path.dirname(save_path)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        plt.figure(figsize=(12, 5))
        
        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Rewards per Episode (SARSA)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot steps
        plt.subplot(1, 2, 2)
        plt.plot(self.steps_per_episode)
        plt.title('Steps per Episode (SARSA)')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Learning progress visualization saved to {save_path}")
    
    def visualize_policy(self):
        """
        Visualize the learned policy.
        
        Returns:
            policy_grid: A grid of arrows representing the best action in each state
        """
        # Create a grid to visualize the policy
        policy_grid = np.zeros((self.env.rows, self.env.cols), dtype=object)
        
        # Fill the grid with the best action for each state
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                q_values = self.q_table[row, col, :]
                best_action_idx = np.argmax(q_values)
                
                # Convert action index to arrow symbol
                if self.actions[best_action_idx] == 'up':
                    policy_grid[row, col] = '↑'
                elif self.actions[best_action_idx] == 'down':
                    policy_grid[row, col] = '↓'
                elif self.actions[best_action_idx] == 'left':
                    policy_grid[row, col] = '←'
                elif self.actions[best_action_idx] == 'right':
                    policy_grid[row, col] = '→'
        
        # Mark goal and fail states
        if self.env.goal_pos is not None:
            policy_grid[self.env.goal_pos[0], self.env.goal_pos[1]] = 'G'
        if self.env.fail_pos is not None:
            policy_grid[self.env.fail_pos[0], self.env.fail_pos[1]] = 'F'
        
        return policy_grid
    
    def demonstrate_policy(self, max_steps=20):
        """
        Demonstrate the learned policy.
        
        Args:
            max_steps: Maximum number of steps to take during demonstration
        """
        # Reset environment
        self.env.generate_grid()
        start_pos = (self.env.rows//2, self.env.cols//2)
        goal_pos = (0, self.env.cols-1)
        fail_pos = (self.env.rows-1, 0)
        self.env.set_positions(start=start_pos, goal=goal_pos, fail=fail_pos)
        
        # Get initial state
        state = self.env.get_agent_position()
        total_reward = 0
        steps = 0
        
        print("Starting SARSA policy demonstration...")
        self.env.render()
        
        for step in range(max_steps):
            # Choose best action (no exploration)
            row, col = state
            action_idx = np.argmax(self.q_table[row, col, :])
            action = self.actions[action_idx]
            
            # Take action
            reward = self.env.move(action)
            state = self.env.get_agent_position()
            total_reward += reward
            steps += 1
            
            # Print step details
            print(f"Step {step+1}: Action = {action}, Reward = {reward:.2f}")
            self.env.render()
            
            # Check if episode is done
            if self.env.is_goal_reached():
                print("Goal reached!")
                break
            elif self.env.is_fail_reached():
                print("Fail state reached!")
                break
        
        print(f"Demonstration complete: Total steps = {steps}, Total reward = {total_reward:.2f}")


# Example usage
if __name__ == "__main__":
    # This code will run when sarsa_agent.py is executed directly
    from src.environments.gridworld import GridWorld
    
    # Create environment
    env = GridWorld(5, 5)
    env.generate_grid()
    env.set_positions(start=(2, 2), goal=(0, 4), fail=(4, 0))
    
    # Add obstacles
    env.add_obstacles([(1, 1), (1, 2), (3, 2), (3, 3)])
    
    # Display the grid and rewards
    print("Initial grid:")
    env.render()
    env.show_rewards()
    
    # Create and train SARSA agent
    agent = SARSAAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    agent.train(episodes=500, max_steps=50, decay_epsilon=True)
    
    # Visualize learning progress
    agent.visualize_learning()
    
    # Display the learned policy
    policy = agent.visualize_policy()
    print("\nLearned policy (↑=up, ↓=down, ←=left, →=right, G=goal, F=fail):")
    for row in policy:
        print(' '.join(row))
    
    # Demonstrate the learned policy
    print("\nDemonstrating the learned policy:")
    agent.demonstrate_policy()
