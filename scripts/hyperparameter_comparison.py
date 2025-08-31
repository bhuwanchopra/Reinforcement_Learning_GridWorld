import numpy as np
import matplotlib.pyplot as plt
from src.environments.gridworld import GridWorld
from src.agents.q_learning_agent import QLearningAgent

def run_comparison(grid_size=5, num_episodes=500, num_runs=3):
    """
    Run comparison experiments with different hyperparameters.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        num_episodes: Number of episodes to train for each experiment
        num_runs: Number of runs for each experiment
    """
    # Hyperparameter combinations to test
    experiments = [
        {"name": "Standard", "alpha": 0.1, "gamma": 0.9, "epsilon": 0.1, "obstacles": []},
        {"name": "High Learning Rate", "alpha": 0.5, "gamma": 0.9, "epsilon": 0.1, "obstacles": []},
        {"name": "Low Discount Factor", "alpha": 0.1, "gamma": 0.5, "epsilon": 0.1, "obstacles": []},
        {"name": "High Exploration", "alpha": 0.1, "gamma": 0.9, "epsilon": 0.3, "obstacles": []},
        {"name": "With Obstacles", "alpha": 0.1, "gamma": 0.9, "epsilon": 0.1, 
         "obstacles": [(1, 1), (1, 2), (3, 2), (3, 3)]},
    ]
    
    # Results storage
    results = {exp["name"]: {"rewards": [], "steps": []} for exp in experiments}
    
    # Run experiments
    for experiment in experiments:
        print(f"\nRunning experiment: {experiment['name']}")
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            
            # Create environment
            env = GridWorld(grid_size, grid_size)
            env.generate_grid()
            
            # Add obstacles if specified
            if experiment["obstacles"]:
                env.add_obstacles(experiment["obstacles"])
            
            # Set start, goal, and fail positions
            env.set_positions(start=(grid_size//2, grid_size//2), 
                             goal=(0, grid_size-1), 
                             fail=(grid_size-1, 0))
            
            # Create and train agent
            agent = QLearningAgent(env, 
                                  alpha=experiment["alpha"], 
                                  gamma=experiment["gamma"], 
                                  epsilon=experiment["epsilon"])
            
            agent.train(episodes=num_episodes, max_steps=grid_size*3, decay_epsilon=True)
            
            # Store results
            results[experiment["name"]]["rewards"].append(agent.episode_rewards)
            results[experiment["name"]]["steps"].append(agent.steps_per_episode)
    
    # Plot comparison results
    plot_comparison_results(results, num_episodes)

def plot_comparison_results(results, num_episodes):
    """Plot comparison results."""
    # Create a new figure with two subplots
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    for name, data in results.items():
        # Calculate the mean rewards across runs
        mean_rewards = np.mean(data["rewards"], axis=0)
        # Smooth the curve (moving average)
        window_size = min(50, num_episodes // 10)
        smoothed_rewards = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
        x_values = range(len(smoothed_rewards))
        plt.plot(x_values, smoothed_rewards, label=name)
    
    plt.title('Average Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot steps
    plt.subplot(2, 1, 2)
    for name, data in results.items():
        # Calculate the mean steps across runs
        mean_steps = np.mean(data["steps"], axis=0)
        # Smooth the curve (moving average)
        window_size = min(50, num_episodes // 10)
        smoothed_steps = np.convolve(mean_steps, np.ones(window_size)/window_size, mode='valid')
        x_values = range(len(smoothed_steps))
        plt.plot(x_values, smoothed_steps, label=name)
    
    plt.title('Average Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png')
    plt.close()

def demonstrate_learned_policy(experiment_name, grid_size=5, num_episodes=500):
    """Demonstrate the policy learned under a specific experiment configuration."""
    print(f"\nDemonstrating policy for experiment: {experiment_name}")
    
    # Set up parameters based on experiment name
    if experiment_name == "Standard":
        alpha, gamma, epsilon = 0.1, 0.9, 0.1
        obstacles = []
    elif experiment_name == "High Learning Rate":
        alpha, gamma, epsilon = 0.5, 0.9, 0.1
        obstacles = []
    elif experiment_name == "Low Discount Factor":
        alpha, gamma, epsilon = 0.1, 0.5, 0.1
        obstacles = []
    elif experiment_name == "High Exploration":
        alpha, gamma, epsilon = 0.1, 0.9, 0.3
        obstacles = []
    elif experiment_name == "With Obstacles":
        alpha, gamma, epsilon = 0.1, 0.9, 0.1
        obstacles = [(1, 1), (1, 2), (3, 2), (3, 3)]
    else:
        print(f"Unknown experiment: {experiment_name}")
        return
    
    # Create environment
    env = GridWorld(grid_size, grid_size)
    env.generate_grid()
    
    # Add obstacles if specified
    if obstacles:
        env.add_obstacles(obstacles)
    
    # Set start, goal, and fail positions
    env.set_positions(start=(grid_size//2, grid_size//2), 
                     goal=(0, grid_size-1), 
                     fail=(grid_size-1, 0))
    
    # Create and train agent
    agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon)
    agent.train(episodes=num_episodes, max_steps=grid_size*3, decay_epsilon=True)
    
    # Show grid with obstacles, goal, and fail state
    print("Environment:")
    env.render()
    
    # Display the learned policy
    policy = agent.visualize_policy()
    print("\nLearned policy (↑=up, ↓=down, ←=left, →=right, G=goal, F=fail):")
    for row in policy:
        print(' '.join(row))
    
    # Demonstrate the learned policy
    print("\nDemonstrating the learned policy:")
    agent.demonstrate_policy(max_steps=grid_size*3)

if __name__ == "__main__":
    # Run hyperparameter comparison
    run_comparison(grid_size=5, num_episodes=500, num_runs=3)
    
    # Demonstrate the policy learned with obstacles
    demonstrate_learned_policy("With Obstacles", grid_size=5, num_episodes=500)
