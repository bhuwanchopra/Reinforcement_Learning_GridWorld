import numpy as np
import matplotlib.pyplot as plt
from src.environments.gridworld import GridWorld
from src.agents.q_learning_agent import QLearningAgent

def test_different_grid_sizes():
    """Test the performance of Q-learning with different grid sizes."""
    grid_sizes = [3, 5, 7, 10]
    num_episodes = 500
    num_runs = 3
    
    # Results storage
    results = {f"{size}x{size} Grid": {"rewards": [], "steps": [], "convergence_episode": []} 
               for size in grid_sizes}
    
    for size in grid_sizes:
        print(f"\nTesting {size}x{size} Grid")
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            
            # Create environment
            env = GridWorld(size, size)
            env.generate_grid()
            
            # Set start, goal, and fail positions proportionally to grid size
            env.set_positions(
                start=(size//2, size//2),
                goal=(0, size-1),
                fail=(size-1, 0)
            )
            
            # Create and train agent
            agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
            agent.train(episodes=num_episodes, max_steps=size*4, decay_epsilon=True)
            
            # Store results
            results[f"{size}x{size} Grid"]["rewards"].append(agent.episode_rewards)
            results[f"{size}x{size} Grid"]["steps"].append(agent.steps_per_episode)
            
            # Determine convergence episode (when agent consistently gets positive rewards)
            convergence_episode = 0
            window_size = 10
            for i in range(len(agent.episode_rewards) - window_size):
                window = agent.episode_rewards[i:i+window_size]
                if np.mean(window) > 0:
                    convergence_episode = i
                    break
            
            results[f"{size}x{size} Grid"]["convergence_episode"].append(convergence_episode)
            
    # Plot comparison results
    plot_grid_size_comparison(results, num_episodes)
    
    # Print convergence statistics
    print("\nConvergence Statistics (episodes needed to consistently get positive rewards):")
    for size, data in results.items():
        avg_convergence = np.mean(data["convergence_episode"])
        std_convergence = np.std(data["convergence_episode"])
        print(f"{size}: {avg_convergence:.1f} ± {std_convergence:.1f} episodes")

def plot_grid_size_comparison(results, num_episodes):
    """Plot grid size comparison results."""
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
    
    plt.title('Average Rewards per Episode for Different Grid Sizes')
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
    
    plt.title('Average Steps per Episode for Different Grid Sizes')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('grid_size_comparison.png')
    plt.close()
    
    # Create a bar chart for convergence episodes
    plt.figure(figsize=(10, 6))
    grid_sizes = list(results.keys())
    avg_convergence = [np.mean(data["convergence_episode"]) for data in results.values()]
    std_convergence = [np.std(data["convergence_episode"]) for data in results.values()]
    
    plt.bar(grid_sizes, avg_convergence, yerr=std_convergence, capsize=5)
    plt.title('Episodes to Convergence for Different Grid Sizes')
    plt.xlabel('Grid Size')
    plt.ylabel('Episodes to Convergence')
    plt.grid(True, axis='y')
    plt.savefig('grid_size_convergence.png')
    plt.close()

if __name__ == "__main__":
    test_different_grid_sizes()
