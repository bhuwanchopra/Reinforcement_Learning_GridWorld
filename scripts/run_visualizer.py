import argparse
from src.visualizers.simple_visualizer import SimpleGridWorldVisualizer

def main():
    parser = argparse.ArgumentParser(description='GridWorld Reinforcement Learning Visualizer')
    parser.add_argument('--grid_size', type=int, default=5, help='Size of the grid (grid_size x grid_size)')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise level for stochastic environment (0.0 to 1.0)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between steps in visualization (seconds)')
    parser.add_argument('--max_steps', type=int, default=20, help='Maximum steps for visualization')
    args = parser.parse_args()
    
    # Create obstacles (you can customize these)
    obstacles = [(1, 1), (1, 2), (3, 2), (3, 3)]
    
    # Create visualizer
    print(f"Creating GridWorld visualizer with grid size {args.grid_size}x{args.grid_size} and noise {args.noise}")
    visualizer = SimpleGridWorldVisualizer(
        grid_size=args.grid_size,
        obstacles=obstacles,
        noise=args.noise
    )
    
    # Train the agent
    print(f"Training agent for {args.episodes} episodes...")
    visualizer.train_agent(episodes=args.episodes)
    
    # Get and display the policy
    policy = visualizer.visualize_policy()
    print("\nLearned policy (↑=up, ↓=down, ←=left, →=right, G=goal, F=fail):")
    for row in policy:
        print(' '.join(row))
    
    # Visualize step by step
    print(f"\nVisualizing agent behavior with {args.delay}s delay between steps...")
    visualizer.visualize_step_by_step(max_steps=args.max_steps, delay=args.delay)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
