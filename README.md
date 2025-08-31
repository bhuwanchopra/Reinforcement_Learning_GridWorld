# Reinforcement Learning GridWorld

This project implements a GridWorld environment for reinforcement learning experiments. It includes multiple environments, learning algorithms, and visualization tools.

## Features

- **GridWorld Environment**: A customizable grid environment with configurable size, obstacles, and rewards
- **Reinforcement Learning Algorithms**: Implementation of Q-learning and SARSA algorithms
- **Stochastic Environment**: Support for stochastic action outcomes (adding noise to the environment)
- **Visualization Tools**: Multiple visualization options from simple text-based to graphical displays
- **Parameter Testing**: Scripts for comparing different hyperparameters and grid sizes

## Project Structure

```
Reinforcement_Learning_GridWorld/
├── requirements.txt        # Project dependencies
├── src/                    # Source code
│   ├── environments/       # Environment implementations
│   │   ├── __init__.py
│   │   ├── gridworld.py             # Base GridWorld environment
│   │   └── stochastic_gridworld.py  # GridWorld with random action outcomes
│   ├── agents/             # Learning agents
│   │   ├── __init__.py
│   │   ├── q_learning_agent.py      # Q-learning implementation
│   │   └── sarsa_agent.py           # SARSA implementation
│   └── visualizers/        # Visualization tools
│       ├── __init__.py
│       ├── simple_visualizer.py     # Text-based visualization
│       └── advanced_visualizer.py   # Graphical visualization with path tracking
└── scripts/                # Runner scripts and experiments
    ├── run_visualizer.py            # Run simple visualization
    ├── run_advanced_visualizer.py   # Run advanced visualization
    ├── grid_size_test.py            # Compare different grid sizes
    └── hyperparameter_comparison.py # Compare different hyperparameters
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Reinforcement_Learning_GridWorld.git
cd Reinforcement_Learning_GridWorld
```

2. Create a virtual environment (optional but recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

To run a simple visualization of the learned policy:

```bash
python scripts/run_visualizer.py
```

For a more advanced visualization with cycle detection and graphical output:

```bash
python scripts/run_advanced_visualizer.py
```

To experiment with a stochastic environment (where actions don't always result in the intended movement):

```bash
python scripts/run_advanced_visualizer.py --noise 0.2
```

## Command-line Arguments

All visualizer scripts support the following arguments:

- `--grid_size`: Size of the grid (default: 5)
- `--episodes`: Number of training episodes (default: 500)
- `--noise`: Probability of action not executing as intended (0.0 to 1.0, default: 0.0)
- `--delay`: Delay between steps in visualization (seconds, default: 0.5)
- `--max_steps`: Maximum steps for visualization (default: 20)

## Components

### Environments

- **GridWorld**: The core GridWorld environment with customizable grid size, obstacles, and rewards
- **StochasticGridWorld**: Extension of GridWorld that adds randomness to action outcomes

### Agents

- **QLearningAgent**: Implementation of the Q-learning algorithm
- **SARSAAgent**: Implementation of the SARSA (State-Action-Reward-State-Action) algorithm

### Visualizers

- **SimpleGridWorldVisualizer**: Text-based visualization of learned policies
- **GridWorldVisualizer**: Advanced graphical visualization with path tracking and cycle detection

## Examples

### Training a Q-learning Agent

```python
from src.environments.gridworld import GridWorld
from src.agents.q_learning_agent import QLearningAgent

# Create environment
env = GridWorld(5, 5)
env.generate_grid()
env.set_positions(start=(2, 2), goal=(0, 4), fail=(4, 0))

# Add obstacles
env.add_obstacles([(1, 1), (1, 2), (3, 2), (3, 3)])

# Create and train agent
agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
agent.train(episodes=500, max_steps=50, decay_epsilon=True)

# Display the learned policy
policy = agent.visualize_policy()
for row in policy:
    print(' '.join(row))
```

### Using the Stochastic Environment

```python
from src.environments.stochastic_gridworld import StochasticGridWorld
from src.agents.q_learning_agent import QLearningAgent

# Create stochastic environment with 20% noise
env = StochasticGridWorld(5, 5, noise=0.2)
env.generate_grid()
env.set_positions()

# Create and train agent
agent = QLearningAgent(env)
agent.train(episodes=500)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.