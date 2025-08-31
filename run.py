#!/usr/bin/env python3
"""
Launcher script for the GridWorld reinforcement learning project.
This script adds the current directory to the Python path and launches the specified script.
"""

import os
import sys
import subprocess

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def print_usage():
    """Print usage information."""
    print("GridWorld Reinforcement Learning Launcher")
    print("Usage:")
    print("  ./run.py <script_name> [args...]")
    print("")
    print("Available scripts:")
    print("  simple        - Run simple text-based visualization")
    print("  advanced      - Run advanced graphical visualization")
    print("  animated      - Run animation-based visualization (creates GIF)")
    print("  grid_test     - Run grid size comparison test")
    print("  hyperparams   - Run hyperparameter comparison test")
    print("")
    print("Example:")
    print("  ./run.py advanced --grid_size 7 --noise 0.2")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    script_name = sys.argv[1]
    script_args = sys.argv[2:]
    
    # Map script name to actual script file
    script_map = {
        "simple": "scripts/run_visualizer.py",
        "advanced": "scripts/run_advanced_visualizer.py",
        "animated": "scripts/run_animated_visualizer.py",
        "grid_test": "scripts/grid_size_test.py",
        "hyperparams": "scripts/hyperparameter_comparison.py",
    }
    
    if script_name not in script_map:
        print(f"Unknown script: {script_name}")
        print_usage()
        sys.exit(1)
    
    script_path = os.path.join(current_dir, script_map[script_name])
    
    # Run the script with the remaining arguments
    cmd = [sys.executable, script_path] + script_args
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
