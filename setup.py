from setuptools import setup, find_packages

setup(
    name="gridworld_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
    ],
    description="Reinforcement Learning in GridWorld environments",
    author="Bhuwan Chopra",
    author_email="your.email@example.com",
)
