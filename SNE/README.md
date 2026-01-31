# Smart Neural Explorer (SNE)

A PyTorch-based implementation of meta-cognitive exploration with dynamic attention for reinforcement learning agents.

## Overview

Smart Neural Explorer (SNE) is an advanced exploration architecture that incorporates meta-cognitive abilities, dynamic attention mechanisms, and creative intention generation to enable more efficient and effective exploration in reinforcement learning environments.

### Key Features

- **Meta-cognitive Monitoring**: Tracks learning progress and adjusts exploration strategy accordingly
- **Dynamic Attention Allocation**: Focuses on relevant state dimensions based on uncertainty and novelty
- **Creative Intention Generation**: Produces diverse exploration strategies including modification, analogy, combination, and counterfactual reasoning
- **Curiosity-Driven Exploration**: Uses prediction error to drive exploration in novel regions
- **Cognitive State Management**: Adapts behavior based on current cognitive state (focused, divergent, confused, insight)

## Architecture

The SNE architecture is modularly designed with the following components:

### 1. Core Components (`sne_basics.py`)
- **Enumerations**: Attention granularity, intention types, cognitive states, exploration phases
- **Data Structures**: Experience replay entries, attention focus, fuzzy intentions

### 2. Neural Network Modules (`sne_neural.py`)
- **StateEncoder**: Encodes states into meaningful representations
- **UncertaintyEstimator**: Estimates uncertainty in value predictions
- **CuriosityModule**: Computes intrinsic motivation based on prediction error

### 3. Cognitive Architecture (`sne_core.py`)
- **WorkingMemory**: Simulates human-like working memory with limited capacity
- **DynamicAttentionManager**: Allocates attention based on multiple factors
- **CreativeIntentionGenerator**: Produces diverse exploration strategies
- **MetaCognitiveMonitor**: Tracks learning progress and cognitive state

### 4. Agent Implementation (`sne_agent.py`)
- **SmartNeuralExplorer**: Main agent class integrating all components

### 5. Environment & Demonstration (`sne_environment.py`)
- **DemonstrationEnvironment**: Simple environment for testing
- **Visualization Tools**: Functions to visualize exploration behavior

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-neural-explorer.git
   cd smart-neural-explorer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run a demonstration of the Smart Neural Explorer:

```bash
python main.py
```

This will:
1. Initialize the agent and environment
2. Run a demonstration episode
3. Visualize the exploration behavior and results

## Project Structure

```
smart-neural-explorer/
├── README.md
├── LICENSE
├── requirements.txt
├── main.py                  # Main entry point
├── sne_basics.py            # Core enumerations and data structures
├── sne_neural.py            # Neural network modules
├── sne_core.py              # Cognitive architecture components
├── sne_agent.py             # Main agent implementation
└── sne_environment.py       # Environment and demonstration tools
```

## Examples

### Basic Usage

```python
from sne_agent import SmartNeuralExplorer
from sne_environment import DemonstrationEnvironment

# Initialize agent and environment
agent = SmartNeuralExplorer(state_dim=4, action_dim=2)
env = DemonstrationEnvironment(state_dim=4, action_dim=2)

# Run an episode
state = env.reset()
done = False
while not done:
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    agent.store_experience(state, action, reward, next_state, done)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by meta-cognitive theories in psychology
- Built on PyTorch for efficient neural network implementation
- Incorporates ideas from curiosity-driven and attention-based reinforcement learning

## Authors

- Your Name - [GitHub Profile](https://github.com/yourusername)

## Citation

If you use this project in your research, please consider citing it:

```
@misc{smart-neural-explorer,  
    author = {Your Name},  
    title = {Smart Neural Explorer: Meta-cognitive Exploration with Dynamic Attention},  
    year = {2026},  
    publisher = {GitHub},  
    journal = {GitHub repository},  
    howpublished = {\url{https://github.com/yourusername/smart-neural-explorer}},  
}
```