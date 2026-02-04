"""
Smart Neural Explorer (SNE) - Main Entry Point
A PyTorch-based implementation of meta-cognitive exploration with dynamic attention
GitHub: https://github.com/yourusername/smart-neural-explorer
"""

import numpy as np
import torch
from sne_agent import SmartNeuralExplorer
from sne_environment import DemonstrationEnvironment, run_demo_episode, visualize_demo_results

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def main():
    """Main function to run the Smart Neural Explorer demonstration"""
    print("Smart Neural Explorer (SNE) - Demonstration")
    print("=" * 60)
    
    # Configuration
    state_dim = 4
    action_dim = 2
    max_steps = 200
    
    # Initialize components
    print("Initializing components...")
    agent = SmartNeuralExplorer(state_dim=state_dim, action_dim=action_dim)
    env = DemonstrationEnvironment(state_dim=state_dim, action_dim=action_dim)
    
    print(f"Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    print(f"Environment initialized with goal at {env.goal.round(2)}")
    print("=" * 60)
    
    # Run demonstration episode
    print("Running demonstration episode...")
    print("=" * 60)
    
    total_reward, history, agent_info = run_demo_episode(
        agent=agent,
        env=env,
        max_steps=max_steps,
        verbose=True
    )
    
    print("=" * 60)
    print("Episode completed!")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Best reward: {agent.best_reward:.3f}")
    print(f"Steps taken: {len(history['rewards'])}")
    print("=" * 60)
    
    # Visualize results
    print("Generating visualization...")
    visualize_demo_results(history, agent_info)
    
    # Print final agent information
    print("\nFinal Agent State:")
    print("-" * 40)
    print(f"Exploration Phase: {agent_info['exploration_phase']}")
    print(f"Cognitive Load: {agent_info['cognitive_load']:.3f}")
    print(f"Experience Buffer Size: {agent_info['experience_buffer_size']}")
    print(f"Meta Cognitive State: {agent_info['meta_metrics']['cognitive_state']}")
    print(f"Learning Progress: {agent_info['meta_metrics']['learning_progress']:.3f}")
    print("-" * 40)
    
    print("\nExploration Type Distribution:")
    for exp_type, count in agent_info['stats']['exploration_types'].items():
        print(f"  {exp_type}: {count}")
    print("=" * 60)
    print("Demonstration complete!")


if __name__ == "__main__":
    main()