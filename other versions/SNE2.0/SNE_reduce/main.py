from sne_simplified_agent import SimpleSmartNeuralExplorer
from sne_simplified_environment import SimpleDemonstrationEnvironment
import numpy as np

def run_demonstration():
    print("=== Simple Smart Neural Explorer Demonstration ===")
    
    state_dim = 4
    action_dim = 2
    
    agent = SimpleSmartNeuralExplorer(state_dim=state_dim, action_dim=action_dim)
    env = SimpleDemonstrationEnvironment(state_dim=state_dim, action_dim=action_dim)
    
    state = env.reset()
    done = False
    total_reward = 0
    
    print(f"Initial state: {state[:2]}")
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        agent.store_experience(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if env.steps % 10 == 0:
            agent_info = agent.get_agent_info()
            env.render(agent_info)
    
    print("=== Demonstration Complete ===")
    print(f"Total steps: {env.steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Best reward: {agent.best_reward:.2f}")
    print(f"Exploration types: {agent.stats['exploration_types']}")
    
    agent_info = agent.get_agent_info()
    print(f"Final agent info: {agent_info}")

if __name__ == "__main__":
    run_demonstration()
