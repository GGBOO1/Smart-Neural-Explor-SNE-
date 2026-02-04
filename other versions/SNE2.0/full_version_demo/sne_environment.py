import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
from sne_agent import SmartNeuralExplorer

class DemonstrationEnvironment:
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = np.zeros(state_dim)
        self.goal = np.ones(state_dim) * 0.5
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.step_count += 1
        
        action = np.clip(action, -1, 1)
        state_update = np.zeros_like(self.state)
        state_update[:len(action)] = action
        self.state = self.state + state_update * 0.1
        
        distance = np.linalg.norm(self.state - self.goal)
        reward = 1.0 / (1.0 + distance)
        
        reward -= 0.01 * np.linalg.norm(action)
        
        done = (distance < 0.1) or (self.step_count >= self.max_steps)
        
        info = {
            'distance_to_goal': distance,
            'step': self.step_count
        }
        
        return self.state.copy(), reward, done, info
    
    def render(self):
        print(f"State: {self.state.round(3)}, Distance to goal: {np.linalg.norm(self.state - self.goal):.3f}")

def run_demo_episode(agent: SmartNeuralExplorer, env: DemonstrationEnvironment, 
                    max_steps: int = 200, verbose: bool = True):
    state = env.reset()
    total_reward = 0
    done = False
    
    history = {
        'states': [],
        'actions': [],
        'rewards': [],
        'attention_focus': [],
        'cognitive_states': [],
        'intention_types': []
    }
    
    step = 0
    while not done and step < max_steps:
        action = agent.select_action(state)
        
        next_state, reward, done, info = env.step(action)
        
        agent.store_experience(state, action, reward, next_state, done)
        
        history['states'].append(state.copy())
        history['actions'].append(action.copy())
        history['rewards'].append(reward)
        
        agent_info = agent.get_agent_info()
        history['cognitive_states'].append(agent_info['meta_metrics']['cognitive_state'])
        
        if step > 0:
            recent_stats = agent.stats['exploration_types']
            if recent_stats:
                most_common = max(recent_stats, key=recent_stats.get)
                history['intention_types'].append(most_common)
        
        total_reward += reward
        state = next_state
        step += 1
        
        if verbose and step % 50 == 0:
            print(f"Step {step}: Reward={reward:.3f}, Total={total_reward:.3f}")
            print(f"  Cognitive State: {agent_info['meta_metrics']['cognitive_state']}")
            print(f"  Exploration Phase: {agent_info['exploration_phase']}")
    
    if verbose:
        print(f"\nEpisode completed: {step} steps, Total reward: {total_reward:.3f}")
        print(f"Best reward achieved: {agent.best_reward:.3f}")
        print(f"Final cognitive load: {agent.cognitive_load:.3f}")
    
    return total_reward, history, agent_info

def visualize_demo_results(history: Dict, agent_info: Dict, save_path: str = None):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Smart Neural Explorer - Demonstration Results', fontsize=16, fontweight='bold')
    
    if 'rewards' in history and history['rewards']:
        ax = axes[0, 0]
        ax.plot(history['rewards'], 'b-', alpha=0.7, linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Progression')
        ax.grid(True, alpha=0.3)
    
    if 'cognitive_states' in history and history['cognitive_states']:
        ax = axes[0, 1]
        states = history['cognitive_states']
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
        if state_counts:
            ax.bar(state_counts.keys(), state_counts.values(), color='green', alpha=0.7)
            ax.set_xlabel('Cognitive State')
            ax.set_ylabel('Count')
            ax.set_title('Cognitive State Distribution')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    if 'intention_types' in history and history['intention_types']:
        ax = axes[0, 2]
        intentions = history['intention_types']
        intent_counts = {}
        for intent in intentions:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        if intent_counts:
            ax.bar(intent_counts.keys(), intent_counts.values(), color='purple', alpha=0.7)
            ax.set_xlabel('Intention Type')
            ax.set_ylabel('Count')
            ax.set_title('Intention Type Distribution')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    if 'states' in history and history['states'] and len(history['states'][0]) >= 2:
        ax = axes[1, 0]
        states = np.array(history['states'])
        ax.plot(states[:, 0], states[:, 1], 'r-', alpha=0.6, linewidth=2)
        ax.scatter(states[0, 0], states[0, 1], color='green', s=100, label='Start')
        ax.scatter(states[-1, 0], states[-1, 1], color='red', s=100, label='End')
        ax.set_xlabel('State Dimension 1')
        ax.set_ylabel('State Dimension 2')
        ax.set_title('State Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if 'actions' in history and history['actions']:
        ax = axes[1, 1]
        actions = np.array(history['actions'])
        if actions.shape[1] >= 2:
            ax.plot(actions[:, 0], actions[:, 1], 'g-', alpha=0.6, linewidth=2)
            ax.set_xlabel('Action Dimension 1')
            ax.set_ylabel('Action Dimension 2')
            ax.set_title('Action Sequence')
            ax.grid(True, alpha=0.3)
    
    if 'rewards' in history and history['rewards']:
        ax = axes[1, 2]
        rewards = history['rewards']
        if len(rewards) > 1:
            moving_avg = np.convolve(rewards, np.ones(5)/5, mode='valid')
            ax.plot(moving_avg, 'b-', alpha=0.8, linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Moving Average Reward')
            ax.set_title('5-Step Moving Average Reward')
            ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    ax.axis('off')
    info_text = f"Episode Summary:\n"
    info_text += f"Total Steps: {len(history['rewards'])}\n"
    info_text += f"Total Reward: {sum(history['rewards']):.3f}\n"
    if 'rewards' in history and history['rewards']:
        info_text += f"Average Reward: {np.mean(history['rewards']):.3f}\n"
    if agent_info:
        info_text += f"Final Exploration Phase: {agent_info.get('exploration_phase', 'N/A')}\n"
        info_text += f"Final Cognitive Load: {agent_info.get('cognitive_load', 0):.3f}\n"
        info_text += f"Experience Buffer Size: {agent_info.get('experience_buffer_size', 0)}"
    ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    
    if agent_info and 'stats' in agent_info:
        ax = axes[2, 1]
        stats = agent_info['stats']
        if 'exploration_types' in stats:
            exp_types = stats['exploration_types']
            if exp_types:
                ax.pie(exp_types.values(), labels=exp_types.keys(), autopct='%1.1f%%')
                ax.set_title('Exploration Type Distribution')
    
    ax = axes[2, 2]
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()