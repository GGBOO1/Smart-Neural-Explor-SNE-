import numpy as np
from sne_agent import SmartNeuralExplorer
from sne_environment import MultiModalEnvironment

if __name__ == "__main__":
    env = MultiModalEnvironment(state_dim=6, action_dim=2, max_steps=150)
    agent = SmartNeuralExplorer(state_dim=6, action_dim=2)
    
    episodes = 10
    total_rewards = []
    exploration_rates = []
    emotional_states = []
    learning_styles = []
    
    print("Starting enhanced SNE algorithm testing...")
    print(f"Environment modalities: {env.get_modalities()}")
    print(f"State dimension: {env.get_state_dim()}")
    print(f"Action dimension: {env.get_action_dim()}")
    print("=" * 80)
    
    for episode in range(episodes):
        multi_modal_state = env.reset()
        unified_state = env._unify_modalities(multi_modal_state)
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action = agent.select_action(unified_state)
            next_multi_modal_state, reward, done, info = env.step(action)
            next_unified_state = info['unified_state']
            
            agent.store_experience(unified_state, action, reward, next_unified_state, done)
            
            episode_reward += reward
            unified_state = next_unified_state
            step_count += 1
        
        total_rewards.append(episode_reward)
        agent_info = agent.get_agent_info()
        
        exploration_rates.append(agent_info['exploration_rate'])
        emotional_states.append(agent_info['emotional_state'])
        learning_styles.append(agent_info['current_learning_style'])
        
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Steps: {step_count}")
        print(f"Best Reward: {agent.best_reward:.2f}")
        print(f"Emotional State: {agent_info['emotional_state']}")
        print(f"Learning Style: {agent_info['current_learning_style']}")
        print(f"Cognitive Flexibility: {agent_info['cognitive_flexibility']:.2f}")
        print(f"Cognitive State: {agent_info['meta_metrics']['cognitive_state']}")
        
        if 'current_focus' in agent_info:
            print(f"Current Focus: {agent_info['current_focus']}")
        
        top_exploration_types = sorted(
            agent_info['stats']['exploration_types'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        print(f"Top Exploration Types: {dict(top_exploration_types)}")
        
        print("=" * 80)
    
    print("Testing completed!")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    
    unique_emotional_states = set(emotional_states)
    unique_learning_styles = set(learning_styles)
    print(f"Emotional States Explored: {unique_emotional_states}")
    print(f"Learning Styles Used: {unique_learning_styles}")
    
    print("\nEnhanced SNE Algorithm Performance Summary:")
    print("==========================================")
    print("1. Human-like cognitive processes:")
    print("   - Emotional state integration")
    print("   - Learning style adaptation")
    print("   - Cognitive flexibility")
    print("   - Multi-modal sensory processing")
    print("   - Semantic association network")
    print("\n2. Exploration capabilities:")
    print("   - Adaptive random action selection")
    print("   - Context-aware decision making")
    print("   - Associative learning")
    print("   - Emotional influence on exploration")
    print("\n3. Performance metrics:")
    print(f"   - Average reward: {np.mean(total_rewards):.2f}")
    print(f"   - Reward consistency: {np.std(total_rewards):.2f}")
    print(f"   - Exploration diversity: {len(unique_learning_styles)} learning styles")