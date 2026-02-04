import numpy as np
import matplotlib.pyplot as plt
from sne_simplified_basics import ActionType, ExplorationStrategy, CognitiveState
from sne_simplified_neural import SimplePolicyNetwork, SimpleValueNetwork, SimpleNoveltyDetector
from sne_simplified_core import SimpleAttentionMechanism, SimpleIntentionGenerator, SimpleExperienceManager
from sne_simplified_agent import SimpleSNEAgent
from sne_simplified_environment import SimplifiedEnvironment

# Language dictionaries
LANG_EN = {
    'title': 'Smart Neural Explorer (SNE) - Simplified Version Demonstration',
    'initializing': 'Initializing components...',
    'running': 'Running demonstration...',
    'step_info': 'Step: {}, Reward: {:.4f}, Epsilon: {:.4f}',
    'generating': 'Generating animation and visualization...',
    'completed': 'Demo completed!',
    'total_steps': 'Total steps: {}',
    'avg_reward': 'Average reward: {:.4f}',
    'max_reward': 'Maximum reward: {:.4f}',
    'min_reward': 'Minimum reward: {:.4f}',
    'priority_stats': 'Priority Statistics:',
    'avg_priority': 'Average Priority: {:.3f}',
    'emergency_active': 'Emergency Active: {}',
    'emergency_count': 'Emergency Count: {}',
    'lang_prompt': 'Please select language / 请选择语言:\n1. English\n2. 中文\nEnter your choice (1/2): '
}

LANG_CN = {
    'title': '智能神经探索器 (SNE) - 简化版演示',
    'initializing': '正在初始化组件...',
    'running': '正在运行演示...',
    'step_info': '步骤: {}, 奖励: {:.4f}, 探索率: {:.4f}',
    'generating': '正在生成动画和可视化...',
    'completed': '演示完成！',
    'total_steps': '总步数: {}',
    'avg_reward': '平均奖励: {:.4f}',
    'max_reward': '最大奖励: {:.4f}',
    'min_reward': '最小奖励: {:.4f}',
    'priority_stats': '优先级统计:',
    'avg_priority': '平均优先级: {:.3f}',
    'emergency_active': '紧急状态激活: {}',
    'emergency_count': '紧急次数: {}',
    'lang_prompt': 'Please select language / 请选择语言:\n1. English\n2. 中文\nEnter your choice (1/2): '
}

def get_language_choice():
    """Get user language choice"""
    while True:
        try:
            choice = input(LANG_EN['lang_prompt'])
            if choice == '1':
                return 'en', LANG_EN
            elif choice == '2':
                return 'cn', LANG_CN
            else:
                print('Invalid choice. Please enter 1 or 2.')
        except:
            return 'en', LANG_EN

def run_simple_sne_demo():
    # Get language choice
    lang_code, lang = get_language_choice()
    
    print(lang['title'])
    print("=" * 80)
    
    print(lang['initializing'])
    env = SimplifiedEnvironment(state_dim=4, action_dim=2, max_steps=100)
    policy_net = SimplePolicyNetwork(input_dim=4, output_dim=2, hidden_dims=[64, 32])
    value_net = SimpleValueNetwork(input_dim=4, hidden_dims=[64, 32])
    novelty_detector = SimpleNoveltyDetector(input_dim=4, hidden_dims=[64, 32])
    attention_mechanism = SimpleAttentionMechanism(input_dim=4, hidden_dim=32)
    intention_generator = SimpleIntentionGenerator(state_dim=4, action_dim=2, hidden_dim=32)
    experience_manager = SimpleExperienceManager(capacity=1000, batch_size=32)
    agent = SimpleSNEAgent(
        state_dim=4,
        action_dim=2,
        policy_network=policy_net,
        value_network=value_net,
        novelty_detector=novelty_detector,
        attention_mechanism=attention_mechanism,
        intention_generator=intention_generator,
        experience_manager=experience_manager,
        exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        learning_rate=0.001,
        discount_factor=0.99,
        novelty_threshold=0.7
    )
    
    print(lang['running'])
    print("=" * 80)
    
    total_rewards = []
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)
        env.render(mode='plot')
        total_rewards.append(reward)
        state = next_state
        step += 1
        if step % 10 == 0:
            print(lang['step_info'].format(step, reward, agent.epsilon))
    
    print(lang['generating'])
    env.animate(filename='simple_sne_trajectory.mp4')
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards)
    plt.xlabel('Step' if lang_code == 'en' else '步骤')
    plt.ylabel('Reward' if lang_code == 'en' else '奖励')
    plt.title('Simple SNE Agent Rewards Over Time' if lang_code == 'en' else '简化版 SNE 智能体奖励随时间变化')
    plt.grid(True)
    plt.savefig('simple_sne_rewards.png')
    plt.close()
    
    print("=" * 80)
    print(lang['completed'])
    print(lang['total_steps'].format(step))
    print(lang['avg_reward'].format(np.mean(total_rewards)))
    print(lang['max_reward'].format(np.max(total_rewards)))
    print(lang['min_reward'].format(np.min(total_rewards)))
    
    # Get agent info and print priority statistics
    agent_info = agent.get_agent_info()
    if 'priority_stats' in agent_info:
        print("\n" + lang['priority_stats'])
        print("-" * 60)
        print(lang['avg_priority'].format(agent_info['priority_stats'].get('avg_priority', 0)))
        print(lang['emergency_active'].format(agent_info['priority_stats'].get('emergency_active', False)))
        print(lang['emergency_count'].format(agent_info['stats'].get('emergency_triggers', 0)))
    
    print("=" * 80)

if __name__ == "__main__":
    run_simple_sne_demo()