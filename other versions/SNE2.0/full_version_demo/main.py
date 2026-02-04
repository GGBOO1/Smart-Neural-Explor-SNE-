from sne_agent import SmartNeuralExplorer
from sne_environment import DemonstrationEnvironment, run_demo_episode, visualize_demo_results
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Language dictionaries
LANG_EN = {
    'title': 'Smart Neural Explorer (SNE) - Full Version Demonstration',
    'initializing': 'Initializing components...',
    'agent_init': 'Agent initialized with state_dim={}, action_dim={}',
    'env_init': 'Environment initialized with goal at {}',
    'running': 'Running demonstration episode...',
    'completed': 'Episode completed!',
    'total_reward': 'Total reward: {:.3f}',
    'best_reward': 'Best reward: {:.3f}',
    'steps_taken': 'Steps taken: {}',
    'generating': 'Generating visualization...',
    'final_state': 'Final Agent State:',
    'exploration_phase': 'Exploration Phase: {}',
    'cognitive_load': 'Cognitive Load: {:.3f}',
    'buffer_size': 'Experience Buffer Size: {}',
    'cognitive_state': 'Meta Cognitive State: {}',
    'learning_progress': 'Learning Progress: {:.3f}',
    'insight_count': 'Insight Count: {}',
    'confusion_count': 'Confusion Count: {}',
    'exploration_dist': 'Exploration Type Distribution:',
    'demo_complete': 'Demonstration complete!',
    'priority_stats': 'Priority Statistics:',
    'avg_priority': 'Average Priority: {:.3f}',
    'emergency_active': 'Emergency Active: {}',
    'emergency_count': 'Emergency Count: {}',
    'lang_prompt': 'Please select language / 请选择语言:\n1. English\n2. 中文\nEnter your choice (1/2): '
}

LANG_CN = {
    'title': '智能神经探索器 (SNE) - 完整版演示',
    'initializing': '正在初始化组件...',
    'agent_init': '智能体已初始化，state_dim={}, action_dim={}',
    'env_init': '环境已初始化，目标位置为 {}',
    'running': '正在运行演示 episode...',
    'completed': 'Episode 完成！',
    'total_reward': '总奖励: {:.3f}',
    'best_reward': '最佳奖励: {:.3f}',
    'steps_taken': '步数: {}',
    'generating': '正在生成可视化...',
    'final_state': '最终智能体状态:',
    'exploration_phase': '探索阶段: {}',
    'cognitive_load': '认知负载: {:.3f}',
    'buffer_size': '经验缓冲区大小: {}',
    'cognitive_state': '元认知状态: {}',
    'learning_progress': '学习进度: {:.3f}',
    'insight_count': '洞察次数: {}',
    'confusion_count': '困惑次数: {}',
    'exploration_dist': '探索类型分布:',
    'demo_complete': '演示完成！',
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

def main():
    """Main function to run the Smart Neural Explorer demonstration"""
    # Get language choice
    lang_code, lang = get_language_choice()
    
    print(lang['title'])
    print("=" * 80)
    
    # Configuration
    state_dim = 4
    action_dim = 2
    max_steps = 200
    
    # Initialize components
    print(lang['initializing'])
    agent = SmartNeuralExplorer(state_dim=state_dim, action_dim=action_dim)
    env = DemonstrationEnvironment(state_dim=state_dim, action_dim=action_dim)
    
    print(lang['agent_init'].format(state_dim, action_dim))
    print(lang['env_init'].format(env.goal.round(2)))
    print("=" * 80)
    
    # Run demonstration episode
    print(lang['running'])
    print("=" * 80)
    
    total_reward, history, agent_info = run_demo_episode(
        agent=agent,
        env=env,
        max_steps=max_steps,
        verbose=True
    )
    
    print("=" * 80)
    print(lang['completed'])
    print(lang['total_reward'].format(total_reward))
    print(lang['best_reward'].format(agent.best_reward))
    print(lang['steps_taken'].format(len(history['rewards'])))
    print("=" * 80)
    
    # Visualize results
    print(lang['generating'])
    visualize_demo_results(history, agent_info)
    
    # Print final agent information
    print("\n" + lang['final_state'])
    print("-" * 60)
    print(lang['exploration_phase'].format(agent_info['exploration_phase']))
    print(lang['cognitive_load'].format(agent_info['cognitive_load']))
    print(lang['buffer_size'].format(agent_info['experience_buffer_size']))
    print(lang['cognitive_state'].format(agent_info['meta_metrics']['cognitive_state']))
    print(lang['learning_progress'].format(agent_info['meta_metrics']['learning_progress']))
    print(lang['insight_count'].format(agent_info['meta_metrics']['insight_count']))
    print(lang['confusion_count'].format(agent_info['meta_metrics']['confusion_count']))
    print("-" * 60)
    
    # Print priority statistics
    print("\n" + lang['priority_stats'])
    print("-" * 60)
    print(lang['avg_priority'].format(agent_info.get('priority_stats', {}).get('avg_priority', 0)))
    print(lang['emergency_active'].format(agent_info.get('emergency_status', {}).get('is_active', False)))
    print(lang['emergency_count'].format(agent_info.get('stats', {}).get('emergency_triggers', 0)))
    print("-" * 60)
    
    print("\n" + lang['exploration_dist'])
    print("-" * 60)
    for exp_type, count in agent_info['stats']['exploration_types'].items():
        print(f"  {exp_type}: {count}")
    print("-" * 60)
    print(lang['demo_complete'])


if __name__ == "__main__":
    main()