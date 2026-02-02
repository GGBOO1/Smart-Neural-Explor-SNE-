import numpy as np
import matplotlib.pyplot as plt
from sne_simplified_basics import ActionType, ExplorationStrategy, CognitiveState
from sne_simplified_neural import SimplePolicyNetwork, SimpleValueNetwork, SimpleNoveltyDetector
from sne_simplified_core import SimpleAttentionMechanism, SimpleIntentionGenerator, SimpleExperienceManager
from sne_simplified_agent import SimpleSNEAgent
from sne_simplified_environment import SimplifiedEnvironment

def run_simple_sne_demo():
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
            print(f"Step: {step}, Reward: {reward:.4f}, Epsilon: {agent.epsilon:.4f}")
    env.animate(filename='simple_sne_trajectory.mp4')
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Simple SNE Agent Rewards Over Time')
    plt.grid(True)
    plt.savefig('simple_sne_rewards.png')
    plt.close()
    print(f"\nDemo completed!")
    print(f"Total steps: {step}")
    print(f"Average reward: {np.mean(total_rewards):.4f}")
    print(f"Maximum reward: {np.max(total_rewards):.4f}")
    print(f"Minimum reward: {np.min(total_rewards):.4f}")

if __name__ == "__main__":
    run_simple_sne_demo()