from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import heapq
from typing import List, Dict, Tuple, Optional, Any
from sne_basics import (
    AttentionGranularity, IntentionType, CognitiveState, ExplorationPhase,
    Experience, AttentionFocus, MetaIntention, EmotionalState, ContextFrame
)
from sne_neural import StateEncoder, UncertaintyEstimator, CuriosityModule
from sne_core import WorkingMemory, DynamicAttentionManager, CreativeIntentionGenerator, MetaCognitiveMonitor

class PrioritizedExperienceBuffer:
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta  # 重要性采样权重指数
        self.beta_increment = beta_increment
        self.buffer = []
        self.position = 0
        self.priorities = []
        self.max_priority = 1.0
    
    def push(self, experience: Experience, priority: float = None):
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            heapq.heappush(self.priorities, (-priority, len(self.buffer) - 1))
        else:
            self.buffer[self.position] = experience
            heapq.heappush(self.priorities, (-priority, self.position))
            self.position = (self.position + 1) % self.capacity
        
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return [], [], []
        
        indices = []
        experiences = []
        weights = []
        
        total = len(self.buffer)
        for _ in range(batch_size):
            if not self.priorities:
                break
            
            priority, idx = heapq.heappop(self.priorities)
            priority = -priority
            
            if idx < len(self.buffer):
                indices.append(idx)
                experiences.append(self.buffer[idx])
                
                # 计算重要性采样权重
                probability = priority / sum(abs(p) for p, _ in self.priorities) if self.priorities else 1.0
                weight = (total * probability) ** (-self.beta)
                weights.append(weight)
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 归一化权重
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], new_priorities: List[float]):
        for idx, priority in zip(indices, new_priorities):
            heapq.heappush(self.priorities, (-priority, idx))
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class SmartNeuralExplorer:
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_count = 0
        
        self.state_encoder = StateEncoder(state_dim)
        self.uncertainty_estimator = UncertaintyEstimator(state_dim, action_dim)
        self.curiosity_module = CuriosityModule(state_dim, action_dim)
        
        self.attention_manager = DynamicAttentionManager(state_dim, action_dim)
        self.intention_generator = CreativeIntentionGenerator(action_dim, state_dim)
        self.meta_monitor = MetaCognitiveMonitor()
        
        # 优先级经验回放缓冲区
        self.experience_buffer = PrioritizedExperienceBuffer(capacity=10000)
        self.best_action = None
        self.best_reward = -float('inf')
        
        # 强化学习相关
        self.dqn = DQNNetwork(state_dim, action_dim)
        self.target_dqn = DQNNetwork(state_dim, action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.99  # 折扣因子
        self.batch_size = 32
        self.target_update_frequency = 100
        
        self.exploration_phase = ExplorationPhase.EARLY
        self.exploration_rate = 1.0
        self.cognitive_load = 0.0
        
        self.emotional_state = EmotionalState.NEUTRAL
        self.cognitive_flexibility = 1.0
        self.learning_styles = ['analytical', 'holistic', 'pragmatic', 'creative']
        self.current_learning_style = 'analytical'
        self.style_switch_counter = 0
        self.style_switch_threshold = 20
        
        # 状态访问计数
        self.state_visit_counts = defaultdict(int)
        self.state_action_visit_counts = defaultdict(int)
        
        # 探索参数
        self.exploration_bonus_weight = 0.1
        self.count_decay = 0.999
        
        self.stats = {
            'exploration_types': defaultdict(int),
            'attention_changes': 0,
            'intentions_generated': 0,
            'meta_state_changes': 0,
            'emotional_states': defaultdict(int),
            'learning_styles': defaultdict(int),
            'rl_updates': 0,
            'average_loss': 0.0,
            'unique_states_visited': 0,
            'exploration_bonus': 0.0
        }
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.step_count += 1
        
        self._update_exploration_phase()
        self._update_emotional_state()
        self._update_learning_style()
        
        if self._should_use_random_action(state):
            action = np.random.uniform(-1, 1, self.action_dim)
            self.stats['exploration_types']['random_adaptive'] += 1
            
            # 更新访问计数
            self._update_visit_counts(state, action)
            return action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        uncertainty = self._estimate_uncertainty(state_tensor)
        
        novelty = self._calculate_novelty(state)
        
        cognitive_state = self.meta_monitor.update_state(
            reward=0,
            uncertainty=uncertainty.mean(),
            novelty=novelty,
            step=self.step_count
        )
        
        self.cognitive_load = self._calculate_cognitive_load()
        
        associative_strength = self._get_associative_strength(state)
        attention_focus = self.attention_manager.select_focus(
            state=state,
            uncertainty=uncertainty,
            novelty=novelty,
            associative_strength=associative_strength,
            cognitive_state=cognitive_state,
            exploration_phase=self.exploration_phase
        )
        
        intentions = self.intention_generator.generate_intentions(
            state=state,
            focus=attention_focus,
            best_action=self.best_action,
            experiences=list(self.experience_buffer),
            num_intentions=5
        )
        
        self.stats['intentions_generated'] += len(intentions)
        
        selected_intention = self.attention_manager.gated_selection(
            intentions=intentions,
            cognitive_load=self.cognitive_load
        )
        
        if selected_intention is None:
            action = np.random.uniform(-1, 1, self.action_dim)
            intention_type = IntentionType.RANDOM
        else:
            action = self._intention_to_action(selected_intention, attention_focus)
            intention_type = selected_intention.intention_type
        
        action = self._add_exploration_noise(action)
        
        # 更新访问计数
        self._update_visit_counts(state, action)
        
        self.stats['exploration_types'][intention_type.value] += 1
        self.stats['emotional_states'][self.emotional_state.value] += 1
        self.stats['learning_styles'][self.current_learning_style] += 1
        
        return action
    
    def _intention_to_action(self, intention: MetaIntention, 
                            focus: AttentionFocus) -> np.ndarray:
        base_action = intention.base_action.copy()
        
        if intention.intention_type == IntentionType.MODIFICATION:
            noise = np.random.randn(*base_action.shape) * intention.fuzziness
            action = base_action + noise
            
        elif intention.intention_type == IntentionType.ANALOGICAL:
            alpha = np.random.uniform(0.3, 0.7)
            action = alpha * base_action + (1 - alpha) * np.random.randn(*base_action.shape)
            
        elif intention.intention_type == IntentionType.COMBINATION:
            action = base_action.copy()
            
        elif intention.intention_type == IntentionType.COUNTERFACTUAL:
            action = np.tanh(base_action)
            
        else:
            action = base_action.copy()
        
        if focus.granularity == AttentionGranularity.FINE:
            action = action * 0.9 + np.random.randn(*action.shape) * 0.1
        elif focus.granularity == AttentionGranularity.COARSE:
            pass
        
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _add_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        if self.exploration_phase == ExplorationPhase.EARLY:
            noise_level = 0.3
        elif self.exploration_phase == ExplorationPhase.MIDDLE:
            noise_level = 0.1
        else:
            noise_level = 0.05
        
        noise = np.random.randn(*action.shape) * noise_level
        return np.clip(action + noise, -1.0, 1.0)
    
    def _estimate_uncertainty(self, state: torch.Tensor) -> np.ndarray:
        if len(self.experience_buffer) < 10:
            return np.ones(self.state_dim) * 0.8
        
        uncertainties = []
        for _ in range(5):
            random_action = torch.randn(1, self.action_dim)
            with torch.no_grad():
                uncertainty = self.uncertainty_estimator(state, random_action)
            uncertainties.append(uncertainty.item())
        
        avg_uncertainty = np.mean(uncertainties)
        return np.ones(self.state_dim) * avg_uncertainty
    
    def _calculate_novelty(self, state: np.ndarray) -> float:
        if len(self.experience_buffer) < 5:
            return 1.0
        
        states = np.array([exp.state for exp in list(self.experience_buffer)[-50:]])
        distances = np.linalg.norm(states - state, axis=1)
        min_distance = np.min(distances) if len(distances) > 0 else 1.0
        
        novelty = np.exp(-min_distance)
        return float(novelty)
    
    def _get_associative_strength(self, state: np.ndarray) -> np.ndarray:
        return np.random.rand(self.state_dim) * 0.5
    
    def _calculate_cognitive_load(self) -> float:
        base_load = 0.3
        
        if self.exploration_phase == ExplorationPhase.EARLY:
            base_load += 0.2
        
        if self.meta_monitor.current_state == CognitiveState.CONFUSED:
            base_load += 0.3
        
        base_load += np.random.uniform(-0.1, 0.1)
        
        return min(1.0, max(0.0, base_load))
    
    def _update_exploration_phase(self):
        if self.step_count < 100:
            self.exploration_phase = ExplorationPhase.EARLY
            self.exploration_rate = 1.0
        elif self.step_count < 500:
            self.exploration_phase = ExplorationPhase.MIDDLE
            self.exploration_rate = 0.5
        else:
            self.exploration_phase = ExplorationPhase.LATE
            self.exploration_rate = 0.2
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        novelty = self._calculate_novelty(state)
        uncertainty = self._estimate_uncertainty(torch.FloatTensor(state).unsqueeze(0)).mean()
        
        # 计算内在奖励
        intrinsic_reward = self._calculate_intrinsic_reward(state, action, next_state)
        
        # 计算基于计数的探索奖励
        exploration_bonus = self._calculate_exploration_bonus(state, action)
        
        total_reward = reward + intrinsic_reward + exploration_bonus
        
        is_success = total_reward > 0
        
        experience = Experience(
            state=state.copy(),
            action=action.copy(),
            reward=total_reward,
            next_state=next_state.copy(),
            done=done,
            novelty=novelty,
            uncertainty=uncertainty,
            timestamp=self.step_count
        )
        
        # 计算优先级
        priority = self._calculate_experience_priority(experience)
        
        if is_success:
            for _ in range(3):
                self.experience_buffer.push(experience, priority * 1.5)  # 成功经验优先级更高
        else:
            self.experience_buffer.push(experience, priority)
        
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_action = action.copy()
            
            if hasattr(self.intention_generator, 'success_patterns'):
                self.intention_generator.success_patterns.append((state.copy(), action.copy(), total_reward))
        
        if total_reward > 0.5:
            if hasattr(self.intention_generator, 'semantic_network'):
                self.intention_generator.semantic_network.store_in_long_term_memory(
                    content={"state": state, "action": action, "reward": total_reward},
                    importance=total_reward,
                    timestamp=self.step_count,
                    associations=[f"success_state_{i}_{state[i]:.2f}" for i in range(len(state))],
                    emotional_tag=EmotionalState.POSITIVE
                )
        
        self.meta_monitor.update_state(
            reward=total_reward,
            uncertainty=uncertainty,
            novelty=novelty,
            step=self.step_count
        )
        
        # 执行强化学习训练
        self._train_rl()
    
    def _calculate_intrinsic_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """计算内在奖励"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        with torch.no_grad():
            _, _, intrinsic_reward, _ = self.curiosity_module(
                state_tensor, action_tensor, next_state_tensor
            )
        
        return float(intrinsic_reward)
    
    def _calculate_experience_priority(self, experience: Experience) -> float:
        """计算经验的优先级"""
        # 基于奖励、新奇度和不确定性计算优先级
        priority = abs(experience.reward) * 2.0
        priority += experience.novelty * 1.0
        priority += experience.uncertainty * 0.5
        
        # 成功经验优先级更高
        if experience.reward > 0:
            priority *= 1.5
        
        return max(priority, 0.1)  # 最小优先级
    
    def _train_rl(self):
        """执行强化学习训练"""
        if len(self.experience_buffer) < self.batch_size * 2:
            return
        
        # 采样经验
        experiences, indices, weights = self.experience_buffer.sample(self.batch_size)
        if not experiences:
            return
        
        # 准备数据
        states = torch.FloatTensor([exp.state for exp in experiences])
        actions = torch.FloatTensor([exp.action for exp in experiences])
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences])
        dones = torch.FloatTensor([exp.done for exp in experiences]).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.dqn(states)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_dqn(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q.max(dim=1, keepdim=True)[0]
        
        # 计算损失
        loss = (weights * (current_q - target_q) ** 2).mean()
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        if self.step_count % self.target_update_frequency == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        # 更新经验优先级
        new_priorities = (abs(current_q.detach() - target_q.detach()).mean(dim=1) + 0.1).numpy()
        self.experience_buffer.update_priorities(indices, new_priorities)
        
        # 更新统计信息
        self.stats['rl_updates'] += 1
        self.stats['average_loss'] = (
            self.stats['average_loss'] * 0.99 + float(loss) * 0.01
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        meta_metrics = self.meta_monitor.get_metrics()
        
        info = {
            'step': self.step_count,
            'exploration_phase': self.exploration_phase.value,
            'exploration_rate': self.exploration_rate,
            'cognitive_load': self.cognitive_load,
            'emotional_state': self.emotional_state.value,
            'cognitive_flexibility': self.cognitive_flexibility,
            'current_learning_style': self.current_learning_style,
            'experience_buffer_size': len(self.experience_buffer),
            'best_reward': self.best_reward,
            'stats': dict(self.stats),
            'meta_metrics': meta_metrics
        }
        
        if self.attention_manager.current_focus:
            info['current_focus'] = str(self.attention_manager.current_focus)
        
        return info
    
    def _update_emotional_state(self):
        if not self.experience_buffer:
            return
        
        recent_experiences = list(self.experience_buffer)[-10:]
        avg_recent_reward = np.mean([exp.reward for exp in recent_experiences])
        
        if len(self.experience_buffer) > 20:
            older_experiences = list(self.experience_buffer)[-30:-10]
            avg_older_reward = np.mean([exp.reward for exp in older_experiences])
            reward_improvement = avg_recent_reward - avg_older_reward
        else:
            reward_improvement = 0
        
        has_positive_reward = any(exp.reward > 0 for exp in recent_experiences)
        best_reward_improved = self.best_reward > 0
        
        if self.best_reward > 3 or (avg_recent_reward > 1 and reward_improvement > 0.2):
            self.emotional_state = EmotionalState.POSITIVE
        elif avg_recent_reward > 0.8 or (has_positive_reward and reward_improvement > 0.1):
            self.emotional_state = EmotionalState.EXCITED
        elif avg_recent_reward < -0.5 or (not has_positive_reward and len(self.experience_buffer) > 50):
            self.emotional_state = EmotionalState.FRUSTRATED
        elif avg_recent_reward < 0:
            self.emotional_state = EmotionalState.NEGATIVE
        else:
            if best_reward_improved:
                self.emotional_state = EmotionalState.EXCITED
            else:
                self.emotional_state = EmotionalState.NEUTRAL
    
    def _update_learning_style(self):
        self.style_switch_counter += 1
        
        has_found_strategy = self.best_reward > 0
        is_learning_progressing = False
        
        if len(self.experience_buffer) > 30:
            recent_rewards = [exp.reward for exp in list(self.experience_buffer)[-20:]]
            older_rewards = [exp.reward for exp in list(self.experience_buffer)[-40:-20]]
            avg_recent = np.mean(recent_rewards)
            avg_older = np.mean(older_rewards)
            is_learning_progressing = avg_recent > avg_older + 0.1
        
        if has_found_strategy or is_learning_progressing:
            self.style_switch_threshold = 100
        else:
            self.style_switch_threshold = 30
        
        if self.style_switch_counter >= self.style_switch_threshold:
            self.style_switch_counter = 0
            
            if has_found_strategy:
                if self.current_learning_style in ['analytical', 'pragmatic']:
                    pass
                else:
                    self.current_learning_style = random.choice(['analytical', 'pragmatic'])
            else:
                self.current_learning_style = random.choice(self.learning_styles)
            
            self.cognitive_flexibility = np.random.uniform(0.9, 1.1)
    
    def _should_use_random_action(self, state: np.ndarray = None):
        base_probability = 0.15
        
        if self.emotional_state == EmotionalState.FRUSTRATED:
            base_probability = 0.4
        elif self.emotional_state == EmotionalState.CURIOUS:
            base_probability = 0.3
        elif self.emotional_state == EmotionalState.SURPRISED:
            base_probability = 0.35
        elif self.emotional_state == EmotionalState.POSITIVE:
            base_probability = 0.1
        
        if self.best_reward > 0:
            base_probability *= 0.7
        
        if len(self.experience_buffer) > 50:
            recent_rewards = [exp.reward for exp in list(self.experience_buffer)[-20:]]
            avg_recent_reward = np.mean(recent_rewards)
            if avg_recent_reward > 0.5:
                base_probability *= 0.5
            elif avg_recent_reward > 0:
                base_probability *= 0.7
        
        if self.current_learning_style == 'creative':
            base_probability *= 1.3
        elif self.current_learning_style == 'analytical':
            base_probability *= 0.7
        
        # 基于状态访问计数的探索调整
        if state is not None:
            state_key = self._get_state_key(state)
            visit_count = self.state_visit_counts.get(state_key, 0)
            
            # 访问次数越少，探索概率越高
            exploration_bonus = 1.0 / (1 + visit_count * 0.1)
            base_probability *= (1 + exploration_bonus * 0.5)
        
        # 基于学习进度的调整
        learning_progress = self._calculate_learning_progress()
        base_probability *= (1 - learning_progress * 0.5)
        
        base_probability *= self.cognitive_flexibility
        base_probability = max(base_probability, 0.05)
        base_probability = min(base_probability, 0.6)
        
        return np.random.random() < base_probability
    
    def _get_state_key(self, state: np.ndarray) -> tuple:
        """将状态转换为可哈希的键"""
        return tuple(np.round(state, 2))
    
    def _get_state_action_key(self, state: np.ndarray, action: np.ndarray) -> tuple:
        """将状态-动作对转换为可哈希的键"""
        state_key = self._get_state_key(state)
        action_key = tuple(np.round(action, 2))
        return (state_key, action_key)
    
    def _update_visit_counts(self, state: np.ndarray, action: np.ndarray):
        """更新状态和状态-动作对的访问计数"""
        state_key = self._get_state_key(state)
        state_action_key = self._get_state_action_key(state, action)
        
        self.state_visit_counts[state_key] += 1
        self.state_action_visit_counts[state_action_key] += 1
        
        # 更新唯一状态计数
        self.stats['unique_states_visited'] = len(self.state_visit_counts)
        
        # 应用计数衰减
        for key in list(self.state_visit_counts.keys()):
            self.state_visit_counts[key] *= self.count_decay
        
        for key in list(self.state_action_visit_counts.keys()):
            self.state_action_visit_counts[key] *= self.count_decay
    
    def _calculate_learning_progress(self) -> float:
        """计算学习进度"""
        if len(self.experience_buffer) < 50:
            return 0.0
        
        recent_rewards = [exp.reward for exp in list(self.experience_buffer)[-30:]]
        older_rewards = [exp.reward for exp in list(self.experience_buffer)[-60:-30]] if len(self.experience_buffer) >= 60 else recent_rewards
        
        avg_recent = np.mean(recent_rewards)
        avg_older = np.mean(older_rewards)
        
        if avg_older == 0:
            return 0.0
        
        progress = min((avg_recent - avg_older) / abs(avg_older), 1.0)
        return max(progress, 0.0)
    
    def _calculate_exploration_bonus(self, state: np.ndarray, action: np.ndarray) -> float:
        """计算基于计数的探索奖励"""
        state_key = self._get_state_key(state)
        state_action_key = self._get_state_action_key(state, action)
        
        state_count = self.state_visit_counts.get(state_key, 0) + 1
        state_action_count = self.state_action_visit_counts.get(state_action_key, 0) + 1
        
        # 基于状态和状态-动作对的访问计数计算探索奖励
        state_bonus = 1.0 / np.sqrt(state_count)
        state_action_bonus = 1.0 / np.sqrt(state_action_count)
        
        total_bonus = (state_bonus + state_action_bonus) * self.exploration_bonus_weight
        
        # 更新统计信息
        self.stats['exploration_bonus'] = (
            self.stats['exploration_bonus'] * 0.99 + total_bonus * 0.01
        )
        
        return total_bonus
    
    def _calculate_cognitive_load(self) -> float:
        base_load = 0.3
        
        if self.exploration_phase == ExplorationPhase.EARLY:
            base_load += 0.2
        
        if self.meta_monitor.current_state == CognitiveState.CONFUSED:
            base_load += 0.3
        
        if self.emotional_state == EmotionalState.FRUSTRATED:
            base_load += 0.2
        
        base_load += np.random.uniform(-0.1, 0.1)
        
        return min(1.0, max(0.0, base_load))
    
    def _get_associative_strength(self, state: np.ndarray) -> np.ndarray:
        strengths = np.random.rand(self.state_dim) * 0.5
        
        if hasattr(self.intention_generator, 'semantic_network'):
            semantic_strengths = self.intention_generator.semantic_network.get_associative_strengths(
                [f"state_dim_{i}_{state[i]:.2f}" for i in range(len(state))]
            )
            if semantic_strengths:
                avg_strength = np.mean(list(semantic_strengths.values()))
                strengths = strengths * 0.5 + avg_strength * 0.5
        
        return strengths