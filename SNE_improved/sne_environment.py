import numpy as np
from typing import Tuple, Dict, Any, List

class MultiModalEnvironment:
    def __init__(self, state_dim: int = 4, action_dim: int = 2, max_steps: int = 100, modalities: List[str] = None, difficulty: str = 'medium'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.target_state = np.random.uniform(-1, 1, state_dim)
        self.difficulty = difficulty
        
        if modalities is None:
            self.modalities = ['visual', 'auditory', 'tactile', 'proprioceptive', 'olfactory', 'vestibular']
        else:
            self.modalities = modalities
        
        self.modality_dims = {
            'visual': state_dim,
            'auditory': state_dim // 2,
            'tactile': state_dim // 2,
            'proprioceptive': state_dim // 3,
            'olfactory': state_dim // 4,
            'vestibular': state_dim // 3,
            'gustatory': state_dim // 4,
            'interoceptive': state_dim // 3
        }
        
        # 模态可靠性权重
        self.modality_reliability = {
            'visual': 1.0,
            'auditory': 0.8,
            'tactile': 0.9,
            'proprioceptive': 0.95,
            'olfactory': 0.7,
            'vestibular': 0.85,
            'gustatory': 0.6,
            'interoceptive': 0.75
        }
        
        # 环境动态参数
        self.environment_dynamics = {
            'visual_noise': 0.1,
            'auditory_noise': 0.2,
            'tactile_noise': 0.15,
            'proprioceptive_noise': 0.05,
            'olfactory_noise': 0.25,
            'vestibular_noise': 0.1,
            'gustatory_noise': 0.3,
            'interoceptive_noise': 0.1,
            'target_moving_speed': 0.01,
            'modality_drift_rate': 0.005,
            'context_change_rate': 0.02
        }
        
        # 根据难度调整参数
        self._adjust_parameters_by_difficulty()
        
        self.contextual_factors = {
            'lighting': np.random.uniform(0.5, 1.0),
            'background_noise': np.random.uniform(0, 0.5),
            'temperature': np.random.uniform(0.7, 1.3),
            'fatigue': 0.0,
            'humidity': np.random.uniform(0.4, 0.8),
            'air_quality': np.random.uniform(0.6, 1.0),
            'distraction_level': np.random.uniform(0, 0.3),
            'task_complexity': 1.0
        }
        
        # 模态融合参数
        self.fusion_parameters = {
            'attention_temperature': 1.0,
            'reliability_weight': 0.7,
            'consistency_weight': 0.3
        }
    
    def _adjust_parameters_by_difficulty(self):
        """根据难度调整环境参数"""
        difficulty_multipliers = {
            'easy': 0.5,
            'medium': 1.0,
            'hard': 2.0,
            'expert': 3.0
        }
        
        multiplier = difficulty_multipliers.get(self.difficulty, 1.0)
        
        # 增加噪声
        for key in self.environment_dynamics:
            if 'noise' in key:
                self.environment_dynamics[key] *= multiplier
        
        # 增加目标移动速度
        self.environment_dynamics['target_moving_speed'] *= multiplier
        
        # 增加上下文变化率
        self.environment_dynamics['context_change_rate'] *= multiplier
        
        # 降低模态可靠性
        for modality in self.modality_reliability:
            self.modality_reliability[modality] = max(0.3, self.modality_reliability[modality] / (1 + multiplier * 0.3))
    
    def reset(self) -> Dict[str, np.ndarray]:
        self.current_step = 0
        self.target_state = np.random.uniform(-1, 1, self.state_dim)
        
        for key in self.contextual_factors:
            if key == 'lighting':
                self.contextual_factors[key] = np.random.uniform(0.5, 1.0)
            elif key == 'background_noise':
                self.contextual_factors[key] = np.random.uniform(0, 0.5)
            elif key == 'temperature':
                self.contextual_factors[key] = np.random.uniform(0.7, 1.3)
            elif key == 'fatigue':
                self.contextual_factors[key] = 0.0
            elif key == 'humidity':
                self.contextual_factors[key] = np.random.uniform(0.4, 0.8)
            elif key == 'air_quality':
                self.contextual_factors[key] = np.random.uniform(0.6, 1.0)
            elif key == 'distraction_level':
                self.contextual_factors[key] = np.random.uniform(0, 0.3)
            elif key == 'task_complexity':
                self.contextual_factors[key] = 1.0
        
        return self._generate_multi_modal_state()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.current_step += 1
        
        self._update_target_state()
        self._update_contextual_factors()
        
        multi_modal_state = self._generate_multi_modal_state()
        unified_state = self._unify_modalities(multi_modal_state)
        
        distance = np.linalg.norm(unified_state - self.target_state)
        reward = self._calculate_reward(distance, multi_modal_state)
        
        done = self.current_step >= self.max_steps
        
        info = {
            'distance': distance,
            'target_state': self.target_state.copy(),
            'step': self.current_step,
            'contextual_factors': self.contextual_factors.copy(),
            'unified_state': unified_state
        }
        
        return multi_modal_state, reward, done, info
    
    def _generate_multi_modal_state(self) -> Dict[str, np.ndarray]:
        multi_modal_state = {}
        base_state = np.random.uniform(-1, 1, self.state_dim)
        
        for modality in self.modalities:
            if modality == 'visual':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['visual_noise']
                noise *= self.contextual_factors['lighting']
                # 添加视觉漂移
                drift = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['modality_drift_rate']
                state = base_state + noise + drift
                state = np.clip(state, -1, 1)
            elif modality == 'auditory':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['auditory_noise']
                noise += self.contextual_factors['background_noise']
                noise *= (1 + self.contextual_factors['distraction_level'])
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'tactile':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['tactile_noise']
                noise *= self.contextual_factors['temperature']
                noise *= (1 + self.contextual_factors['humidity'] * 0.5)
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'proprioceptive':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['proprioceptive_noise']
                noise *= (1 + self.contextual_factors['fatigue'])
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'olfactory':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['olfactory_noise']
                noise *= (1 - self.contextual_factors['air_quality'] * 0.3)
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'vestibular':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['vestibular_noise']
                # 前庭系统对运动敏感
                motion_factor = np.linalg.norm(base_state) * 0.1
                noise *= (1 + motion_factor)
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'gustatory':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['gustatory_noise']
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            elif modality == 'interoceptive':
                noise = np.random.randn(self.modality_dims[modality]) * self.environment_dynamics['interoceptive_noise']
                # 内感受系统受疲劳影响
                noise *= (1 + self.contextual_factors['fatigue'] * 0.5)
                state = base_state[:self.modality_dims[modality]] + noise
                state = np.clip(state, -1, 1)
            else:
                noise = np.random.randn(self.modality_dims.get(modality, self.state_dim)) * 0.2
                state = base_state[:self.modality_dims.get(modality, self.state_dim)] + noise
                state = np.clip(state, -1, 1)
            
            # 应用模态可靠性
            reliability = self.modality_reliability.get(modality, 0.7)
            state = state * reliability + base_state[:len(state)] * (1 - reliability)
            
            multi_modal_state[modality] = state
        
        return multi_modal_state
    
    def _unify_modalities(self, multi_modal_state: Dict[str, np.ndarray]) -> np.ndarray:
        """基于注意力的模态融合"""
        if not multi_modal_state:
            return np.zeros(self.state_dim)
        
        # 计算各模态的注意力权重
        attention_weights = self._calculate_modality_attention(multi_modal_state)
        
        # 准备各模态的状态表示
        modality_states = []
        max_modality_dim = max(len(state) for state in multi_modal_state.values())
        
        for modality, state in multi_modal_state.items():
            # 填充到最大维度
            padded_state = np.zeros(max_modality_dim)
            padded_state[:len(state)] = state
            modality_states.append(padded_state)
        
        # 加权融合
        unified_state = np.zeros(max_modality_dim)
        for i, (modality, weight) in enumerate(attention_weights.items()):
            if modality in multi_modal_state:
                state = modality_states[i]
                unified_state += state * weight
        
        # 调整到目标维度
        if len(unified_state) > self.state_dim:
            # 降维：取前state_dim维
            unified_state = unified_state[:self.state_dim]
        elif len(unified_state) < self.state_dim:
            # 升维：补零
            unified_state = np.pad(unified_state, (0, self.state_dim - len(unified_state)), 'constant')
        
        return np.clip(unified_state, -1, 1)
    
    def _calculate_modality_attention(self, multi_modal_state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算各模态的注意力权重"""
        weights = {}
        total_weight = 0.0
        
        for modality, state in multi_modal_state.items():
            # 基于可靠性的权重
            reliability_weight = self.modality_reliability.get(modality, 0.7)
            
            # 基于状态一致性的权重（状态越稳定，权重越高）
            consistency_weight = 1.0 - np.std(state) * 0.5
            consistency_weight = max(0.1, consistency_weight)
            
            # 基于上下文的权重
            context_weight = self._calculate_context_weight(modality)
            
            # 综合权重
            weight = (
                reliability_weight * self.fusion_parameters['reliability_weight'] +
                consistency_weight * self.fusion_parameters['consistency_weight'] +
                context_weight * 0.1
            )
            
            weights[modality] = weight
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            for modality in weights:
                weights[modality] /= total_weight
        else:
            # 均匀分布
            for modality in weights:
                weights[modality] = 1.0 / len(weights)
        
        return weights
    
    def _calculate_context_weight(self, modality: str) -> float:
        """计算基于上下文的模态权重"""
        context_weight = 1.0
        
        if modality == 'visual':
            context_weight *= self.contextual_factors['lighting']
        elif modality == 'auditory':
            context_weight *= (1 - self.contextual_factors['background_noise'])
        elif modality == 'tactile':
            context_weight *= self.contextual_factors['temperature'] * 0.8 + 0.2
        elif modality == 'olfactory':
            context_weight *= self.contextual_factors['air_quality']
        elif modality == 'interoceptive':
            context_weight *= (1 - self.contextual_factors['fatigue'] * 0.5)
        
        return max(0.1, context_weight)
    
    def _calculate_reward(self, distance: float, multi_modal_state: Dict[str, np.ndarray]) -> float:
        """综合考虑多种因素的奖励计算"""
        # 基础奖励：距离目标越近奖励越高
        base_reward = np.exp(-distance) * 10
        
        # 模态质量奖励
        modality_bonus = 0.0
        for modality, state in multi_modal_state.items():
            if modality == 'visual':
                clarity = 1.0 - np.std(state) * 0.5
                modality_bonus += clarity * 0.5 * self.modality_reliability.get(modality, 0.7)
            elif modality == 'auditory':
                signal_to_noise = 1.0 - self.contextual_factors['background_noise']
                modality_bonus += signal_to_noise * 0.3 * self.modality_reliability.get(modality, 0.7)
            elif modality == 'tactile':
                stability = 1.0 - np.std(state) * 0.3
                modality_bonus += stability * 0.4 * self.modality_reliability.get(modality, 0.7)
            elif modality == 'proprioceptive':
                precision = 1.0 - self.contextual_factors['fatigue']
                modality_bonus += precision * 0.6 * self.modality_reliability.get(modality, 0.7)
            elif modality == 'olfactory':
                sensitivity = self.contextual_factors['air_quality']
                modality_bonus += sensitivity * 0.2 * self.modality_reliability.get(modality, 0.7)
            elif modality == 'vestibular':
                balance = 1.0 - np.std(state) * 0.4
                modality_bonus += balance * 0.3 * self.modality_reliability.get(modality, 0.7)
            elif modality == 'interoceptive':
                awareness = 1.0 - self.contextual_factors['fatigue'] * 0.5
                modality_bonus += awareness * 0.3 * self.modality_reliability.get(modality, 0.7)
        
        # 上下文适应奖励
        context_adaptation_bonus = 0.0
        context_adaptation_bonus += (1 - self.contextual_factors['distraction_level']) * 0.5
        context_adaptation_bonus += self.contextual_factors['air_quality'] * 0.2
        context_adaptation_bonus += (1 - self.contextual_factors['fatigue']) * 0.3
        
        # 模态一致性奖励
        consistency_bonus = self._calculate_modality_consistency(multi_modal_state) * 0.5
        
        # 综合奖励
        reward = base_reward * (
            1 + modality_bonus * 0.1 +
            context_adaptation_bonus * 0.1 +
            consistency_bonus * 0.1
        )
        
        # 探索奖励：访问新状态
        exploration_bonus = self._calculate_exploration_bonus(multi_modal_state)
        reward += exploration_bonus
        
        # 成功奖励：接近目标
        if distance < 0.2:
            reward += 5.0
            # 额外的成功奖励基于模态质量
            reward += modality_bonus * 2.0
        elif distance < 0.5:
            reward += 2.0
        
        # 难度调整
        difficulty_multiplier = {
            'easy': 1.0,
            'medium': 1.2,
            'hard': 1.5,
            'expert': 2.0
        }.get(self.difficulty, 1.0)
        reward *= difficulty_multiplier
        
        # 随机奖励波动
        if np.random.random() < 0.1:
            reward *= 1.5
        
        return reward
    
    def _calculate_modality_consistency(self, multi_modal_state: Dict[str, np.ndarray]) -> float:
        """计算模态之间的一致性"""
        if len(multi_modal_state) < 2:
            return 1.0
        
        # 计算各模态状态的相似性
        modalities = list(multi_modal_state.keys())
        consistency = 0.0
        pairs = 0
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                state1 = multi_modal_state[mod1]
                state2 = multi_modal_state[mod2]
                
                # 计算相似性（使用余弦相似度）
                min_len = min(len(state1), len(state2))
                if min_len > 0:
                    vec1 = state1[:min_len]
                    vec2 = state2[:min_len]
                    
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 0 and norm2 > 0:
                        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                        consistency += (cosine_sim + 1) / 2  # 归一化到[0, 1]
                        pairs += 1
        
        return consistency / pairs if pairs > 0 else 1.0
    
    def _calculate_exploration_bonus(self, multi_modal_state: Dict[str, np.ndarray]) -> float:
        """计算探索奖励"""
        # 基于模态多样性的探索奖励
        if not multi_modal_state:
            return 0.0
        
        # 计算模态之间的差异性
        modalities = list(multi_modal_state.keys())
        diversity = 0.0
        pairs = 0
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                state1 = multi_modal_state[mod1]
                state2 = multi_modal_state[mod2]
                
                min_len = min(len(state1), len(state2))
                if min_len > 0:
                    vec1 = state1[:min_len]
                    vec2 = state2[:min_len]
                    
                    # 计算欧氏距离
                    distance = np.linalg.norm(vec1 - vec2)
                    diversity += distance
                    pairs += 1
        
        if pairs > 0:
            avg_diversity = diversity / pairs
            # 归一化多样性奖励
            return min(avg_diversity * 0.5, 1.0)
        else:
            return 0.0
    
    def _update_target_state(self):
        movement = np.random.randn(self.state_dim) * self.environment_dynamics['target_moving_speed']
        self.target_state = self.target_state + movement
        self.target_state = np.clip(self.target_state, -1, 1)
    
    def _update_contextual_factors(self):
        for key in self.contextual_factors:
            if key == 'lighting':
                self.contextual_factors[key] += np.random.uniform(-0.05, 0.05)
                self.contextual_factors[key] = np.clip(self.contextual_factors[key], 0.1, 1.5)
            elif key == 'background_noise':
                self.contextual_factors[key] += np.random.uniform(-0.02, 0.02)
                self.contextual_factors[key] = np.clip(self.contextual_factors[key], 0, 1.0)
            elif key == 'temperature':
                self.contextual_factors[key] += np.random.uniform(-0.01, 0.01)
                self.contextual_factors[key] = np.clip(self.contextual_factors[key], 0.5, 1.5)
            elif key == 'fatigue':
                self.contextual_factors[key] += 0.01
                self.contextual_factors[key] = min(self.contextual_factors[key], 1.0)
    
    def get_state_dim(self) -> int:
        return self.state_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def get_modalities(self) -> List[str]:
        return self.modalities
    
    def get_modality_dim(self, modality: str) -> int:
        return self.modality_dims.get(modality, self.state_dim)
    
    def get_unified_state_dim(self) -> int:
        return self.state_dim