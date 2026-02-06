from collections import defaultdict, deque
import numpy as np
from typing import List, Dict, Optional, Any
from sne_basics import SemanticNode, LongTermMemory, EmotionalState, ContextFrame

class SemanticNetwork:
    def __init__(self, decay_rate: float = 0.95, activation_threshold: float = 0.1, max_hierarchy_level: int = 5):
        self.nodes = defaultdict(SemanticNode)
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.association_strengths = defaultdict(float)
        self.long_term_memory = []
        self.max_hierarchy_level = max_hierarchy_level
        
        # 分层语义结构
        self.hierarchy_levels = defaultdict(int)  # 概念 -> 层级
        self.hierarchy_structure = defaultdict(list)  # 层级 -> 概念列表
        
        # 联想链管理
        self.association_chains = []  # 存储多步联想链
        self.chain_strengths = defaultdict(float)  # 联想链强度
    
    def add_concept(self, concept: str, initial_activation: float = 0.0, hierarchy_level: int = 0):
        if concept not in self.nodes:
            self.nodes[concept] = SemanticNode(concept=concept, activation=initial_activation)
            
        # 设置层级
        if concept not in self.hierarchy_levels or hierarchy_level > self.hierarchy_levels[concept]:
            self.hierarchy_levels[concept] = hierarchy_level
            self._update_hierarchy_structure(concept, hierarchy_level)
    
    def _update_hierarchy_structure(self, concept: str, new_level: int):
        # 从旧层级移除
        old_level = self.hierarchy_levels.get(concept, -1)
        if old_level >= 0 and concept in self.hierarchy_structure[old_level]:
            self.hierarchy_structure[old_level].remove(concept)
        
        # 添加到新层级
        self.hierarchy_levels[concept] = new_level
        self.hierarchy_structure[new_level].append(concept)
    
    def add_association(self, concept1: str, concept2: str, strength: float, hierarchical: bool = False):
        self.add_concept(concept1)
        self.add_concept(concept2)
        
        self.nodes[concept1].neighbors[concept2] = strength
        self.nodes[concept2].neighbors[concept1] = strength
        
        key = tuple(sorted([concept1, concept2]))
        self.association_strengths[key] = strength
        
        # 如果是层级关联，调整层级
        if hierarchical:
            level1 = self.hierarchy_levels.get(concept1, 0)
            level2 = self.hierarchy_levels.get(concept2, 0)
            if abs(level1 - level2) > 1:
                # 调整为相邻层级
                if level1 > level2:
                    self._update_hierarchy_structure(concept2, level1 - 1)
                else:
                    self._update_hierarchy_structure(concept1, level2 - 1)
    
    def activate_concept(self, concept: str, activation: float):
        if concept in self.nodes:
            self.nodes[concept].activation = max(self.nodes[concept].activation, activation)
    
    def spread_activation(self, steps: int = 5, max_activations: int = 20, consider_hierarchy: bool = True):
        activated_nodes = []
        for concept, node in self.nodes.items():
            if node.activation > self.activation_threshold:
                activated_nodes.append((concept, node.activation))
        
        for _ in range(steps):
            new_activations = {}
            for concept, activation in activated_nodes:
                if concept not in self.nodes:
                    continue
                
                node = self.nodes[concept]
                for neighbor, strength in node.neighbors.items():
                    # 考虑层级影响
                    if consider_hierarchy:
                        level1 = self.hierarchy_levels.get(concept, 0)
                        level2 = self.hierarchy_levels.get(neighbor, 0)
                        level_factor = 1.0 / (1 + abs(level1 - level2))
                    else:
                        level_factor = 1.0
                    
                    spread_activation = activation * strength * 0.5 * level_factor
                    if spread_activation > self.activation_threshold:
                        if neighbor not in new_activations or spread_activation > new_activations[neighbor]:
                            new_activations[neighbor] = spread_activation
            
            for concept, activation in new_activations.items():
                self.activate_concept(concept, activation)
            
            activated_nodes = list(new_activations.items())[:max_activations]
        
        self.decay_activations()
    
    def generate_association_chain(self, start_concept: str, max_length: int = 10, max_chains: int = 5) -> List[List[str]]:
        """生成多步联想链"""
        if start_concept not in self.nodes:
            return []
        
        chains = []
        visited = set()
        queue = deque([[start_concept]])
        
        while queue and len(chains) < max_chains:
            current_chain = queue.popleft()
            last_concept = current_chain[-1]
            
            if len(current_chain) >= max_length:
                chains.append(current_chain)
                continue
            
            if last_concept not in self.nodes:
                continue
            
            # 按照关联强度排序邻居
            neighbors = sorted(
                self.nodes[last_concept].neighbors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # 只考虑前5个最强关联
            
            for neighbor, strength in neighbors:
                if neighbor not in visited and neighbor not in current_chain:
                    new_chain = current_chain + [neighbor]
                    queue.append(new_chain)
                    visited.add(neighbor)
        
        # 评估和存储联想链
        for chain in chains:
            self._evaluate_and_store_chain(chain)
        
        return chains
    
    def _evaluate_and_store_chain(self, chain: List[str]):
        """评估并存储联想链"""
        if len(chain) < 2:
            return
        
        # 计算链强度
        chain_strength = 0.0
        for i in range(len(chain) - 1):
            concept1, concept2 = chain[i], chain[i+1]
            key = tuple(sorted([concept1, concept2]))
            chain_strength += self.association_strengths.get(key, 0.0)
        
        chain_strength /= (len(chain) - 1)  # 平均强度
        
        # 存储联想链
        chain_tuple = tuple(chain)
        self.association_chains.append(chain)
        self.chain_strengths[chain_tuple] = chain_strength
    
    def get_strongest_association_chains(self, max_chains: int = 5) -> List[List[str]]:
        """获取最强的联想链"""
        # 按照强度排序
        sorted_chains = sorted(
            [(chain, strength) for chain, strength in self.chain_strengths.items()],
            key=lambda x: x[1],
            reverse=True
        )[:max_chains]
        
        return [list(chain) for chain, _ in sorted_chains]
    
    def get_hierarchy_level(self, concept: str) -> int:
        """获取概念的层级"""
        return self.hierarchy_levels.get(concept, 0)
    
    def get_concepts_by_level(self, level: int) -> List[str]:
        """获取指定层级的概念"""
        return self.hierarchy_structure.get(level, [])
    
    def decay_activations(self):
        for concept, node in self.nodes.items():
            node.activation *= self.decay_rate
            if node.activation < self.activation_threshold:
                node.activation = 0.0
    
    def get_active_concepts(self, threshold: float = 0.2) -> List[str]:
        active_concepts = []
        for concept, node in self.nodes.items():
            if node.activation > threshold:
                active_concepts.append(concept)
        return active_concepts
    
    def calculate_similarity(self, concept1: str, concept2: str) -> float:
        if concept1 not in self.nodes or concept2 not in self.nodes:
            return 0.0
        
        key = tuple(sorted([concept1, concept2]))
        if key in self.association_strengths:
            return self.association_strengths[key]
        
        node1 = self.nodes[concept1]
        node2 = self.nodes[concept2]
        
        common_neighbors = set(node1.neighbors.keys()) & set(node2.neighbors.keys())
        if not common_neighbors:
            return 0.0
        
        similarity = 0.0
        for neighbor in common_neighbors:
            strength1 = node1.neighbors.get(neighbor, 0.0)
            strength2 = node2.neighbors.get(neighbor, 0.0)
            similarity += min(strength1, strength2)
        
        return similarity / len(common_neighbors)
    
    def store_in_long_term_memory(self, content: Any, importance: float, timestamp: int, 
                                associations: List[str] = None, 
                                emotional_tag: EmotionalState = EmotionalState.NEUTRAL, 
                                context: Dict[str, Any] = None):
        # 动态重要性评估
        dynamic_importance = self._calculate_dynamic_importance(
            content, importance, associations, emotional_tag, context
        )
        
        memory = LongTermMemory(
            content=content,
            importance=dynamic_importance,
            timestamp=timestamp,
            associations=associations,
            emotional_tag=emotional_tag,
            context=context
        )
        self.long_term_memory.append(memory)
        
        # 限制长期记忆大小
        max_memory_size = 1000
        if len(self.long_term_memory) > max_memory_size:
            # 移除重要性最低的记忆
            self.long_term_memory.sort(key=lambda x: x.importance)
            self.long_term_memory = self.long_term_memory[-max_memory_size:]
        
        if associations and len(associations) >= 2:
            for i in range(len(associations)):
                for j in range(i + 1, len(associations)):
                    self.add_association(associations[i], associations[j], dynamic_importance * 0.5)
    
    def _calculate_dynamic_importance(self, content: Any, base_importance: float, 
                                     associations: List[str], emotional_tag: EmotionalState, 
                                     context: Dict[str, Any]) -> float:
        """动态计算记忆的重要性"""
        importance = base_importance
        
        # 情感加权
        emotional_weights = {
            EmotionalState.POSITIVE: 1.5,
            EmotionalState.NEGATIVE: 1.3,
            EmotionalState.NEUTRAL: 1.0,
            EmotionalState.SURPRISED: 1.8,
            EmotionalState.CURIOSITY: 1.6
        }
        importance *= emotional_weights.get(emotional_tag, 1.0)
        
        # 关联丰富度加权
        if associations:
            # 关联数量加权
            association_bonus = min(len(associations) * 0.1, 1.0)
            importance *= (1.0 + association_bonus)
            
            # 关联强度加权
            avg_association_strength = 0.0
            for i in range(len(associations)):
                for j in range(i + 1, len(associations)):
                    key = tuple(sorted([associations[i], associations[j]]))
                    avg_association_strength += self.association_strengths.get(key, 0.0)
            
            if len(associations) >= 2:
                avg_strength = avg_association_strength / (len(associations) * (len(associations) - 1) / 2)
                importance *= (1.0 + avg_strength * 0.5)
        
        # 上下文丰富度加权
        if context and len(context) > 0:
            context_bonus = min(len(context) * 0.05, 0.5)
            importance *= (1.0 + context_bonus)
        
        return min(importance, 5.0)  # 上限为5.0
    
    def retrieve_from_long_term_memory(self, cue: str, context: ContextFrame = None, 
                                     max_results: int = 5, include_association_chains: bool = True) -> List[LongTermMemory]:
        scores = []
        for memory in self.long_term_memory:
            score = 0.0
            
            # 1. 关联相似度评分
            if memory.associations:
                for association in memory.associations:
                    similarity = self.calculate_similarity(cue, association)
                    score += similarity
            
            # 2. 上下文匹配评分
            if context and memory.context:
                context_score = self._calculate_context_similarity(context, memory.context)
                score += context_score * 0.3  # 上下文权重
            
            # 3. 重要性加权
            score *= memory.importance
            
            # 4. 时间衰减
            current_time = int(np.random.rand() * 1000000)  # 模拟当前时间
            time_diff = abs(current_time - memory.timestamp)
            time_decay = np.exp(-time_diff / 100000.0)  # 指数衰减
            score *= time_decay
            
            # 5. 情感匹配评分
            if hasattr(context, 'emotional_state'):
                emotional_match = self._calculate_emotional_match(
                    context.emotional_state, memory.emotional_tag
                )
                score += emotional_match * 0.2
            
            # 6. 联想链增强
            if include_association_chains and memory.associations:
                chain_bonus = self._calculate_chain_bonus(memory.associations)
                score += chain_bonus * 0.1
            
            if score > 0:
                scores.append((score, memory))
        
        # 排序并返回结果
        scores.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scores[:max_results]]
    
    def _calculate_context_similarity(self, query_context: ContextFrame, memory_context: Dict[str, Any]) -> float:
        """计算上下文相似度"""
        if not memory_context:
            return 0.0
        
        match_count = 0
        total_keys = len(memory_context)
        
        for key, value in memory_context.items():
            if hasattr(query_context, key):
                query_value = getattr(query_context, key)
                if str(query_value) == str(value):
                    match_count += 1
        
        return match_count / total_keys if total_keys > 0 else 0.0
    
    def _calculate_emotional_match(self, query_emotion: EmotionalState, memory_emotion: EmotionalState) -> float:
        """计算情感匹配度"""
        if query_emotion == memory_emotion:
            return 1.0
        
        # 情感相似性映射
        emotion_similarity = {
            EmotionalState.POSITIVE: [EmotionalState.POSITIVE, EmotionalState.SURPRISED],
            EmotionalState.NEGATIVE: [EmotionalState.NEGATIVE],
            EmotionalState.NEUTRAL: [EmotionalState.NEUTRAL, EmotionalState.CURIOSITY],
            EmotionalState.SURPRISED: [EmotionalState.SURPRISED, EmotionalState.POSITIVE, EmotionalState.CURIOSITY],
            EmotionalState.CURIOSITY: [EmotionalState.CURIOSITY, EmotionalState.SURPRISED, EmotionalState.NEUTRAL]
        }
        
        similar_emotions = emotion_similarity.get(query_emotion, [])
        return 0.5 if memory_emotion in similar_emotions else 0.0
    
    def _calculate_chain_bonus(self, associations: List[str]) -> float:
        """计算联想链奖励"""
        if len(associations) < 2:
            return 0.0
        
        # 检查是否存在包含这些关联的强联想链
        bonus = 0.0
        for chain, strength in self.chain_strengths.items():
            chain_set = set(chain)
            association_set = set(associations)
            
            # 计算重叠度
            overlap = len(chain_set & association_set)
            if overlap >= 2:
                bonus += strength * (overlap / len(association_set))
        
        return min(bonus, 1.0)
    
    def update_memory_importance(self, memory_index: int, new_importance: float):
        """更新记忆的重要性"""
        if 0 <= memory_index < len(self.long_term_memory):
            self.long_term_memory[memory_index].importance = new_importance
    
    def forget_unimportant_memories(self, importance_threshold: float = 0.1, max_retention: int = 1000):
        """遗忘不重要的记忆"""
        # 按重要性排序
        self.long_term_memory.sort(key=lambda x: x.importance, reverse=True)
        
        # 保留重要记忆
        important_memories = [
            memory for memory in self.long_term_memory 
            if memory.importance >= importance_threshold
        ]
        
        # 限制数量
        self.long_term_memory = important_memories[:max_retention]
    
    def get_memory_statistics(self):
        """获取记忆统计信息"""
        if not self.long_term_memory:
            return {}
        
        importances = [memory.importance for memory in self.long_term_memory]
        emotional_distribution = {}
        for memory in self.long_term_memory:
            emotional_distribution[memory.emotional_tag] = emotional_distribution.get(memory.emotional_tag, 0) + 1
        
        return {
            'total_memories': len(self.long_term_memory),
            'average_importance': np.mean(importances),
            'max_importance': max(importances),
            'min_importance': min(importances),
            'emotional_distribution': emotional_distribution
        }
    
    def get_associative_strengths(self, concepts: List[str]) -> Dict[str, float]:
        strengths = {}
        for concept in concepts:
            if concept in self.nodes:
                total_strength = 0.0
                for neighbor, strength in self.nodes[concept].neighbors.items():
                    total_strength += strength
                strengths[concept] = total_strength / len(self.nodes[concept].neighbors) if self.nodes[concept].neighbors else 0.0
        return strengths
    
    def update_associations_from_experience(self, state: np.ndarray, action: np.ndarray, 
                                          reward: float, context: Dict[str, Any] = None):
        state_concepts = [f"state_dim_{i}_{state[i]:.2f}" for i in range(len(state))]
        action_concepts = [f"action_dim_{i}_{action[i]:.2f}" for i in range(len(action))]
        
        for concept in state_concepts + action_concepts:
            self.add_concept(concept)
        
        for i, state_concept in enumerate(state_concepts):
            for j, action_concept in enumerate(action_concepts):
                strength = abs(reward) * 0.1 if reward != 0 else 0.01
                self.add_association(state_concept, action_concept, strength)
        
        if reward > 0:
            emotional_tag = EmotionalState.POSITIVE
        elif reward < 0:
            emotional_tag = EmotionalState.NEGATIVE
        else:
            emotional_tag = EmotionalState.NEUTRAL
        
        self.store_in_long_term_memory(
            content={"state": state, "action": action, "reward": reward},
            importance=abs(reward) if reward != 0 else 0.1,
            timestamp=int(np.random.rand() * 1000000),
            associations=state_concepts + action_concepts,
            emotional_tag=emotional_tag,
            context=context
        )
    
    def get_semantic_map(self) -> Dict[str, List[str]]:
        semantic_map = {}
        for concept, node in self.nodes.items():
            neighbors = list(node.neighbors.keys())
            if neighbors:
                semantic_map[concept] = neighbors
        return semantic_map
    
    def clear_activations(self):
        for concept, node in self.nodes.items():
            node.activation = 0.0
    
    def prune_network(self, threshold: float = 0.05):
        concepts_to_remove = []
        for concept, node in self.nodes.items():
            if node.activation == 0.0 and len(node.neighbors) == 0:
                concepts_to_remove.append(concept)
        
        for concept in concepts_to_remove:
            del self.nodes[concept]
        
        associations_to_remove = []
        for key, strength in self.association_strengths.items():
            if strength < threshold:
                associations_to_remove.append(key)
        
        for key in associations_to_remove:
            del self.association_strengths[key]