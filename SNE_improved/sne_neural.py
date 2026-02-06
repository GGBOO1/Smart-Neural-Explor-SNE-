import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 注意力机制
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.attention_softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 注意力机制
        attention_weights = self.attention(x)
        attention_weights = self.attention_softmax(attention_weights)
        x = x * attention_weights
        
        # 第三层
        x = self.fc3(x)
        x = self.bn3(x)
        
        return x

class UncertaintyEstimator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.3, num_ensemble: int = 5):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.dropout_rate = dropout_rate
        
        # 贝叶斯神经网络（使用dropout作为贝叶斯近似）
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, sample: bool = False) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        
        if sample:
            # 贝叶斯推理：多次前向传播获取不确定性
            predictions = []
            for _ in range(self.num_ensemble):
                # 启用dropout进行采样
                with torch.no_grad():
                    h1 = F.relu(self.dropout1(self.fc1(x)))
                    h2 = F.relu(self.dropout2(self.fc2(h1)))
                    pred = torch.sigmoid(self.fc3(h2))
                    predictions.append(pred)
            
            # 计算预测均值和方差
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            var_pred = predictions.var(dim=0)
            
            # 返回均值作为预测，方差作为不确定性
            return mean_pred, var_pred
        else:
            # 常规前向传播（测试模式）
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

class CuriosityModule(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32, curiosity_weight: float = 0.1, decay_factor: float = 0.999):
        super().__init__()
        self.curiosity_weight = curiosity_weight
        self.decay_factor = decay_factor
        self.current_weight = curiosity_weight
        
        # 前向模型：预测下一个状态
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 逆向模型：预测执行的动作
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 状态新颖性检测器
        self.novelty_detector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> tuple:
        # 预测下一个状态
        pred_next_state = self.forward_model(torch.cat([state, action], dim=1))
        # 预测执行的动作
        pred_action = self.inverse_model(torch.cat([state, next_state], dim=1))
        
        # 计算损失
        forward_loss = F.mse_loss(pred_next_state, next_state)
        inverse_loss = F.mse_loss(pred_action, action)
        
        # 计算内在奖励（基于预测误差）
        intrinsic_reward = self.current_weight * forward_loss
        
        # 更新好奇心权重
        self.current_weight *= self.decay_factor
        
        # 计算状态新颖性
        state_novelty = self.novelty_detector(state).mean()
        
        return forward_loss, inverse_loss, intrinsic_reward, state_novelty
    
    def reset_weight(self):
        # 重置好奇心权重
        self.current_weight = self.curiosity_weight