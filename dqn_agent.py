"""
DQN Agent untuk Drone Obstacle Avoidance
=========================================
Implementasi Deep Q-Network (DQN) dengan:
- Experience Replay Buffer
- Target Network
- Epsilon-Greedy Exploration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os

from config import DQN_CONFIG, STATE_CONFIG, ACTION_CONFIG, PATHS


# Named tuple untuk experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer
    Menyimpan pengalaman agent untuk training yang lebih stabil
    """
    
    def __init__(self, capacity):
        """
        Args:
            capacity: Ukuran maksimal buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Tambah experience ke buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Random sample dari buffer
        
        Args:
            batch_size: Jumlah sample
        
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences]))
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences]))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural Network untuk approximating Q-values
    """
    
    def __init__(self, state_size, action_size, hidden_layers):
        """
        Args:
            state_size: Dimensi state space
            action_size: Jumlah actions
            hidden_layers: List ukuran hidden layers [128, 128, 64]
        """
        super(QNetwork, self).__init__()
        
        # Build layers dynamically
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass"""
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network Agent
    """
    
    def __init__(self, state_size=None, action_size=None, device=None):
        """
        Args:
            state_size: Dimensi state (default dari config)
            action_size: Jumlah actions (default dari config)
            device: 'cuda' atau 'cpu'
        """
        # Sizes
        self.state_size = state_size or STATE_CONFIG["state_size"]
        self.action_size = action_size or ACTION_CONFIG["action_size"]
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] Using device: {self.device}")
        
        # Hyperparameters
        self.gamma = DQN_CONFIG["gamma"]
        self.learning_rate = DQN_CONFIG["learning_rate"]
        self.epsilon = DQN_CONFIG["epsilon_start"]
        self.epsilon_end = DQN_CONFIG["epsilon_end"]
        self.epsilon_decay = DQN_CONFIG["epsilon_decay"]
        self.batch_size = DQN_CONFIG["batch_size"]
        self.target_update_freq = DQN_CONFIG["target_update_freq"]
        
        # Networks
        self.q_network = QNetwork(
            self.state_size, 
            self.action_size,
            DQN_CONFIG["hidden_layers"]
        ).to(self.device)
        
        self.target_network = QNetwork(
            self.state_size,
            self.action_size,
            DQN_CONFIG["hidden_layers"]
        ).to(self.device)
        
        # Copy weights ke target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network tidak di-train langsung
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(DQN_CONFIG["buffer_size"])
        
        # Training stats
        self.training_step = 0
        self.episode_count = 0
        self.losses = []
    
    def select_action(self, state, training=True):
        """
        Pilih action menggunakan epsilon-greedy policy
        
        Args:
            state: Current state
            training: Jika True, gunakan epsilon-greedy; jika False, greedy
        
        Returns:
            int: Action yang dipilih
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: action dengan Q-value tertinggi
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Simpan experience ke replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Satu step training dengan batch dari replay buffer
        
        Returns:
            float: Loss value (atau None jika buffer belum cukup)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Hitung current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Hitung target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Hitung loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights dari Q-network ke target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        print("[INFO] Target network updated")
    
    def decay_epsilon(self):
        """Decay epsilon untuk mengurangi exploration seiring waktu"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def end_episode(self):
        """Dipanggil di akhir setiap episode"""
        self.episode_count += 1
        self.decay_epsilon()
        
        # Update target network secara periodik
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def save(self, filepath=None):
        """
        Simpan model ke file
        
        Args:
            filepath: Path untuk menyimpan model
        """
        if filepath is None:
            os.makedirs(PATHS["model_save_dir"], exist_ok=True)
            filepath = os.path.join(PATHS["model_save_dir"], f"dqn_drone_ep{self.episode_count}.pth")
        
        torch.save({
            'episode': self.episode_count,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'losses': self.losses[-1000:],  # Simpan 1000 loss terakhir
        }, filepath)
        
        print(f"[INFO] Model saved to {filepath}")
        return filepath
    
    def load(self, filepath):
        """
        Load model dari file
        
        Args:
            filepath: Path ke file model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode']
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint.get('losses', [])
        
        print(f"[INFO] Model loaded from {filepath}")
        print(f"       Episode: {self.episode_count}, Epsilon: {self.epsilon:.4f}")
    
    def get_q_values(self, state):
        """
        Dapatkan Q-values untuk semua actions dari state tertentu
        
        Args:
            state: State untuk dievaluasi
        
        Returns:
            np.array: Q-values untuk setiap action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def get_stats(self):
        """
        Dapatkan statistik training
        
        Returns:
            dict: Training statistics
        """
        stats = {
            'episode': self.episode_count,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
        }
        return stats


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent
    Menggunakan Q-network untuk memilih action dan target network untuk mengevaluasi
    Mengurangi overestimation bias
    """
    
    def train_step(self):
        """Training step dengan Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: gunakan Q-network untuk memilih action, target network untuk evaluate
        with torch.no_grad():
            # Q-network memilih best action
            next_actions = self.q_network(next_states).argmax(1)
            # Target network mengevaluasi Q-value untuk action tersebut
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss dan optimization
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        return loss.item()


if __name__ == "__main__":
    # Test agent
    print("Testing DQN Agent...")
    
    agent = DQNAgent()
    
    # Simulate some experiences
    for i in range(200):
        state = np.random.rand(3).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.rand(3).astype(np.float32)
        done = random.random() < 0.1
        
        agent.store_experience(state, action, reward, next_state, done)
        
        loss = agent.train_step()
        if loss is not None and i % 50 == 0:
            print(f"Step {i}: Loss = {loss:.4f}")
    
    # Test save/load
    agent.episode_count = 10
    filepath = agent.save("./test_model.pth")
    
    agent2 = DQNAgent()
    agent2.load(filepath)
    
    print(f"\nAgent stats: {agent2.get_stats()}")
    
    # Cleanup
    os.remove("./test_model.pth")
    print("\nTest completed!")
