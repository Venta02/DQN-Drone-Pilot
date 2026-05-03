"""
Utility Functions untuk Drone RL
=================================
Berisi fungsi-fungsi helper untuk logging, plotting, dan lainnya
"""

import numpy as np
import os
from datetime import datetime
import json


class TrainingLogger:
    """Logger untuk menyimpan training progress"""
    
    def __init__(self, log_dir):
        """
        Args:
            log_dir: Direktori untuk menyimpan log
        """
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")
        self.json_file = os.path.join(log_dir, f"training_data_{timestamp}.json")
        
        # Create CSV header
        with open(self.log_file, 'w') as f:
            f.write("episode,reward,length,outcome,epsilon,loss,timestamp\n")
        
        self.data = []
    
    def log_episode(self, episode, reward, length, outcome, epsilon, loss):
        """Log satu episode"""
        timestamp = datetime.now().isoformat()
        
        # Write to CSV
        with open(self.log_file, 'a') as f:
            f.write(f"{episode},{reward:.4f},{length},{outcome},{epsilon:.6f},{loss:.6f},{timestamp}\n")
        
        # Store data
        self.data.append({
            'episode': episode,
            'reward': reward,
            'length': length,
            'outcome': outcome,
            'epsilon': epsilon,
            'loss': loss,
            'timestamp': timestamp
        })
    
    def save_json(self):
        """Simpan semua data ke JSON"""
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=2)


def plot_training_results(history, save_dir):
    """
    Plot hasil training
    
    Args:
        history: Dictionary berisi training history
        save_dir: Direktori untuk menyimpan plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib tidak terinstall, skip plotting")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = range(1, len(history['episode_rewards']) + 1)
    
    # Create figure dengan multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Training Progress - Drone Obstacle Avoidance', fontsize=14)
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, history['episode_rewards'], 'b-', alpha=0.3, label='Raw')
    if len(history['episode_rewards']) >= 10:
        moving_avg = np.convolve(history['episode_rewards'], 
                                  np.ones(10)/10, mode='valid')
        ax1.plot(range(10, len(history['episode_rewards'])+1), moving_avg, 
                 'r-', linewidth=2, label='Moving Avg (10)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax2 = axes[0, 1]
    ax2.plot(episodes, history['episode_lengths'], 'g-', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    ax3 = axes[1, 0]
    ax3.plot(episodes, history['epsilons'], 'purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss
    ax4 = axes[1, 1]
    if history['losses']:
        ax4.plot(episodes, history['losses'], 'orange', alpha=0.5)
        if len(history['losses']) >= 10:
            loss_avg = np.convolve(history['losses'], np.ones(10)/10, mode='valid')
            ax4.plot(range(10, len(history['losses'])+1), loss_avg, 
                     'red', linewidth=2, label='Moving Avg')
            ax4.legend()
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Training plot saved to {plot_path}")
    
    # Plot outcome distribution
    if history['episode_results']:
        fig2, ax = plt.subplots(figsize=(8, 6))
        
        outcomes = history['episode_results']
        unique_outcomes = list(set(outcomes))
        counts = [outcomes.count(o) for o in unique_outcomes]
        
        colors = {
            'success_stop': 'green',
            'partial_stop': 'lightgreen',
            'collision': 'red',
            'stop_too_far': 'orange',
            'moving': 'blue',
            'unknown': 'gray'
        }
        bar_colors = [colors.get(o, 'gray') for o in unique_outcomes]
        
        ax.bar(unique_outcomes, counts, color=bar_colors)
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Count')
        ax.set_title('Episode Outcomes Distribution')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        outcome_path = os.path.join(save_dir, 'outcome_distribution.png')
        plt.savefig(outcome_path, dpi=150, bbox_inches='tight')
        plt.close()


def print_episode_summary(episode, reward, steps, info, epsilon, loss, duration):
    """
    Print ringkasan episode dengan formatting yang bagus
    
    Args:
        episode: Nomor episode
        reward: Total reward
        steps: Jumlah steps
        info: Info dictionary dari environment
        epsilon: Current epsilon
        loss: Average loss
        duration: Durasi episode dalam detik
    """
    outcome = info.get('reason', 'unknown')
    distance = info.get('distance', None)
    
    # Outcome emoji
    outcome_emoji = {
        'success_stop': '✅',
        'partial_stop': '🟡',
        'collision': '💥',
        'stop_too_far': '⚠️',
        'moving': '➡️',
        'unknown': '❓'
    }
    emoji = outcome_emoji.get(outcome, '❓')
    
    # Color codes (ANSI)
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    # Determine color based on outcome
    if outcome == 'success_stop':
        color = GREEN
    elif outcome == 'collision':
        color = RED
    else:
        color = YELLOW
    
    distance_str = f"{distance:.2f}m" if distance else "N/A"
    
    print(f"Ep {episode:4d} | {emoji} {color}{outcome:12s}{RESET} | "
          f"Reward: {reward:7.2f} | Steps: {steps:4d} | "
          f"ε: {epsilon:.3f} | Loss: {loss:.4f} | "
          f"Dist: {distance_str:6s} | Time: {duration:.2f}s")


def calculate_success_metrics(history, window=100):
    """
    Hitung metrik keberhasilan dari history
    
    Args:
        history: Training history dictionary
        window: Window size untuk moving average
    
    Returns:
        dict: Metrics dictionary
    """
    outcomes = history['episode_results']
    rewards = history['episode_rewards']
    
    total = len(outcomes)
    if total == 0:
        return {}
    
    # Overall metrics
    success_count = sum(1 for o in outcomes if o == 'success_stop')
    collision_count = sum(1 for o in outcomes if o == 'collision')
    
    metrics = {
        'total_episodes': total,
        'success_rate': success_count / total,
        'collision_rate': collision_count / total,
        'avg_reward': np.mean(rewards),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
    }
    
    # Recent metrics (last window episodes)
    if total >= window:
        recent_outcomes = outcomes[-window:]
        recent_rewards = rewards[-window:]
        
        recent_success = sum(1 for o in recent_outcomes if o == 'success_stop')
        
        metrics['recent_success_rate'] = recent_success / window
        metrics['recent_avg_reward'] = np.mean(recent_rewards)
    
    return metrics


def moving_average(data, window):
    """
    Hitung moving average
    
    Args:
        data: List atau array data
        window: Window size
    
    Returns:
        np.array: Moving average
    """
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def save_hyperparameters(save_path, config_dict):
    """
    Simpan hyperparameters ke file JSON
    
    Args:
        save_path: Path file output
        config_dict: Dictionary hyperparameters
    """
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"[INFO] Hyperparameters saved to {save_path}")


def load_hyperparameters(load_path):
    """
    Load hyperparameters dari file JSON
    
    Args:
        load_path: Path file input
    
    Returns:
        dict: Hyperparameters dictionary
    """
    with open(load_path, 'r') as f:
        return json.load(f)


class EarlyStopping:
    """
    Early stopping untuk menghentikan training jika tidak ada improvement
    """
    
    def __init__(self, patience=50, min_delta=0.01, mode='max'):
        """
        Args:
            patience: Jumlah episode tanpa improvement sebelum stop
            min_delta: Minimum improvement yang dianggap signifikan
            mode: 'max' atau 'min' (maksimasi atau minimasi)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value):
        """
        Check apakah harus early stop
        
        Args:
            value: Nilai metrik untuk dicek
        
        Returns:
            bool: True jika harus stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False


def create_experiment_dir(base_dir, experiment_name=None):
    """
    Buat direktori untuk experiment baru
    
    Args:
        base_dir: Base directory
        experiment_name: Nama experiment (optional)
    
    Returns:
        str: Path ke experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"
    
    exp_dir = os.path.join(base_dir, dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    
    print(f"[INFO] Created experiment directory: {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test moving average
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ma = moving_average(data, 3)
    print(f"Moving average: {ma}")
    
    # Test early stopping
    es = EarlyStopping(patience=3, min_delta=0.1, mode='max')
    test_values = [1.0, 1.1, 1.05, 1.05, 1.05, 1.05]
    for i, v in enumerate(test_values):
        stopped = es(v)
        print(f"Step {i}: value={v}, should_stop={stopped}")
    
    print("\nUtility tests completed!")
