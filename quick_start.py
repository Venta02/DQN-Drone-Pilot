#!/usr/bin/env python3
"""
Quick Start Script
==================
Script sederhana untuk langsung menjalankan training dan testing
tanpa perlu command line arguments

Jalankan dengan: python quick_start.py
"""

import os
import sys

# Tambahkan current directory ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone_env import DroneObstacleEnvSimulated
from dqn_agent import DQNAgent
from config import DQN_CONFIG, PATHS
import numpy as np


def quick_train(num_episodes=100):
    """
    Quick training untuk testing
    
    Args:
        num_episodes: Jumlah episode (default 100 untuk quick test)
    """
    print("\n" + "="*50)
    print("   QUICK START - DRONE RL TRAINING")
    print("="*50)
    print(f"Episodes: {num_episodes}")
    print("Mode: Simulated (tanpa AirSim)")
    print("="*50 + "\n")
    
    # Create directories
    os.makedirs(PATHS["model_save_dir"], exist_ok=True)
    os.makedirs(PATHS["log_dir"], exist_ok=True)
    
    # Initialize
    env = DroneObstacleEnvSimulated()
    agent = DQNAgent()
    
    # Training tracking
    rewards_history = []
    best_reward = float('-inf')
    
    try:
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated):
                # Select action
                action = agent.select_action(state, training=True)
                
                # Execute
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store & train
                agent.store_experience(state, action, reward, next_state, done)
                agent.train_step()
                
                state = next_state
                episode_reward += reward
                step += 1
            
            # End episode
            agent.end_episode()
            rewards_history.append(episode_reward)
            
            # Track best
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(os.path.join(PATHS["model_save_dir"], "best_model.pth"))
            
            # Print progress setiap 10 episode
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(f"Episode {episode+1:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg(10): {avg_reward:7.2f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Result: {info.get('reason', 'unknown')}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Training dihentikan oleh user")
    
    finally:
        env.close()
        
        # Save final model
        final_path = os.path.join(PATHS["model_save_dir"], "final_model.pth")
        agent.save(final_path)
        
        print("\n" + "="*50)
        print("   TRAINING SELESAI!")
        print("="*50)
        print(f"Total Episodes: {len(rewards_history)}")
        print(f"Best Reward   : {best_reward:.2f}")
        print(f"Final Epsilon : {agent.epsilon:.4f}")
        print(f"Model saved   : {final_path}")
        print("="*50)
    
    return agent


def quick_test(model_path=None, num_episodes=5):
    """
    Quick testing untuk melihat hasil training
    
    Args:
        model_path: Path ke model (default: best_model.pth)
        num_episodes: Jumlah episode test
    """
    if model_path is None:
        model_path = os.path.join(PATHS["model_save_dir"], "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        print("[INFO] Jalankan training dulu dengan quick_train()")
        return
    
    print("\n" + "="*50)
    print("   QUICK TEST - DRONE RL")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("="*50 + "\n")
    
    env = DroneObstacleEnvSimulated()
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.0  # Pure exploitation
    
    results = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        step = 0
        
        print(f"\n--- Episode {episode+1} ---")
        
        while not (done or truncated):
            # Get Q-values untuk analisis
            q_values = agent.get_q_values(state)
            
            # Select action
            action = agent.select_action(state, training=False)
            action_name = "STOP" if action == 0 else "FORWARD"
            
            # Execute
            next_state, reward, done, truncated, info = env.step(action)
            
            print(f"  Step {step+1}: {action_name} | "
                  f"Q: [{q_values[0]:.2f}, {q_values[1]:.2f}] | "
                  f"Reward: {reward:.2f} | "
                  f"Distance: {info.get('distance', 'N/A')}")
            
            state = next_state
            episode_reward += reward
            step += 1
        
        results.append({
            'reward': episode_reward,
            'steps': step,
            'outcome': info.get('reason', 'unknown')
        })
        
        outcome_emoji = {'success_stop': '✅', 'collision': '💥', 
                        'stop_too_far': '⚠️'}.get(info.get('reason'), '❓')
        print(f"\n  Result: {outcome_emoji} {info.get('reason', 'unknown')}")
        print(f"  Total Reward: {episode_reward:.2f}")
    
    env.close()
    
    # Summary
    print("\n" + "="*50)
    print("   TEST SUMMARY")
    print("="*50)
    success = sum(1 for r in results if r['outcome'] == 'success_stop')
    avg_reward = np.mean([r['reward'] for r in results])
    print(f"Success Rate : {success}/{num_episodes} ({success/num_episodes*100:.1f}%)")
    print(f"Average Reward: {avg_reward:.2f}")
    print("="*50)


def main():
    """Main function - menampilkan menu"""
    print("\n" + "="*50)
    print("   🚁 DRONE RL - OBSTACLE AVOIDANCE")
    print("="*50)
    print("\nPilih mode:")
    print("  1. Quick Train (100 episodes)")
    print("  2. Full Train (1000 episodes)")
    print("  3. Test Model")
    print("  4. Exit")
    print()
    
    try:
        choice = input("Pilihan (1-4): ").strip()
        
        if choice == '1':
            quick_train(num_episodes=100)
        elif choice == '2':
            quick_train(num_episodes=1000)
        elif choice == '3':
            quick_test()
        elif choice == '4':
            print("Goodbye! 👋")
        else:
            print("Pilihan tidak valid")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")


if __name__ == "__main__":
    main()
