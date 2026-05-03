"""
Training Script untuk Drone RL
===============================
Script ini melakukan training DQN agent untuk obstacle avoidance
"""

import numpy as np
import time
import os
import argparse
from datetime import datetime

from drone_env import DroneObstacleEnv, DroneObstacleEnvSimulated
from dqn_agent import DQNAgent, DoubleDQNAgent
from config import DQN_CONFIG, PATHS
from utils import TrainingLogger, plot_training_results, print_episode_summary


def train(
    use_airsim=False,
    num_episodes=None,
    resume_from=None,
    use_double_dqn=False,
    verbose=True
):
    """
    Training loop utama
    
    Args:
        use_airsim: True untuk gunakan AirSim, False untuk simulasi
        num_episodes: Jumlah episode (default dari config)
        resume_from: Path ke model untuk melanjutkan training
        use_double_dqn: Gunakan Double DQN
        verbose: Print detailed info
    
    Returns:
        tuple: (agent, training_history)
    """
    # Setup
    num_episodes = num_episodes or DQN_CONFIG["num_episodes"]
    save_freq = DQN_CONFIG["save_freq"]
    
    # Buat direktori jika belum ada
    os.makedirs(PATHS["model_save_dir"], exist_ok=True)
    os.makedirs(PATHS["log_dir"], exist_ok=True)
    
    # Initialize environment
    if use_airsim:
        print("[INFO] Menggunakan AirSim Environment")
        print("[INFO] Pastikan AirSim simulator sudah berjalan!")
        env = DroneObstacleEnv()
    else:
        print("[INFO] Menggunakan Simulated Environment (tanpa AirSim)")
        env = DroneObstacleEnvSimulated()
    
    # Initialize agent
    AgentClass = DoubleDQNAgent if use_double_dqn else DQNAgent
    agent = AgentClass()
    
    # Resume dari checkpoint jika ada
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        start_episode = agent.episode_count
        print(f"[INFO] Melanjutkan training dari episode {start_episode}")
    else:
        start_episode = 0
    
    # Training logger
    logger = TrainingLogger(PATHS["log_dir"])
    
    # Training history
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_results': [],
        'epsilons': [],
        'losses': [],
    }
    
    # Best model tracking
    best_reward = float('-inf')
    best_model_path = None
    
    print("\n" + "="*60)
    print("           DRONE RL TRAINING - DQN")
    print("="*60)
    print(f"Total Episodes    : {num_episodes}")
    print(f"Environment       : {'AirSim' if use_airsim else 'Simulated'}")
    print(f"Agent             : {'Double DQN' if use_double_dqn else 'DQN'}")
    print(f"Start Episode     : {start_episode}")
    print(f"Save Frequency    : Every {save_freq} episodes")
    print("="*60 + "\n")
    
    # Training loop
    try:
        for episode in range(start_episode, num_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            state, info = env.reset()
            
            episode_reward = 0
            episode_loss = []
            step = 0
            done = False
            truncated = False
            
            # Episode loop
            while not (done or truncated):
                # Select action
                action = agent.select_action(state, training=True)
                
                # Execute action
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update state
                state = next_state
                episode_reward += reward
                step += 1
                
                # Verbose output untuk setiap step
                if verbose and step % 50 == 0:
                    print(f"  Step {step}: action={info['action']}, reward={reward:.2f}, "
                          f"distance={info.get('distance', 'N/A')}")
            
            # End episode
            agent.end_episode()
            
            episode_time = time.time() - episode_start_time
            avg_loss = np.mean(episode_loss) if episode_loss else 0.0
            
            # Store history
            history['episode_rewards'].append(episode_reward)
            history['episode_lengths'].append(step)
            history['episode_results'].append(info.get('reason', 'unknown'))
            history['epsilons'].append(agent.epsilon)
            history['losses'].append(avg_loss)
            
            # Log
            logger.log_episode(episode, episode_reward, step, info.get('reason', 'unknown'), 
                             agent.epsilon, avg_loss)
            
            # Print summary
            if verbose or episode % 10 == 0:
                print_episode_summary(episode, episode_reward, step, info, agent.epsilon, 
                                    avg_loss, episode_time)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_path = os.path.join(PATHS["model_save_dir"], "best_model.pth")
                agent.save(best_model_path)
            
            # Periodic save
            if (episode + 1) % save_freq == 0:
                checkpoint_path = os.path.join(
                    PATHS["model_save_dir"], 
                    f"checkpoint_ep{episode+1}.pth"
                )
                agent.save(checkpoint_path)
                
                # Plot progress
                plot_training_results(history, PATHS["log_dir"])
    
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        env.close()
        
        # Save final model
        final_path = os.path.join(PATHS["model_save_dir"], "final_model.pth")
        agent.save(final_path)
        
        # Final plots
        plot_training_results(history, PATHS["log_dir"])
        
        # Print summary
        print("\n" + "="*60)
        print("           TRAINING COMPLETED")
        print("="*60)
        print(f"Total Episodes    : {agent.episode_count}")
        print(f"Best Reward       : {best_reward:.2f}")
        print(f"Final Epsilon     : {agent.epsilon:.4f}")
        print(f"Final Model       : {final_path}")
        print(f"Best Model        : {best_model_path}")
        print("="*60)
    
    return agent, history


def main():
    """Main function dengan argument parser"""
    parser = argparse.ArgumentParser(description='Train DQN Agent for Drone Obstacle Avoidance')
    
    parser.add_argument('--airsim', action='store_true',
                        help='Use AirSim environment (default: simulated)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--double', action='store_true',
                        help='Use Double DQN')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    train(
        use_airsim=args.airsim,
        num_episodes=args.episodes,
        resume_from=args.resume,
        use_double_dqn=args.double,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
