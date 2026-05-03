"""
Testing Script untuk Drone RL
==============================
Script ini menguji model DQN yang sudah di-train
"""

import numpy as np
import time
import os
import argparse

from drone_env import DroneObstacleEnv, DroneObstacleEnvSimulated
from dqn_agent import DQNAgent, DoubleDQNAgent
from config import ACTION_CONFIG, PATHS, ENV_CONFIG, STATE_CONFIG


STOP_DISTANCE = 4.0


def get_raw_distance(state, normalized=True):
    if normalized:
        return state[0] * STATE_CONFIG["max_distance"]
    return state[0]


def test(
    model_path,
    use_airsim=False,
    num_episodes=10,
    use_double_dqn=False,
    render=True,
    delay=0.1
):
    """
    Testing loop untuk mengevaluasi model
    
    Args:
        model_path: Path ke model yang akan ditest
        use_airsim: True untuk gunakan AirSim
        num_episodes: Jumlah episode testing
        use_double_dqn: True jika model adalah Double DQN
        render: Tampilkan visualisasi
        delay: Delay antar step (untuk visualisasi)
    
    Returns:
        dict: Test results
    """
    if use_airsim:
        print("[INFO] Menggunakan AirSim Environment")
        env = DroneObstacleEnv(render_mode="human" if render else None)
    else:
        print("[INFO] Menggunakan Simulated Environment")
        env = DroneObstacleEnvSimulated(render_mode="human" if render else None)
    
    AgentClass = DoubleDQNAgent if use_double_dqn else DQNAgent
    agent = AgentClass()
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"[INFO] Model loaded from {model_path}")
    else:
        print(f"[ERROR] Model not found: {model_path}")
        return None
    
    agent.epsilon = 0.0
    
    results = {
        'rewards': [],
        'lengths': [],
        'outcomes': [],
        'final_distances': [],
    }
    
    print("\n" + "="*60)
    print("           DRONE RL TESTING")
    print("="*60)
    print(f"Model             : {model_path}")
    print(f"Episodes          : {num_episodes}")
    print(f"Environment       : {'AirSim' if use_airsim else 'Simulated'}")
    print("="*60 + "\n")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        
        episode_reward = 0
        step = 0
        done = False
        truncated = False
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not (done or truncated):
            q_values = agent.get_q_values(state)
            raw_distance = get_raw_distance(state, STATE_CONFIG["normalize_state"])
            
            if raw_distance <= STOP_DISTANCE:
                action = 0
            else:
                action = agent.select_action(state, training=False)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            action_name = ACTION_CONFIG["actions"][action]
            distance = info.get('distance', raw_distance)
            
            print(f"  Step {step+1:3d}: Action={action_name:12s} | "
                  f"Q=[{q_values[0]:.2f}, {q_values[1]:.2f}] | "
                  f"Dist={distance:.2f}m | "
                  f"Reward={reward:.2f}")
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if delay > 0:
                time.sleep(delay)
        
        results['rewards'].append(episode_reward)
        results['lengths'].append(step)
        
        final_dist = info.get('distance', None)
        reason = info.get('reason', 'unknown')
        
        if reason in ['partial_stop', 'stop_too_early', 'success_stop'] and final_dist and final_dist <= STOP_DISTANCE:
            reason = 'success_stop'
        
        results['outcomes'].append(reason)
        results['final_distances'].append(final_dist)
        
        outcome_emoji = {
            'success_stop': '✅',
            'partial_stop': '🟡',
            'collision': '💥',
            'stop_too_early': '⚠️',
            'stop_no_obstacle': '❓',
        }.get(reason, '❓')
        
        print(f"\n  Episode {episode + 1} Summary:")
        print(f"    {outcome_emoji} Outcome       : {reason}")
        print(f"    Total Reward  : {episode_reward:.2f}")
        print(f"    Steps         : {step}")
        print(f"    Final Distance: {final_dist if final_dist else 'N/A'}")
    
    env.close()
    
    avg_reward = np.mean(results['rewards'])
    avg_length = np.mean(results['lengths'])
    success_count = sum(1 for o in results['outcomes'] if o in ['success_stop', 'partial_stop'])
    success_rate = success_count / num_episodes
    
    print("\n" + "="*60)
    print("           TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Average Reward      : {avg_reward:.2f}")
    print(f"Average Steps       : {avg_length:.2f}")
    print(f"Success Rate        : {success_rate*100:.1f}% ({success_count}/{num_episodes})")
    print(f"\nOutcome Distribution:")
    for outcome in set(results['outcomes']):
        count = results['outcomes'].count(outcome)
        emoji = {'success_stop': '✅', 'partial_stop': '🟡', 'collision': '💥'}.get(outcome, '❓')
        print(f"  {emoji} {outcome}: {count}/{num_episodes} ({count/num_episodes*100:.1f}%)")
    print("="*60)
    
    return results


def demo(model_path=None, use_airsim=False):
    """
    Demo mode: menunjukkan step-by-step decision making
    
    Args:
        model_path: Path ke model (jika None, gunakan random agent)
        use_airsim: Gunakan AirSim
    """
    if use_airsim:
        env = DroneObstacleEnv()
    else:
        env = DroneObstacleEnvSimulated()
    
    agent = DQNAgent()
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0.0
        mode = "TRAINED MODEL"
    else:
        mode = "RANDOM AGENT"
    
    print("\n" + "="*60)
    print(f"           DEMO MODE - {mode}")
    print("="*60)
    print("Press Enter to advance each step, 'q' to quit\n")
    
    state, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    while not (done or truncated):
        q_values = agent.get_q_values(state)
        raw_distance = get_raw_distance(state, STATE_CONFIG["normalize_state"])
        
        print(f"\n--- Step {step + 1} ---")
        print(f"State: [Distance: {raw_distance:.2f}m, Velocity: {state[1]:.3f}, Moving: {state[2]:.0f}]")
        print(f"Q-values: STOP={q_values[0]:.3f}, FORWARD={q_values[1]:.3f}")
        
        if raw_distance <= STOP_DISTANCE:
            action = 0
        else:
            action = agent.select_action(state, training=False)
        
        action_name = ACTION_CONFIG["actions"][action]
        print(f"Selected Action: {action_name}")
        
        user_input = input("Press Enter to continue (q to quit): ")
        if user_input.lower() == 'q':
            break
        
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.2f}")
        print(f"Info: {info}")
        
        state = next_state
        total_reward += reward
        step += 1
    
    print(f"\n--- Demo Ended ---")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Outcome: {info.get('reason', 'unknown')}")
    
    env.close()


def main():
    """Main function dengan argument parser"""
    parser = argparse.ArgumentParser(description='Test DQN Agent for Drone Obstacle Avoidance')
    
    parser.add_argument('model', type=str, nargs='?', default=None,
                        help='Path to model file')
    parser.add_argument('--airsim', action='store_true',
                        help='Use AirSim environment')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--double', action='store_true',
                        help='Model is Double DQN')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (step-by-step)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps (seconds)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    
    args = parser.parse_args()
    
    if args.model is None:
        default_path = os.path.join(PATHS["model_save_dir"], "best_model.pth")
        if os.path.exists(default_path):
            args.model = default_path
        else:
            print("[ERROR] Please specify model path or run training first")
            return
    
    if args.demo:
        demo(args.model, args.airsim)
    else:
        test(
            model_path=args.model,
            use_airsim=args.airsim,
            num_episodes=args.episodes,
            use_double_dqn=args.double,
            render=not args.no_render,
            delay=args.delay
        )


if __name__ == "__main__":
    main()
