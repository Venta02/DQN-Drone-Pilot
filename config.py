"""
Konfigurasi Hyperparameter untuk Drone RL dengan DQN
=====================================================
PERBAIKAN v6 - OPTIMIZED:
- Jarak aman lebih realistis untuk kecepatan 5 m/s
- Hyperparameter dioptimasi untuk learning lebih cepat
- Reward structure yang lebih balanced
"""

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
ENV_CONFIG = {
    # Drone settings
    "drone_name": "Drone1",
    "initial_position": [0, 0, -3.5],  # x, y, z (sesuai flight_height)
    
    # Movement settings - SMOOTH CONTINUOUS MOVEMENT
    "forward_speed": 5.0,   # m/s
    "duration": 0.3,        # detik - interval antar step
    
    # =========================================================================
    # OBSTACLE DETECTION - DIPERBAIKI untuk lebih realistis
    # =========================================================================
    # Dengan kecepatan 5 m/s:
    # - 1 step = 5.0 * 0.3 = 1.5m movement
    # - Perlu ~2-3 steps untuk stop completely
    # - Safety margin minimal = 3-4.5m
    
    "safe_distance": 4.0,       # Jarak IDEAL untuk berhenti (meter)
                                 # Memberikan ~0.8 detik reaction time
    
    "too_far_distance": 7.0,    # Jarak ACCEPTABLE untuk berhenti (meter)
                                 # Masih dianggap berhasil tapi kurang optimal
    
    "collision_distance": 1.0,  # Jarak dianggap tabrakan (meter)
                                 # Buffer lebih besar dari 0.5m sebelumnya
    
    # Episode settings
    "max_steps_per_episode": 500,
}

# =============================================================================
# REWARD CONFIGURATION - DIPERBAIKI untuk learning lebih cepat
# =============================================================================
REWARD_CONFIG = {
    # Success rewards - graduated untuk encourage precision
    "stop_near_obstacle": 100.0,      # Perfect stop (dalam safe_distance)
                                       # Dinaikkan dari 50 → 100 untuk signal lebih kuat
    
    "partial_success": 50.0,          # Good stop (antara safe dan too_far)
                                       # Reward baru untuk partial success
    
    # Penalties
    "collision_penalty": -100.0,      # Tabrakan - penalty besar
                                       # Dinaikkan dari -30 → -100
    
    "stop_too_far_penalty": -10.0,    # Berhenti terlalu jauh (> too_far_distance)
    
    "stop_no_obstacle_penalty": -20.0, # Berhenti tanpa obstacle terdeteksi
    
    # Per-step rewards/penalties
    "time_penalty": -0.1,             # Penalty kecil per step (encourage efficiency)
    
    "approaching_reward": 2.0,        # Reward saat mendekat ke obstacle dengan benar
                                       # BARU: encourage forward movement
    
    "danger_zone_penalty": -5.0,      # Penalty jika terlalu dekat tapi masih maju
                                       # BARU: discourage risky behavior
}

# =============================================================================
# DQN HYPERPARAMETERS - DIOPTIMASI untuk learning lebih cepat
# =============================================================================
DQN_CONFIG = {
    # Network architecture - sedikit lebih besar untuk capacity
    "hidden_layers": [256, 256, 128],  # Diperbesar dari [128, 128, 64]
    "activation": "relu",
    
    # Learning parameters - OPTIMIZED
    "learning_rate": 0.0005,    # Diturunkan dari 0.001 untuk stability
                                 # Tapi tidak terlalu kecil agar tetap cepat
    
    "gamma": 0.99,              # Discount factor (tetap)
    
    # Exploration - DIPERCEPAT decay
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.997,     # Dipercepat dari 0.995
                                 # Akan mencapai ~0.05 di episode ~1000
    
    # Replay buffer - DIPERBESAR
    "buffer_size": 200000,      # Diperbesar dari 100000
                                 # Lebih banyak experience untuk belajar
    
    "batch_size": 128,          # Diperbesar dari 64
                                 # Batch lebih besar = gradient lebih stable
    
    # Target network - LEBIH SERING update
    "target_update_freq": 5,    # Diperkecil dari 10
                                 # Update target lebih sering = learning lebih cepat
    
    # Training
    "num_episodes": 1000,
    "save_freq": 50,            # Save lebih sering (dari 100)
}

# =============================================================================
# STATE SPACE CONFIGURATION
# =============================================================================
STATE_CONFIG = {
    "state_size": 3,
    "normalize_state": True,
    "max_distance": 50.0,       # Dikurangi dari 100.0 (sesuai sensor max)
    "max_velocity": 10.0,
}

# =============================================================================
# ACTION SPACE CONFIGURATION
# =============================================================================
ACTION_CONFIG = {
    "action_size": 2,
    "actions": {
        0: "STOP",
        1: "MOVE_FORWARD"
    }
}

# =============================================================================
# FILE PATHS
# =============================================================================
PATHS = {
    "model_save_dir": "./models/",
    "log_dir": "./logs/",
    "checkpoint_dir": "./checkpoints/",
}


# =============================================================================
# TRAINING TIPS (untuk referensi)
# =============================================================================
"""
EXPECTED LEARNING CURVE dengan konfigurasi ini:

Episode 1-100:    Random exploration, reward sangat negatif
Episode 100-300:  Mulai belajar stop, masih banyak collision
Episode 300-500:  Collision berkurang, mulai stop di jarak tepat
Episode 500-800:  Konsisten stop dengan benar
Episode 800-1000: Fine-tuning, optimasi timing

JIKA LEARNING LAMBAT:
1. Cek apakah obstacle terdeteksi dengan benar (jalankan diagnose.py)
2. Kurangi epsilon_decay menjadi 0.998 untuk exploration lebih lama
3. Naikkan batch_size ke 256 jika GPU memory cukup

JIKA TERLALU BANYAK COLLISION:
1. Naikkan collision_penalty ke -150 atau -200
2. Naikkan safe_distance ke 5.0m
3. Kurangi forward_speed ke 3.0-4.0 m/s
"""
