# Drone RL with AirSim
# 基於 AirSim 的無人機強化學習

> Reinforcement learning agent that teaches a drone to fly forward and brake safely in front of obstacles, trained in Microsoft AirSim with Deep Q-Network.
>
> 強化學習代理，訓練無人機在 Microsoft AirSim 環境中向前飛行並在障礙物前安全煞車，使用深度 Q 網路（DQN）。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![AirSim](https://img.shields.io/badge/AirSim-Microsoft-red.svg)](https://github.com/Microsoft/AirSim)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Demo / 演示

<p align="center">
  <img src="docs/demo.gif" alt="Drone RL Demo" width="800"/>
</p>

> Trained DQN agent navigating forward and braking before hitting an obstacle in AirSim Neighborhood environment.
>
> 已訓練的 DQN 代理在 AirSim Neighborhood 環境中向前飛行並在撞擊障礙物前煞車。

---

## Overview / 專案概覽

This project trains a drone to **navigate forward and brake safely in front of obstacles** using Deep Q-Network (DQN) reinforcement learning. The agent learns from sensor data (front distance sensor) and outputs binary actions: `STOP` or `MOVE_FORWARD`.

本專案使用深度 Q 網路（DQN）強化學習訓練無人機**向前飛行並在障礙物前安全煞車**。代理從感測器資料（前方距離感測器）學習，輸出二元動作：`STOP` 或 `MOVE_FORWARD`。

| Aspect / 項目 | Detail / 細節 |
|---------------|---------------|
| Algorithm / 演算法 | DQN (Double DQN supported) / DQN（支援 Double DQN） |
| Environment / 環境 | Microsoft AirSim (Multirotor) |
| Action Space / 動作空間 | `STOP` (0), `MOVE_FORWARD` (1) |
| State Space / 狀態空間 | `[distance, velocity, is_moving]` |
| Sensor / 感測器 | Front distance sensor (50m max) / 前方距離感測器（最大 50 公尺） |
| Flight Height / 飛行高度 | ~3.5 m above ground / 離地約 3.5 公尺 |

---

## Architecture / 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                  AirSim Simulator                       │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────┐ │
│  │  Drone   │───▶│ Distance Sensor  │───▶│  State   │ │
│  └──────────┘    └──────────────────┘    └────┬─────┘ │
└──────────────────────────────────────────────┼────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────┐
│                   DQN Agent (PyTorch)                   │
│  ┌──────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │ Q-Network    │   │ Target Net  │   │ Replay Buf  │ │
│  │ [256,256,128]│   │ (sync 5 ep) │   │ (200K)      │ │
│  └──────────────┘   └─────────────┘   └─────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Reward Structure / 獎勵設計

| Condition / 條件 | Reward / 獎勵 |
|------------------|---------------|
| Stop within safe distance (≤4 m) / 在安全距離內停止 | **+100** |
| Partial success (4–7 m) / 部分成功 | **+50** |
| Collision / 碰撞 | **−100** |
| Stop too far (>7 m) / 停止太遠 | **−10** |
| Stop with no obstacle / 無障礙物停止 | **−20** |
| Approaching correctly / 正確接近 | **+2 per step** |
| Time penalty / 時間懲罰 | **−0.1 per step** |

---

## Project Structure / 專案結構

```
drone-rl-airsim/
├── config.py             # Hyperparameters & environment config
├── drone_env.py          # Custom Gymnasium environment
├── dqn_agent.py          # DQN & Double DQN agents
├── train.py              # Training entry point
├── test.py               # Evaluation script
├── quick_start.py        # Interactive runner
├── diagnose.py           # AirSim diagnostic utility
├── utils.py              # Logging, plotting, early stopping
├── airsim_settings.json  # AirSim configuration template
├── requirements.txt      # Python dependencies
├── docs/                 # Documentation & demo media
│   └── demo.gif          # Demo recording
├── models/               # Saved model checkpoints (gitignored)
└── logs/                 # Training logs & plots (gitignored)
```

---

## Installation / 安裝

### Prerequisites / 前置需求

- Python 3.10+
- [Microsoft AirSim](https://microsoft.github.io/AirSim/) with a packaged environment (e.g. AirSimNH)
- CUDA-capable GPU (optional, for faster training)

- Python 3.10+
- [Microsoft AirSim](https://microsoft.github.io/AirSim/) 與已封裝的環境（例如 AirSimNH）
- 支援 CUDA 的 GPU（選用，加速訓練）

### Setup / 設定

```bash
# Clone the repository
git clone https://github.com/Venta02/drone-rl-airsim.git
cd drone-rl-airsim

# Install dependencies
pip install -r requirements.txt

# Copy AirSim settings to your AirSim config directory
# Windows: %USERPROFILE%\Documents\AirSim\settings.json
# Linux/macOS: ~/Documents/AirSim/settings.json
```

> Use `airsim_settings.json` from this repo as a template.
>
> 請使用本專案的 `airsim_settings.json` 作為範本。

---

## Usage / 使用方式

### Quick Start / 快速開始

```bash
python quick_start.py
```

Interactive menu offers training (100 or 1000 episodes) and testing modes in the simulated environment (no AirSim required).

互動式選單提供訓練（100 或 1000 回合）與測試模式，使用模擬環境（不需 AirSim）。

### Train (Simulated, no AirSim) / 模擬訓練

```bash
python train.py
```

### Train with AirSim / AirSim 訓練

Start the AirSim simulator first, then:
先啟動 AirSim 模擬器，再執行：

```bash
python train.py --airsim --episodes 1000
```

### Train with Double DQN / Double DQN 訓練

```bash
python train.py --airsim --double --episodes 1000
```

### Test a trained model / 測試已訓練模型

```bash
python test.py --airsim --episodes 10
```

### Resume training / 繼續訓練

```bash
python train.py --resume models/checkpoint_ep500.pth
```

---

## Training Arguments / 訓練參數

| Argument / 參數 | Default / 預設 | Description / 說明 |
|-----------------|----------------|--------------------|
| `--airsim` | False | Use AirSim instead of simulated env / 使用 AirSim 而非模擬環境 |
| `--episodes` | 1000 | Number of episodes / 回合數 |
| `--resume` | None | Checkpoint to resume from / 從檢查點繼續 |
| `--double` | False | Use Double DQN / 使用 Double DQN |
| `--quiet` | False | Reduce log verbosity / 減少日誌輸出 |

---

## Diagnostic Tool / 診斷工具

If training behaves unexpectedly, run the diagnostic to verify AirSim connection, sensor readings, and drone movement:

如果訓練行為異常，執行診斷工具以驗證 AirSim 連線、感測器讀數與無人機動作：

```bash
python diagnose.py
```

---

## Key Hyperparameters / 主要超參數

Edit `config.py` to tune:
編輯 `config.py` 進行調整：

| Parameter / 參數 | Default / 預設 |
|------------------|----------------|
| `learning_rate` | 0.0005 |
| `gamma` (discount) | 0.99 |
| `epsilon_decay` | 0.997 |
| `buffer_size` | 200,000 |
| `batch_size` | 128 |
| `hidden_layers` | [256, 256, 128] |
| `target_update_freq` | every 5 episodes |

---

## Outputs / 輸出

After training you will find:
訓練後將產生：

- `models/best_model.pth` — Best-performing model / 表現最佳模型
- `models/final_model.pth` — Final episode model / 最終回合模型
- `models/checkpoint_epN.pth` — Periodic checkpoints / 定期檢查點
- `logs/training_log_*.csv` — Per-episode metrics / 各回合指標
- `logs/training_progress.png` — Reward, loss, epsilon curves / 獎勵、損失、epsilon 曲線
- `logs/outcome_distribution.png` — Outcome breakdown / 結果分布

---

## Troubleshooting / 疑難排解

### Cannot connect to AirSim / 無法連線到 AirSim
Make sure the simulator is running and `settings.json` matches the template provided.
確保模擬器正在執行，且 `settings.json` 與本專案範本一致。

### Distance sensor not detected / 偵測不到距離感測器
The sensor must be named `DistanceSensorFront` in `settings.json`. Run `diagnose.py` to verify.
感測器名稱在 `settings.json` 中必須是 `DistanceSensorFront`。執行 `diagnose.py` 驗證。

### CUDA out of memory / CUDA 記憶體不足
Reduce `batch_size` in `config.py` or force CPU training in `dqn_agent.py`.
在 `config.py` 中減少 `batch_size`，或在 `dqn_agent.py` 中強制使用 CPU。

### Drone doesn't move / 無人機不移動
Run `diagnose.py` to verify takeoff and movement work in your environment. Make sure the AirSim window is focused.
執行 `diagnose.py` 驗證起飛與移動功能。確保 AirSim 視窗為焦點。

---

## Roadmap / 路線圖

- [x] DQN agent with replay buffer & target network
- [x] Custom Gymnasium environment for AirSim
- [x] Double DQN support
- [x] Simulated environment (no AirSim required)
- [x] Training logger & plotting
- [ ] PPO / SAC agents
- [ ] LiDAR & camera-based perception
- [ ] Multi-obstacle and dynamic environments
- [ ] Sim-to-real transfer experiments

---

## Authors / 作者

This project was developed jointly by:
本專案由以下兩位共同開發：

- **Embun Ventani** — [@Venta02](https://github.com/Venta02)
- **Luqman Arif Bin Mohamad** — [@arifmachinelearning](https://github.com/arifmachinelearning)

Equal contributors / 同等貢獻者。

---

## License / 授權

MIT License — see [LICENSE](LICENSE) for details.

MIT 授權 — 詳見 [LICENSE](LICENSE)。

---

## Acknowledgments / 致謝

- [Microsoft AirSim](https://github.com/Microsoft/AirSim) — Drone simulation platform / 無人機模擬平台
- [PyTorch](https://pytorch.org/) — Deep learning framework / 深度學習框架
- [Gymnasium](https://gymnasium.farama.org/) — RL environment API / 強化學習環境介面
- The DQN paper: *Playing Atari with Deep Reinforcement Learning* (Mnih et al., 2013)

---

<p align="center">
  <strong>Teaching drones to think before they crash.</strong><br/>
  <strong>讓無人機在撞上之前學會思考。</strong>
</p>
