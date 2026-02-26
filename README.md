## Vision-based Racing

This repository contains the official implementation of a **vision-based quadrotor racing** system built on top of a custom drone simulator.  
It accompanies an **ICRA 2026** paper (vision-based racing with depth perception and reinforcement learning).

### Paper & demo video

- **Conference**: ICRA 2026  
- **Demo video**: [`https://youtu.be/_u7rpTWxA-I`](https://youtu.be/_u7rpTWxA-I)

### Repository structure

- `envs/`: racing and waypoint environments (e.g. `demo3_ellipse_onboard.RacingEnv`).
- `utils/`: algorithms (PPO, SHAC, BPTT), policy networks, data utilities.
- `configs/`: quadrotor dynamics and controller parameters (e.g. `example_offboard.json`).
- `examples/ete_racing_sim/`: end-to-end racing training and evaluation scripts.
- `examples/ete_racing_real/`: scripts for processing and replaying real-world flight data (sim-to-real, bag processing).

---

### Installation

1. Install dependencies using the provided conda environment:

```bash
conda env create -f environment.yml
conda activate vision-racing
```

2. Make sure you have a working GPU + CUDA setup if you want to train large policies.

---

### Quick start (English) – train a racing policy in simulation

1. Prepare the simulator dataset and scene configs under `datasets/spy_datasets/configs/`  
   (see `examples/ete_racing_sim/racing_demo.py` for expected scene paths).

2. Run a demo training script, for example:

```bash
python examples/ete_racing_sim/racing_demo.py --train --comment demo1_straight
```

This will:
- Instantiate a racing environment (e.g. `envs.demo1_straight.RacingEnv2`).
- Train a PPO policy with multi-modal observations (state, depth, gate index).
- Save logs and checkpoints under the corresponding `saved/` folder inside `examples/ete_racing_sim/`.

3. To evaluate a trained policy, run the same script with `--train 0` and `--weight` pointing to a saved model name:

```bash
python examples/ete_racing_sim/racing_demo.py --train 0 --weight <saved_model_name>
```

---

### 快速上手（中文）——在仿真中训练端到端赛车策略

- **环境配置**
  - 安装并激活环境（同上）：

    ```bash
    conda env create -f environment.yml
    conda activate vision-racing
    ```

  - 准备场景数据：将仿真场景放在 `datasets/spy_datasets/configs/` 下，  
    可以参考 `examples/ete_racing_sim/racing_demo.py` 中的 `scene_path` 设置。

- **开始训练**

  在项目根目录执行：

  ```bash
  python examples/ete_racing_sim/racing_demo.py --train --comment demo1_straight
  ```

  这一步会：
  - 创建对应的赛车环境（如 `envs.demo1_straight.RacingEnv2`）；
  - 使用 PPO 在深度图 + 状态信息的多模态观测上进行训练；
  - 将模型权重和 TensorBoard 日志保存到 `examples/ete_racing_sim/saved/` 目录。

- **加载已训练权重进行评估 / 可视化**

  ```bash
  python examples/ete_racing_sim/racing_demo.py --train 0 --weight <已保存模型名称>
  ```

  其中 `<已保存模型名称>` 为 `saved/` 下面的文件名（不需要加扩展名）。

---

### Real-world logs and sim-to-real

Scripts in `examples/ete_racing_real/` show how to:
- Convert ROS bag data to depth images.
- Fit dynamics models from real flight logs.
- Deploy trained policies in HITL/real flight scenarios.

These scripts are optional and are not required for pure simulation training.

---

### Notes on repository size

To keep the GitHub repository lightweight:
- Large data files such as `.bag`, `.zip`, `.onnx`, `.pt`, images and videos are ignored via `.gitignore`.
- Example folders `examples/diff_baseline/`, `examples/multi_task_IL/`, `examples/sim_to_sim/`, and `examples/VLA_task/` are kept only locally and not tracked by Git.

If you need these assets, please generate them locally or contact the maintainer.