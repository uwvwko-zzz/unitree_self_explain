# 检测model结果？

import os
import sys

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

import isaacgym
import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain import Terrain

# ========== Monkey Patch: 修复 Terrain.selected_terrain ==========
def fixed_selected_terrain(self):
    from isaacgym import terrain_utils
    terrain_type = self.cfg.terrain_kwargs.pop('type')
    if not hasattr(terrain_utils, terrain_type):
        raise ValueError(f"Terrain type '{terrain_type}' not found in isaacgym.terrain_utils")
    terrain_func = getattr(terrain_utils, terrain_type)

    for k in range(self.cfg.num_sub_terrains):
        (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        terrain_func(terrain, **self.cfg.terrain_kwargs)
        self.add_terrain_to_map(terrain, i, j)

Terrain.selected_terrain = fixed_selected_terrain

def evaluate_model(env, policy, num_steps=1000, command=[1.0, 0.0, 0.0]):
    """
    固定指令下运行模型，返回平均 reward 和最终前进距离
    """
    env.reset()  # 重置环境，内部会设置 obs_buf, rew_buf 等
    total_reward = 0.0

    with torch.no_grad():
        for _ in range(num_steps):
            # 设定固定命令
            env.commands[:, 0] = command[0]
            env.commands[:, 1] = command[1]
            env.commands[:, 2] = command[2]

            # 获取观测
            obs = env.obs_buf.clone()

            # 推理动作
            actions = policy.actor(obs)

            # 执行一步（只更新状态，不返回 reward）
            env.step(actions)

            # 从 rew_buf 获取 reward
            rewards = env.rew_buf.clone()  # shape: [num_envs]
            total_reward += rewards.sum().item()

    avg_reward = total_reward / num_steps
    forward_distance = env.base_pos[0, 0].item()  # 假设第0个环境

    return avg_reward, forward_distance

# ========== 主评估函数 ==========
def evaluate_all_models(args, log_dir, command=[1.0, 0.0, 0.0], num_steps=1000):
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.terrain.terrain_kwargs = {
        "type": "pyramid_stairs_terrain",
        "step_width": 0.3,
        "step_height": 0.15,
        "platform_size": 2.0
    }
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.env.episode_length_s = 1e6
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True
    env_cfg.headless = True  # 自动评估，无需 GUI

    # 获取模型文件列表
    model_files = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        print(f"❌ No model files found in {log_dir}")
        return

    # 提取 step 并排序
    model_info = []
    for f in model_files:
        try:
            step = int(f.split('_')[1].split('.')[0])
            model_info.append((step, f))
        except:
            continue
    model_info.sort()

    # 创建环境（只创建一次，复用）
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    print(f"🔍 Found {len(model_info)} models. Evaluating with command: {command}")

    results = []

    for step, filename in model_info:
        model_path = os.path.join(log_dir, filename)

        # 创建策略网络
        from rsl_rl.modules import ActorCritic
        policy = ActorCritic(
            num_actor_obs=env_cfg.env.num_observations,
            num_critic_obs=env_cfg.env.num_observations,
            num_actions=env_cfg.env.num_actions,
            actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
            critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
            activation=train_cfg.policy.activation,
            init_noise_std=getattr(train_cfg.policy, "init_noise_std", 1.0)
        ).to(env.device)

        # 加载权重
        ckpt = torch.load(model_path, map_location=env.device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            actor_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
            policy.actor.load_state_dict(actor_dict)
        else:
            policy.load_state_dict(ckpt)

        policy.eval()

        # 评估
        avg_reward, forward_dist = evaluate_model(env, policy, num_steps=num_steps, command=command)
        print(f"[Step {step:5d}] Avg Reward: {avg_reward:8.3f} | Forward: {forward_dist:6.2f} m")
        results.append((step, avg_reward, forward_dist, filename))

        # 可选：重置环境确保干净状态
        env.reset()

    # 排序（按 reward 降序）
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*70)
    print("🏆 Top 5 Models (by Avg Reward)")
    print("="*70)
    for i, (step, r, dist, name) in enumerate(results[:5]):
        print(f"{i+1}. Step {step:5d} | {name:<15} | Reward: {r:8.3f} | Forward: {dist:6.2f} m")

    best = results[0]
    print(f"\n✅ Best Model: {best[3]} (Step {best[0]}, Reward: {best[1]:.3f})")

    # 保存结果
    with open(os.path.join(log_dir, "auto_eval_results.txt"), "w") as f:
        f.write("Step,Filename,AvgReward,ForwardDistance\n")
        for step, r, dist, name in results:
            f.write(f"{step},{name},{r:.4f},{dist:.4f}\n")

    env.gym.destroy_sim(env.sim)

# ========== 入口 ==========
if __name__ == '__main__':
    args = get_args()

    # 设置你的日志目录（包含多个 model_XXXX.pt）
    log_dir = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/isaacgym/unitree_rl_gym/logs/rough_go2/Nov24_12-53-28_"

    # 评估命令：前进 1.0 m/s
    evaluate_all_models(
        args,
        log_dir=log_dir,
        command=[1.0, 0.0, 0.0],
        num_steps=2000  # 每个模型跑 2000 步
    )