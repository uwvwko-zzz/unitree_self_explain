# 楼梯塔


import os
import sys

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

import isaacgym
import torch
import numpy as np
from pynput import keyboard

# 导入 legged_gym 模块
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

# ========== 全局键盘状态 ==========
vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
reset_flag = False
exit_flag = False

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, reset_flag, exit_flag
    try:
        if key.char == 'w': vx_cmd = 1.5
        elif key.char == 's': vx_cmd = -1.5
        elif key.char == 'a': vy_cmd = 1.5
        elif key.char == 'd': vy_cmd = -1.5
        elif key.char == 'q': wz_cmd = -1.0
        elif key.char == 'e': wz_cmd = 1.0
        elif key.char == 'r': reset_flag = True
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag = True

def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in 'ws': vx_cmd = 0.0
        elif key.char in 'ad': vy_cmd = 0.0
        elif key.char in 'qe': wz_cmd = 0.0
    except AttributeError:
        pass

# ========== 主函数 ==========
def play(args):
    global reset_flag, exit_flag

    if args.headless:
        print("Keyboard control requires GUI. Run without --headless.")
        return

    # --- 1. 获取配置 ---
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

    # --- 2. 创建环境 ---
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])

    # --- 3. 手动创建策略网络并加载 .pt 权重 ---
    from rsl_rl.modules import ActorCritic  # 使用 rsl_rl

    policy = ActorCritic(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_observations,  # no privileged obs
        num_actions=env_cfg.env.num_actions,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        activation=train_cfg.policy.activation,
        init_noise_std=getattr(train_cfg.policy, "init_noise_std", 1.0) 
    ).to(env.device)

    #policy_path = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/isaacgym/unitree_rl_gym/logs/rough_go2/Nov23_21-33-04_/model_3000.pt"
    policy_path = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/isaacgym/unitree_rl_gym/logs/rough_go2/Nov24_12-53-28_/model_6000.pt"
    print(f"Loading policy from: {policy_path}")
    ckpt = torch.load(policy_path, map_location=env.device)

    # 加载 actor 部分
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        actor_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
        policy.actor.load_state_dict(actor_dict)
    else:
        # 如果 .pt 直接是 actor
        policy.load_state_dict(ckpt)

    policy.eval()

    # --- 4. 键盘监听 ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n=== Keyboard Control ===\nW/S: forward/backward\nA/D: strafe\nQ/E: rotate\nR: reset\nESC: quit\n")

    try:
        obs = env.get_observations()
        while not exit_flag:
            env.commands[0, 0] = vx_cmd
            env.commands[0, 1] = vy_cmd
            env.commands[0, 2] = wz_cmd

            with torch.no_grad():
                actions = policy.actor(obs) 

            obs, _, _, _, _ = env.step(actions)

            if reset_flag:
                env.reset_idx(torch.tensor([0], device=env.device))
                obs = env.get_observations()
                reset_flag = False

            env.render()

    finally:
        listener.stop()
        if hasattr(env, 'viewer') and env.viewer:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

# ========== 入口 ==========
if __name__ == '__main__':
    args = get_args()
    play(args)