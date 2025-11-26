# 多障碍环境

import os
import sys

# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

#  python legged_gym/scripts/play_many_terrain_1ro.py --task=go2

import isaacgym
import torch
import numpy as np
from pynput import keyboard

# 导入 legged_gym 模块
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.terrain import Terrain

# ========== 自定义地形函数（gap 和 pit）==========
def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

# ========== Monkey Patch: 让 selected_terrain 生成多种地形（每种一次） ==========
def multi_terrain_selected(self):
    from isaacgym import terrain_utils
    terrain_configs = [
        (terrain_utils.pyramid_sloped_terrain, {"slope": 0.3, "platform_size": 2.0}),
        (terrain_utils.pyramid_sloped_terrain, {"slope": -0.3, "platform_size": 2.0}),
        (terrain_utils.pyramid_stairs_terrain, {"step_width": 0.3, "step_height": 0.12, "platform_size": 2.0}),
        (terrain_utils.discrete_obstacles_terrain, {
            "max_height": 0.12,
            "min_size": 1.0,       
            "max_size": 2.0,        
            "num_rects": 20,        
            "platform_size": 3.0
        }),
        (terrain_utils.stepping_stones_terrain, {
            "stone_size": 1.2,
            "stone_distance": 0.1,
            "max_height": 0.0,
            "platform_size": 4.0
        }),
        (gap_terrain, {"gap_size": 0.4, "platform_size": 2.0}),
        (pit_terrain, {"depth": 0.4, "platform_size": 2.0}),
    ]

    total_sub = self.cfg.num_rows * self.cfg.num_cols
    num_types = len(terrain_configs)

    if total_sub < num_types:
        terrain_configs = terrain_configs[:total_sub]
        print(f"⚠️  Only {total_sub} sub-terrains, using first {total_sub} terrain types.")
    elif total_sub > num_types:
        # 填充平坦地形（保持原 height_field_raw=0）
        while len(terrain_configs) < total_sub:
            terrain_configs.append((lambda t, **kw: None, {}))

    for k in range(len(terrain_configs)):
        (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
        func, kwargs = terrain_configs[k]

        sub_terrain = terrain_utils.SubTerrain(
            "sub_terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,  # 修正：使用 length_per_env_pixels
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )

        func(sub_terrain, **kwargs)
        self.add_terrain_to_map(sub_terrain, i, j)

# 应用 monkey patch
Terrain.selected_terrain = multi_terrain_selected

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
    # 注意：不再设置 terrain_kwargs，因为 monkey patch 已接管
    env_cfg.terrain.num_rows = 3  # 至少 3x3 = 9 >= 7 种地形
    env_cfg.terrain.num_cols = 3
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

    policy_path = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/isaacgym/unitree_rl_gym/logs/rough_go2/Nov25_22-44-31_/model_0.pt"
    #policy_path = "/home/extra/zhy/桌面/IsaacGym_Preview_4_Package/isaacgym/unitree_rl_gym/logs/rough_go2/Nov18_17-12-01_/model_1500.pt"
    print(f"Loading policy from: {policy_path}")
    ckpt = torch.load(policy_path, map_location=env.device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        actor_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
        policy.actor.load_state_dict(actor_dict)
    else:
        policy.load_state_dict(ckpt)

    policy.eval()

    # --- 4. 键盘监听 ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n=== Keyboard Control (Multi-Terrain) ===")
    print("W/S: forward/backward\nA/D: strafe\nQ/E: rotate\nR: reset\nESC: quit\n")
    print("Terrain grid: 3x3 (slope+, slope-, stairs, obstacles, stones, gap, pit, flat, flat)")

    try:
        obs = env.get_observations()
        while not exit_flag:
            env.commands[0, 0] = vx_cmd
            env.commands[0, 1] = vy_cmd
            env.commands[0, 2] = wz_cmd

            with torch.no_grad():
                actions = policy.actor(obs)

            # 执行一步，可能触发自动 reset
            obs, _, _, _, _ = env.step(actions)

            # 🔥 关键：禁止自动 reset！
            env.reset_buf[:] = False  # 覆盖掉摔倒、超时等导致的 reset

            # 仅响应手动复位
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