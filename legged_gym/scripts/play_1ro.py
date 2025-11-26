# 最基础的大平面控制


import os
import sys

# asdw
# 我这个逻辑是啥：
# commands 是策略网络（Policy）的输入之一，用于告诉机器人“你想让它做什么”。
# 策略网络根据 commands + 其他状态（如速度、关节角度等） 输出 actions（关节目标偏移）。
# actions 经过 _compute_torques 转为力矩，驱动机器人运动
# 
# commands[:, 0]  目标线速度 x
# commands[:, 1]  目标线速度 y
# commands[:, 2]  目标角速度 z    heading_command=False
# commands[:, 3]  目标朝向角      heading_command=True
# 我就是通过自己模拟commands命令来实现键盘控制的


# --- 路径设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, export_policy_as_jit
import torch
from pynput import keyboard

# ========== 全局键盘状态 ==========
vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
reset_flag = False
exit_flag = False

def on_press(key):
    global vx_cmd, vy_cmd, wz_cmd, reset_flag, exit_flag
    try:
        print(f"Pressed key: {key.char}")
        if key.char == 'w':
            vx_cmd = 1.5
        elif key.char == 's':
            vx_cmd = -1.5
        elif key.char == 'a':
            vy_cmd = 1.5
        elif key.char == 'd':
            vy_cmd = -1.5
        elif key.char == 'q':
            wz_cmd = -1.0
        elif key.char == 'e':
            wz_cmd = 1.0
        elif key.char == 'r':
            reset_flag = True
    except AttributeError:
        if key == keyboard.Key.esc:
            exit_flag = True

def on_release(key):
    global vx_cmd, vy_cmd, wz_cmd
    try:
        if key.char in 'ws':
            vx_cmd = 0.0
        elif key.char in 'ad':
            vy_cmd = 0.0
        elif key.char in 'qe':
            wz_cmd = 0.0
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
    # 一个机器人
    env_cfg.env.num_envs = 1
    # 关闭地形（大平面）
    env_cfg.terrain.curriculum = False
    # 关闭观测噪声
    env_cfg.noise.add_noise = False
    # 关闭推力噪声
    env_cfg.domain_rand.push_robots = False
    # 慢动作仿真
    env_cfg.env.test = True
    # 关闭 heading_command
    # heading_command是用机器人的目标朝向，但是我们要用角速度控制，关闭他
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6  # 禁用命令重采样
    env_cfg.env.episode_length_s = 1e6      # 禁用超时


    # --- 2. 创建环境 ---
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env.set_camera([2.0, 2.0, 2.0], [0.0, 0.0, 1.0])

    # --- 3. 加载训练好的策略 ---
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    print("\n=== Keyboard Control (WASD + QE + R) ===")
    print("Focus this TERMINAL window to control!")
    print("W/S: forward/backward")
    print("A/D: strafe left/right")
    print("Q/E: turn left/right")
    print("R: reset robot")
    print("ESC: quit\n")

    # --- 4. 启动键盘监听 ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        # 获取初始观测
        obs = env.get_observations()
        while not exit_flag:
            # --- 设置运动命令---
            env.commands[0, 0] = vx_cmd  # vx
            env.commands[0, 1] = vy_cmd  # vy
            env.commands[0, 2] = wz_cmd  # wz (yaw rate)

            # --- 策略推理---
            with torch.no_grad():
                actions = policy(obs)

            # --- 仿真推进 ---
            obs, _, _, _, _ = env.step(actions)

            # --- 重置处理 ---
            if reset_flag:
                env.reset_idx(torch.tensor([0], device=env.device))
                obs = env.get_observations()  # 重置后重新获取观测
                reset_flag = False

            # --- 渲染 ---
            env.render()

    finally:
        listener.stop()
        env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

# ========== 入口 ==========
if __name__ == '__main__':
    args = get_args()
    play(args)