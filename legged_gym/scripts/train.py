import os
import numpy as np
from datetime import datetime
import sys
import isaacgym
import torch

# Add project root to Python path to ensure correct module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

"""
 legged_gym.envs：包含各种机器人环境。
 get_args：解析命令行参数（如 --task, --headless 等）。
 task_registry：一个注册表，根据任务名称（args.task）动态创建对应的环境和训练配置。
"""

# python legged_gym/scripts/train.py --task=go2 --headless


from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # env：实际的 Gym 环境实例，可在其中 step/reset/获取观测与奖励。  
    # env_cfg：该环境的配置（如机器人参数、地形、奖励权重等）
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # 创建 PPO（Proximal Policy Optimization）强化学习训练器。
    # ppo_runner 封装了策略网络、价值网络、优化器、经验回放、日志记录等。
    # train_cfg 包含训练超参（如学习率、批大小、更新频率、最大迭代次数等）。
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    # 开始训练循环，总迭代次数由配置决定（也可通过 --max_iterations 覆盖）
    
if __name__ == '__main__':
    args = get_args()
    train(args)


""" 
--task: 必选参数，值可选(go2, g1, h1, h1_2)
--headless: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
--resume: 从日志中选择 checkpoint 继续训练
--experiment_name: 运行/加载的 experiment 名称
--run_name: 运行/加载的 run 名称
--load_run: 加载运行的名称，默认加载最后一次运行
--checkpoint: checkpoint 编号，默认加载最新一次文件
--num_envs: 并行训练的环境个数
--seed: 随机种子
--max_iterations: 训练的最大迭代次数
--sim_device: 仿真计算设备，指定 CPU 为 --sim_device=cpu
--rl_device: 强化学习计算设备，指定 CPU 为 --rl_device=cpu 
"""


# /home/extra/zhy/桌面/IsaacGym_Preview_4_Package/isaacgym/unitree_rl_gym/logs/rough_go2/Nov18_17-12-01_/model_1500.pt 