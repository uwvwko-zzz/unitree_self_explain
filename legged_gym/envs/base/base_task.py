
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch


# 定义了一个 BaseTask 类，它是 基于 Isaac Gym 的强化学习（RL）仿真任务的基类（抽象基类），用于构建四足机器人或其他物理智能体的 RL 环境
# 它封装了 Isaac Gym 的底层仿真逻辑，并为具体任务提供统一接口

# 个人理解，这个就好像是一个中间件，底层是Isaac Gym，上层是RL算法（PPO）

# [ RL Algorithm (PPO) ]
#          ↑
# [ VecEnv 接口 (rsl_rl.env.VecEnv) ]
#          ↑
# [ BaseTask ] ←── 具体任务（Go2）
#          ↑
# [ Isaac Gym API (gymapi) ]
#          ↑
# [ PhysX + CUDA ]

# 把物理仿真细节 转化为 RL 友好的张量接口

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 获取gym实例
        self.gym = gymapi.acquire_gym()
        # 下面的参数在isaacgym\unitree_rl_gym\legged_gym\utils\task_registry.py中也有提到
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless        
        # 解析设备
        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # headless开无图型化       
        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        # 这四个就是供RL算法使用的属性
        # 这些参数在isaacgym\unitree_rl_gym\legged_gym\envs\base\legged_robot.py中配置
        self.num_envs = cfg.env.num_envs    # 并行环境数量
        self.num_obs = cfg.env.num_observations     # 观测维度
        self.num_privileged_obs = cfg.env.num_privileged_obs    # 特权观测
        self.num_actions = cfg.env.num_actions    # 动作维度

        # optimization flags for pytorch JIT
        # 关闭pytorch的JIT，以提高性能
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        # 分配核心缓冲区
        # self.obs_buf        # [num_envs, num_obs]      → 当前观测
        # self.rew_buf        # [num_envs]               → 当前奖励
        # self.reset_buf      # [num_envs]               → 是否需要重置（1=重置）
        # self.episode_length_buf  # [num_envs]          → 当前 episode 步数
        # self.time_out_buf   # [num_envs]               → 是否因超时终止
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        
        # self.privileged_obs_buf  # 可选，用于 teacher-student 训练
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs
            
        # self.extras         # 字典，用于返回额外信息
        self.extras = {}

        # create envs, sim and viewer
        # 创建模拟环境
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        # 初始化查看器
        # 到时候看看能不能在这里加一个WASD控制
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    # 下面这些函数是BaseTask向上层提供的接口
    # 供RL算法调用

    # 返回当前观测缓冲区（供RL算法调用）
    def get_observations(self):
        return self.obs_buf   
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    # 重置指定环境
    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    # 重置所有环境
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    # 执行动作，进入仿真
    def step(self, actions):
        raise NotImplementedError
        # 将动作发送给机器人（如设置关节目标）
        # 推进仿真（gym.simulate）
        # 计算奖励（rew_buf）
        # 更新 reset_buf 和 time_out_buf
        # 填充 obs_buf 和 privileged_obs_buf
        # 返回 (obs, privileged_obs, rew, reset, extras)

    # 渲染
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)