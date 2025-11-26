from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

# 定义了 LeggedRobot 类，这是一个 完整的、可运行的 Isaac Gym 强化学习环境

# 继承自 BaseTask

# 对于环境，主要是这两个函数：
    # create_sim
    # _get_env_origins

# 原版的，用于大平面训练
    
class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        # 保存输入参数
        self.cfg = cfg
        # 保存lsaacgym的仿真参数
        self.sim_params = sim_params
        # 预留空的变量，用来储存周围的高度信息
        self.height_samples = None
        # 控制器是否可视化
        self.debug_viz = False
        # 初始化完成标志
        self.init_done = False
        # 解析配置
        self._parse_cfg(self.cfg)
        # 调用父类 BaseTask 初始化
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # 设置相机
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # 初始化核心缓冲区
        self._init_buffers()
        # 准备奖励函数
        self._prepare_reward_function()
        # 初始化标记完成
        self.init_done = True

    # 将策略输出的动作（actions）转化为物理仿真中的控制输入，推进仿真 decimation 步，然后计算新的状态、奖励和终止条件，最终返回 RL 算法所需的标准五元组
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 动作裁剪与设备同步，防止动作值过大
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        # 渲染当前帧（GUI模式）
        self.render()
        # 循环推进物理仿真
        # 策略频率=仿真频率/decimation
        # 物理仿真需要高频率，但是策略推理较慢
        for _ in range(self.cfg.control.decimation):
            # 计算关节力矩
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # 将力矩发送给仿真器
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # 进一步物理仿真
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            # cpu模式下同步结果
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
                # 刷新关节状态张量
            self.gym.refresh_dof_state_tensor(self.sim)
        # 仿真后处理，具体见下一个函数
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # 观测裁剪，返回
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
        # obs_buf: 普通观测（策略输入）
        # privileged_obs_buf: 特权观测（critic 输入，可为 None）
        # rew_buf: 即时奖励
        # reset_buf: 是否终止（1=终止）
        # extras: 额外信息（如 "time_outs", "episode" 统计量）
     
    # post_physics_step 是 LeggedRobot 环境中最关键的后处理函数之一，
    # 它在每次物理仿真推进之后被调用，负责更新状态、计算奖励、判断终止条件、重置环境、生成观测值等所有与强化学习训练直接相关的逻辑
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # 流程图：
            # refresh states
            #     ↓
            # update counters
            #     ↓
            # compute base states (pos, vel, gravity, etc.)
            #     ↓
            # _post_physics_step_callback() → update commands
            #     ↓
            # check_termination() → set reset_buf
            #     ↓
            # compute_reward() → fill rew_buf
            #     ↓
            # get env_ids to reset
            #     ↓
            # reset_idx(env_ids) → reset those envs
            #     ↓
            # _push_robots() (if enabled)
            #     ↓
            # compute_observations() → fill obs_buf
            #     ↓
            # update last_* buffers

        
        # 刷新仿真状态张量
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # 更新计数器
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # 准备常用状态量
        # 从root_states张量中提取基座位置和四元数
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        # 将四元数转化为欧拉角（判断是否翻滚/俯仰过大）
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        # 将世界坐标系下的速度转化为机体坐标系下的速度
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 计算重力在机体坐标系下的投影
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 调用回调函数    _post_physics_step_callback ，具体见下文  
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        # 检查终止条件，见下文
        self.check_termination()
        # 计算奖励，见下文
        self.compute_reward()
        # 获取需要重置的环境ID
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 重置指定环境
        self.reset_idx(env_ids)
        
        # 应用域随机化：随机能力，每隔一定步，给机器人一个随机的横向速度
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        # 计算观察值，见下文
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        # 更新上一时刻的缓冲区
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    # 为每个并行环境（共 num_envs 个）判断是否应该终止当前 episode，结果存入 self.reset_buf（一个布尔型张量，True 表示需重置）
    # 即使没摔倒、没翻滚，只要超时，也要重置环境。
    # 但注意：超时的 episode 仍然会被记录，只是不视为“失败”
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # 摔倒检测
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # 检测是否严重翻滚或俯仰
        # pitch（俯仰角） > 1.0 弧度（≈57°）
        # roll（翻滚角） > 0.8 弧度（≈45°）
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        # 检查是否超时
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    # 重置部分仿真环境
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # 如果ids为空，则返回
        if len(env_ids) == 0:
            return
        
        # reset robot states
        # 重置关节状态
        self._reset_dofs(env_ids)
        # 重置根状态
        self._reset_root_states(env_ids)
        # 重采样运动命令
        self._resample_commands(env_ids)

        # reset buffers
        # 重置内部缓冲区
        self.actions[env_ids] = 0.      # 动作
        self.last_actions[env_ids] = 0. 
        self.last_dof_vel[env_ids] = 0. # 关节速度
        self.feet_air_time[env_ids] = 0.    # 脚空闲时间
        self.episode_length_buf[env_ids] = 0    # 当前 episode 步数 ——> 用于判断超时
        self.reset_buf[env_ids] = 1     # 标记为”已重置“

        # 记录 episode 统计信息
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # 记录课程学习信息
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        # 发送超时信息给RL算法
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    # 四足机器人强化学习环境中的核心奖励计算逻辑
    # 计算每个环境（num_envs 个）在当前 timestep 的总奖励
    # 只计算配置中 scale ≠ 0 的奖励项
    # 支持“仅正奖励”模式
    # 特殊处理 termination 
     
    #  流程图
    #     清零 rew_buf
    #    所有启用的奖励函数（除 termination）
    #      ↓
    #   调用函数 → 加权 → 累加到 rew_buf 和 episode_sums
    #      ↓
    #   如果 only_positive_rewards=True → 截断 rew_buf ≥ 0
    #      ↓
    #    termination 奖励 → 计算摔倒惩罚 → 直接加到 rew_buf
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # 清零总奖励缓冲区
        self.rew_buf[:] = 0.
        # 遍历奖励项
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # 只有正向奖励（避免早期崩溃）
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        # 处理termination奖励（摔倒惩罚）
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    # 将机器人的各种状态信息（速度、姿态、关节、命令等）拼接成一个固定长度的向量，作为策略网络（Policy）的输入
    def compute_observations(self):
        """ Computes observations
        """
        # 归一化，参数具体在legged_robot_config.py中
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,        # 机体线速度
                                    self.base_ang_vel  * self.obs_scales.ang_vel,       # 机体角速度
                                    self.projected_gravity,                             # 重力向量
                                    self.commands[:, :3] * self.commands_scale,         # 运动命令
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    # 关节位置偏差
                                    self.dof_vel * self.obs_scales.dof_vel,             # 关节速度
                                    self.actions                                        # 上一时刻的动作
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        # 添加传感器噪声
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # 初始化物理仿真世界（simulation world），创建地面（terrain），并批量生成所有机器人实例（environments）
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self.sim_device_id,       物理仿真使用的 GPU/CPU ID（如 0）
        # self.graphics_device_id,  渲染使用的 GPU ID（如 0），headless 时为 -1
        # self.physics_engine,      物理引擎类型（通常是 gymapi.SIM_PHYSX）
        # self.sim_params           仿真参数（dt, gravity, PhysX 设置等）

        # 创建地面
        self._create_ground_plane()
        # 创建所有机器人环境
        self._create_envs()

    # 视觉
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        # legged_robot_config.py中的
        # class viewer:
            # ref_env = 0
            # po = [11., 5, 3.]  # [m]s = [10, 0, 6]  # [m]
            # lookat
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        # viewer：图形窗口句柄（在 BaseTask.__init__ 中创建）
        # env_handle：指定某个环境的相机（设为 None 表示全局相机，观察整个仿真世界）
        # eye（cam_pos）：相机位置
        # target（cam_target）：相机注视点
     

    #------------- Callbacks --------------
    #  在创建每个仿真环境时，随机化机器人各刚体的摩擦系数
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        # 检测是否开启
        if self.cfg.domain_rand.randomize_friction:
            # 只在env_id=0时计算
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            # 运用到环境的所有刚体
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        # isaacgym用这个来创建刚体
        return props

    # 在创建环境时，从 URDF 模型中读取每个关节（DOF）的物理限制（位置、速度、力矩），并根据配置计算“软限制”（soft limits），用于后续的奖励计算和安全控制
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        # 就在第一个环境
        if env_id==0:
            # 初始化关节张量
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            # 从URDF中提取关节限制
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                # 计算软限制（类似于代价地图，就是避免靠近的边界）
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    # 在创建每个仿真环境时，随机化机器人基座
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            # 添加随机质量
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    
    # 动态更新机器人的运动指令，使其既能接受线速度/角速度命令，也能接受更直观的“目标朝向”命令
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 定期重新采样命令
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        # 处理朝向命令
        if self.cfg.commands.heading_command:
            # 计算机人的朝向
            forward = quat_apply(self.base_quat, self.forward_vec)
            # 计算朝向角
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            # 计算朝向误差，转化为角速度命令
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    # 为env_ids 中的环境随机生成新的运动指令
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # 重采样X线速度
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # 重采样Y线速度
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # 朝向模式
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # 角速度模式
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        # 过滤掉小的线速度命令
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # 输入：策略网络输出的动作 actions（形状 [num_envs, num_actions]）
    # 输出：实际发送给仿真的关节力矩 torques（形状 [num_envs, num_dof]）
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        # 三种控制模式
        # 动作缩放
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        # 位置控制
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        # 速度控制
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        # 直接力矩控制
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # 力矩裁剪
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # 重置指定环境
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # 重置关节位置（带随机扰动）
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        # 关节速度为0
        self.dof_vel[env_ids] = 0.

        # 将新状态同步到仿真器
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    # 重置指定环境（env_ids）中机器人根（root）状态的位置和速度。
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        # 重置base position（基座位姿）
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # 重置base velocity（基座速度）
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        # 同步到仿真器
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # 模拟外部扰动
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        # 筛选需要扰动的环境
        env_ids = torch.arange(self.num_envs, device=self.device)
        # 当前步数正好是push_interval的倍数的环境进行扰动
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        # 若没有，直接返回0
        if len(push_env_ids) == 0:
            return
        # 设置随机的推送速度
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        # 同步到仿真器
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

   
    # 课程学习，动态扩大机器人的线速度命令范围
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # 判断是否满足条件
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # 扩大线速度命令范围（x方向）
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    # 构建一个与观测（observation）维度相同的噪声缩放向量，用于在强化学习训练中向观测值添加可控的传感器噪声，模拟真实硬件中的测量误差，从而提升策略的鲁棒性
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # compute_observations中
        noise_vec = torch.zeros_like(self.obs_buf[0])       # 创建与单个obs同维度的零向量
        self.add_noise = self.cfg.noise.add_noise           # 是否添加噪声
        noise_scales = self.cfg.noise.noise_scales          # 噪声缩放
        noise_level = self.cfg.noise.noise_level            # 噪声强度
        # 线速度噪声
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # 角速度噪声
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # 重力噪声
        noise_vec[6:9] = noise_scales.gravity * noise_level
        # 命令不添加噪声
        noise_vec[9:12] = 0. # commands
        # 关节位置噪声
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # 关节速度噪声
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # 上一时刻动作不加噪声
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    #----------------------------------------
    # 为仿真状态、控制变量、观测中间量等创建 PyTorch 张量（buffer），并将 Isaac Gym 的底层 GPU 内存张量（如 root state、dof state）“包装”成可直接用的 PyTorch 张量，以便高效访问和计算
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        # 初始化底层仿真状态张量
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        # 包装成pytroch可用张量
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        self.base_quat = self.root_states[:, 3:7]           # 四元数
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)  # 欧拉角
        self.base_pos = self.root_states[:self.num_envs, 0:3]# 位置
        
        # 检测脚部碰撞，摔倒
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
    
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # 重力向量
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # 机体前向单位向量
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        # 将世界速度转化为机体坐标系下的线速度
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # 重力在机体坐标系下的投影
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
      

        # joint positions offsets and PD gains
        # 默认关节位置与PD增益初始化
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            # 初始化PD控制器增益
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    # 动态构建奖励计算函数列表，并为每个启用的奖励项分配缓冲区
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        # 清理无效的奖励项，时间归一化
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)         # 移除无效的奖励项
            else:
                self.reward_scales[key] *= self.dt  # 将奖励缩放乘以dt
        # prepare list of functions
        self.reward_functions = []      # 奖励函数
        self.reward_names = []          # 奖励项名称
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue                # termination专门处理（compute_reward中）
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        # 初始化episode累计奖励缓冲区
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    # 在 Isaac Gym 仿真环境中添加一个无限大的平面作为地面，并根据配置文件设置其物理属性
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        # 创建一个PlaneParams对象
        plane_params = gymapi.PlaneParams()
        # 地面法向量为+Z轴
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # 物理参数
        plane_params.static_friction = self.cfg.terrain.static_friction # 静态摩擦力
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction # 动态摩擦力
        plane_params.restitution = self.cfg.terrain.restitution # 恢复系数  
        
        # 添加到仿真器中
        self.gym.add_ground(self.sim, plane_params)

    # 加载机器人模型（URDF/MJCF），批量创建多个并行仿真环境（num_envs 个），并为每个环境配置物理属性、初始状态、身体索引等
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 从配各种属性置填充
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        # 提取模型元信息
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        
        # 筛选关键身体部件
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # 准备初始状态
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # 初始化环境
        self._get_env_origins()         # 生成每个env的世界坐标偏移
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            # 随机化刚体形态属性
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 创建机器人实例
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            # 随机化关节
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            # 随机化刚体属性
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # 保存关节身体索引
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    # 为每个并行仿真环境（共 num_envs 个）分配一个在世界坐标系中的“原点”位置（env_origins[i]）
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # 不使用复杂地形
        self.custom_origins = False
        # 初始化张量，每个环境的x,y,z起始偏移
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        # 尽量放置机器人排成正方形网格
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        # 创建网格索引
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt  # 策略步长时间
        self.obs_scales = self.cfg.normalization.obs_scales         # 观测缩放因子
        self.reward_scales = class_to_dict(self.cfg.rewards.scales) # 奖励权重
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)# 命令范围
     
        # 奖励权重，命令范围
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # 转化时间间隔为仿真步数
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    #------------ reward functions----------------
    # 下面就是奖励函数了
    
    # 惩罚机器人在垂直方向（Z轴）上的线速度，鼓励机器人保持平稳运动，避免上下弹跳或剧烈颠簸
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    # 惩罚绕 X 和 Y 轴的角速度（即翻滚 Roll 和俯仰 Pitch）     
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    # 惩罚重力向量在X，Y的投影（不倾斜）    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # 惩罚机器人的高度偏离目标高度
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    # 惩罚过大的关节力矩 
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    # 惩罚关节速度过大
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    # 惩罚关节加速度过大
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    # 惩罚动作过快  
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    # 惩罚机器人非与地形的碰撞
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    # 摔倒
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    # 关节超出软限位
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # 关节速度接近最大速度
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # 力矩接近最大输出
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    # 鼓励机器人跟踪XY平面线速度指令
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    # 鼓励长步态，跑
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # 腾空时间，跳
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    # 惩罚脚撞到垂直墙面
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    # 当指令为0时，保持默认站立姿态
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # 惩罚脚接触力过大
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
