from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


# 部署代码入口
# 这段代码就是“把训练好的策略安全地运行在真实 Unitree 机器人上”的部署控制器。
# 它完成了通信初始化、状态读取、观测构建、策略推理、命令映射与下发、以及若干初始化与安全态流程

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        # 初始化远程控制器
        self.remote_controller = RemoteController()

        # Initialize the policy network
        # 加载训练好的策略网络
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        # 初始化过程变量
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        # 动作
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        # 目标关节位置
        self.target_dof_pos = config.default_angles.copy()
        # 观测
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # 指令，速度，角速度
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        # 根据不同的消息类型设置下发底层指令和接收底层状态的函数、话题等
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            # 初始化底层指令和状态消息
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            # 初始化电机模式
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0
            # 初始化底层指令下发的话题，函数
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        # 等待订阅者接收底层状态数据
        self.wait_for_low_state()

        # Initialize the command msg
        # 初始化底层指令消息
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        # crc 校验
        cmd.crc = CRC().Crc(cmd)
        # 发送底层指令
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        # 等待远程控制器按下启动按钮
        while self.remote_controller.button[KeyMap.start] != 1:
            # 零力矩状态
            create_zero_cmd(self.low_cmd)
            # 发送零力矩指令
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    # 进入到初始位姿，时间为2秒
    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        # 将腿部关节和胳膊和腰部关节电机的序号进行合并
        # 比如leg_joint2motor_idx=[0,1],arm_waist_joint2motor_idx=[2,3,4]
        # dof_idx=[0,1,2,3,4]
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        # 电机的P，D 参数合并
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        # 将腿部的默认初始角度和胳膊和腰部的初始角度合并
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        # 关节的数量
        dof_size = len(dof_idx)
        
        # 记录每个关节位置
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        # 移动到默认位置
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

# 控制机器人关节从当前状态至默认初始状态
    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        # 记录腿部关节的位置和速度
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        # 记录IMU传感器的四元数和角速度
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            # h1和h1_2的imu安装在躯干"torso"上，需要被转化至骨盆"pelvis"坐标系
            # 腰部的位置和速度
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        # 建立强化学习网络策略的观测值
        # 在rotation.py下面
        gravity_orientation = get_gravity_orientation(quat)
        # 电机的位置和速度
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        # 电机旋转的角度/角速度
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        # 机器人角速度
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        # 将偏移时间归一化到 [0, 1] 区间，表示当前时刻在周期中的相对位置
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        ## 底层指令
		## 机器人遥控器的左边摇杆y值代表给机器人的期望x轴线速度指令
        self.cmd[0] = self.remote_controller.ly
        ## 机器人遥控器的左边摇杆x值代表给机器人的期望x轴线速度指令
        ## 乘-1的原因，当遥控者想让机器人左侧横移，lx为负值（向左掰），但是由于机器人y轴左侧为正，所以需要取反
        self.cmd[1] = self.remote_controller.lx * -1
        ## 机器人遥控器的右边摇杆x值代表给机器人的期望z轴角速度指令
         ## 乘-1的原因，当遥控者想让机器人逆时针旋转，rx为负值（向左掰），但是由于机器人绕z轴逆时针为正，所以需要取反。
        self.cmd[2] = self.remote_controller.rx * -1


        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs[9 + num_actions * 3] = sin_phase
        self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        # 从强化学习网络策略中获取动作
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        # 下发腿部电机目标角度
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        # 下发腰部，胳膊电机目标角度
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    # 调用解析命令行参数的标准模块
    parser = argparse.ArgumentParser()
    # 启动部署脚本命令行需要设置的参数，包括机器人网卡名字以及配置文件名字
    # 可以与deploy_real下的README里提到的启动用法对应
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    # 获取参数
    args = parser.parse_args()

    # Load config
    # yaml参数文件路径 
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    # 初始化DDS通信，在unitree_sdk2.py中进行设置
    ChannelFactoryInitialize(0, args.net)

    # 初始化config类
    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    # 零扭矩状态
    controller.zero_torque_state()

    # Move to the default position
    # 各关节电机移动到默认位置
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    # 按下A键继续执行run，进入运动控制模式
    controller.default_pos_state()

    while True:
        try:
            # 收集观测值，下发底层指令
            controller.run()
            # Press the select key to exit
            # 按下select键退出
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    # 阻尼状态，在command_helper.py中定义
    create_damping_cmd(controller.low_cmd)
    # 发送底层指令
    controller.send_cmd(controller.low_cmd)
    print("Exit")
