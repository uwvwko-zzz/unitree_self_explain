from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
# 引入 legged_gym.envs.base.legged_robot_config 模块，就是在isaacgym\unitree_rl_gym\legged_gym\envs\base\legged_robot_config.py

# GO2RoughCfg：继承自 LeggedRobotCfg 
# 该类用于配置 Go2 机器人在粗糙地形上的仿真环境参数
class GO2RoughCfg( LeggedRobotCfg ):
    # 设置机器人初始位置（z = 0.42 m，略高于地面）。
    # 定义 默认关节角度（即当策略输出动作 = 0 时，机器人各关节的目标角度），用于让机器人处于一个合理的站立姿态
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    # 使用 P 控制（比例控制），即只用 stiffness（刚度）做 PD 控制中的 P 项
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad] 刚度
        damping = {'joint': 0.5}     # [N*m*s/rad] 阻尼
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25         # 策略输出的动作（通常在 [-1, 1]）会被缩放后叠加到默认角度上，作为目标关节角度
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4             # 策略更新频率， 4 个仿真步才更新一次动作

    # URDF 文件路径   
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        # 如果大腿或小腿碰到地面，会给予负奖励
        penalize_contacts_on = ["thigh", "calf"]
        # 如果机器人躯干（base）触地，立即终止 （摔了）
        terminate_after_contacts_on = ["base"]
        # 禁用自碰撞检测（提高仿真效率）
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        # 关节位置接近限位（90% 范围）时开始惩罚
        soft_dof_pos_limit = 0.9
        # 期望的躯干高度（米），用于高度相关的奖励计算
        base_height_target = 0.25
        # 奖励系数，用于调整不同奖励项的重要性
        class scales( LeggedRobotCfg.rewards.scales ):
            # 关节力矩的惩罚（节能）
            torques = -0.0002
            # 关节接近限位的惩罚（重罚）
            dof_pos_limits = -10.0

# 配置 PPO（Proximal Policy Optimization）强化学习算法的超参数和训练设置
class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # 策略熵，鼓励探索
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        # 实验名称，用于区分不同的训练实验
        experiment_name = 'rough_go2'

  
