# 定义了两个核心配置类：LeggedRobotCfg 和 LeggedRobotCfgPPO，它们都继承自 BaseConfig，
# 用于在基于 Isaac Gym 或 RSL-RL（如 legged_gym 项目）的强化学习框架中，为四足机器人控制任务提供完整的环境与训练超参数设置

from .base_config import BaseConfig

# LeggedRobotCfg：定义了四足机器人的环境/任务/物理/奖励

# LeggedRobotCfgPPO：定义了四足机器人的PPO训练超参数，网络结构，优化器，激活函数。训练流程

# GUI -- 400

# 一开始的配置文件

class LeggedRobotCfg(BaseConfig):
    # 环境通用设置
    class env:
        num_envs = 4096      # 并行仿真实例数量（机器人数量）   (4096)
        num_observations = 48 # 策略输入的维度
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12      # 策略输出的维度
        env_spacing = 3.  # not used with heightfields/trimeshes    8.
        send_timeouts = True # 向算法传递episode是否超时结束
        episode_length_s = 20 # 每个episode持续20秒
        test = False

    # 地形设置
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane(平面), heightfield(高度场) or trimesh(三角网络)
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True           # 使用课程学习，从简单到复杂
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True          # 启用周围地形高度感知
        
        # 定义机器人周围 17×11 = 187 个测高点
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)  (10)
        num_cols = 20 # number of terrain cols (types)  (20)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # 不同地形的比例
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # 运动指令
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # 每10秒更换一次命令
        heading_command = True # 使用目标朝向误差计算角速度
        class ranges:
            # 期望的前后/侧向速度
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            # 期望的角速度/绝对朝向
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    # 初始状态
    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    # 控制器
    class control:
        # P控制
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # 动作缩放因子
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        # 每4次仿真步，更新一次动作
        decimation = 4
    # 机器人模型
    class asset:
        file = ""
        name = "legged_robot"  # actor name
        # 检测脚部接触
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        # 定义哪些部位触底惩罚/终止
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter    启用自碰撞
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    # 域随机化
    class domain_rand:
        randomize_friction = True
        # 随机摩擦系数
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        # 每隔 15 秒给机器人一个随机横向推力（模拟扰动），提高鲁棒性
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    # 奖励函数
    class rewards:
        class scales:
            termination = -0.0
            # 跟踪速度命令
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # 惩罚垂直速度
            lin_vel_z = -2.0
            # 惩罚侧滚/俯仰角速度
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            # 惩罚关节加速度（减小抖动）
            dof_acc = -2.5e-7
            base_height = -0. 
            # 鼓励有腾空相的动态步态
            feet_air_time =  1.0
            # 碰撞惩罚
            collision = -1.
            feet_stumble = -0.0 
            # 惩罚动作剧烈变化
            action_rate = -0.01
            stand_still = -0.

        # 总奖励为负时设为 0（避免早期崩溃）
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        # 使用高斯形式奖励
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    # 观测归一化
    class normalization:
        class obs_scales:
            # 进行缩放
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        # 防止极端值
        clip_observations = 100.
        clip_actions = 100.

    # 噪声
    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    # 仿真器设置
    class sim:
        dt =  0.005         # 策略频率50Hz
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]  重力
        up_axis = 1  # 0 is y, 1 is z

        # Isaac Gym 底层PhysX引擎参数
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

# PPO算法训练配置
class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    # 神经网络结构
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid   激活函数  
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    
    # PPO核心参数
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2                # PPO的clipping范围
        entropy_coef = 0.01             # 鼓励探索
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches  每个epoch的mini batch数量
        learning_rate = 1.e-3 #5.e-4     学习率
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99                    # 折扣因子
        lam = 0.95                      # GAE 参数
        desired_kl = 0.01               # 自适应学习率调整
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration      每24步数据更新一次
        max_iterations = 1500 # number of policy updates     最大训练迭代次数

        # logging
        save_interval = 50 # check for potential saves every this many iterations       每50保存一次
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt