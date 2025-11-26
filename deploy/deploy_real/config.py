from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml

# 把 YAML 配置文件的所有部署参数加载到内存，供部署主循环（Controller）使用，
# 保证训练时的动作/观测映射、控制周期、话题名、消息格式与真实机器人一致

class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            # 打开deploy_real/configs文件下的config文件，具体文件名在启动deploy_real.py时在终端
            config = yaml.load(f, Loader=yaml.FullLoader)
            
            # 将yaml配置文件加载进来
        	# Loader=yaml.FullLoader参数表示加载完整的YAML语言，支持YAML规范中的所有语法特性（如锚点、标签、流控制符等）
        	# 并能将 YAML 文档完整地转换为Python对象
        	# 其他类型参数还有SafeLoader，BaseLoader
         
            # 该参数是控制周期参数，即多长时间向机器人发布一次指令
            self.control_dt = config["control_dt"]
            
            # 该参数是控制消息类型，有"hg"和"go"两种类型
            self.msg_type = config["msg_type"]
            # 该参数是IMU消息类型，有"pelvis"(骨盆)和"torso"(躯干)两种类型
            # 人型的两个机器人（h1和h1_2）的imu位置应该是装在了躯干上，需要转换到骨盆坐标系下
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]
            # 底层电机指令的话题名称
            self.lowcmd_topic = config["lowcmd_topic"]
            # 底层状态的话题名称
            self.lowstate_topic = config["lowstate_topic"]
            # 网络pt文件的路径，读入config文件后将{LEGGED_GYM_ROOT_DIR}字符串替换为LEGGED_GYM_ROOT_DIR变量的值
			# LEGGED_GYM_ROOT_DIR的值在legged_gym文件夹下的__init__.py脚本里定义
            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            # 腿部关节转化到电机序号索引
            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            # 腿部关节电机的P参和D参
            self.kps = config["kps"]
            self.kds = config["kds"]
            # 腿部关节电机的初始默认角度
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
			# 胳膊和腰部关节电机的序号索引
            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            # 胳膊和腰部关节电机的P参和D参
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            # 胳膊和腰部关节电机的初始默认角度
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)
			
			# 输入训练好的强化学习网络的观测值比例
			# 本体角速度的比例
            self.ang_vel_scale = config["ang_vel_scale"]
            # 关节位置比例
            self.dof_pos_scale = config["dof_pos_scale"]
            # 关节速度比例
            self.dof_vel_scale = config["dof_vel_scale"]
            # 动作比例
            self.action_scale = config["action_scale"]
            # 指令比例
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            # 指令最大值
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)
			# 动作的数量
            self.num_actions = config["num_actions"]
            # 观测值的数量
            self.num_obs = config["num_obs"]



