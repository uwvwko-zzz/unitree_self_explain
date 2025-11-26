import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# 定义了 三个用于四足机器人强化学习仿真的实用数学工具函数
# @ torch.jit.script
# 仅应用四元数中的偏航角（yaw），忽略俯仰（pitch）和翻滚（roll），将向量 vec 绕 Z 轴旋转
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
# 将角度（弧度）限制在区间 [−π,π) ，即“角度归一化”
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
# 生成一个在 [lower, upper] 区间内、按平方根分布的随机数张量
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower