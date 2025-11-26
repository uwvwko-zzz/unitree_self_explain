主要是写了这些方面的注释：
    unitree_rl_gym\deploy\deploy_real
    unitree_rl_gym\legged_gym

1.txt是对整个框架的浅显的理解

然后因为他只提供了一个最最最基本的模型，就是大平面训练，
如果要换成terrain模式的话  legged_robot.py  和  legged_robot_config.py   都要修改
unitree_rl_gym\legged_gym\envs\base\txt
这个txt是我自己的一些尝试（但是效果不佳）

然后  unitree_rl_gym\legged_gym\scripts\play_1ro.py  是一个可以通过WASDQE键盘控制机器人在大平面运动的play脚本

unitree_rl_gym\legged_gym\scripts\play_many_terrain_1ro.py  是在terrain的全地形

unitree_rl_gym\legged_gym\scripts\play_terrain_1ro.py   是单一个楼梯地形

unitree_rl_gym\legged_gym\scripts\test_model.py    用来检测模型质量，但是没搞好

我一开始看的就是宇树的，后面发现这基本都是基于legged_gym的，而且legged_gym的开源比unitree多，所以还是推荐理解一下代码然后自己在legged中注册go2，替换地形去学习

然后那个1500.pt就是平面模型，在平面上测试效果还不错
  

    
