import mujoco

# 加载模型
model = mujoco.MjModel.from_xml_path("/home/zwc/scoop_it2/assets/urdf/test.xml")#路径要改成你自己的
data = mujoco.MjData(model)

# 强制进行一次正向运动学计算，确定所有物体的世界坐标
mujoco.mj_forward(model, data)

# 获取两个 body 的世界坐标 (xpos)
pos_a = data.body("bar_left").xpos
pos_b = data.body("bar_right").xpos

# 计算欧氏距离
import numpy as np
dist = np.linalg.norm(pos_a - pos_b)
print(f"Body A 和 Body B 的初始距离为: {dist}")