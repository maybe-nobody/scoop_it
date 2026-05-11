import mujoco

# 加载模型
model = mujoco.MjModel.from_xml_path("/home/lsq/scoop_it/assets/urdf/test_old.xml")#路径要改成你自己的
data = mujoco.MjData(model)

# 强制进行一次正向运动学计算，确定所有物体的世界坐标
mujoco.mj_forward(model, data)

# 获取两个 body 的世界坐标 (xpos)
pos_a = data.body("bar_left").xpos
pos_b = data.body("bar_right").xpos
bottom_left = data.site("hook_bottom_left").xpos
bottom_right = data.site("hook_bottom_right").xpos
top_left = data.site("hook_top_left").xpos
top_right = data.site("hook_top_right").xpos
frame_corner_fr = data.site("frame_corner_fr").xpos
frame_corner_br = data.site("frame_corner_br").xpos
frame_corner_bl = data.site("frame_corner_bl").xpos
frame_corner_fl = data.site("frame_corner_fl").xpos
# 计算欧氏距离
import numpy as np
dist = np.linalg.norm(pos_a - pos_b)
# print(f"Body A 和 Body B 的初始距离为: {dist}")
# print(f"Body A 的位置: {pos_a}")
# print(f"Body B 的位置: {pos_b}")   
dist_hook = np.linalg.norm(bottom_left - bottom_right)
print(f"hook_bottom_left 和 hook_bottom_right 的距离为: {dist_hook}")
print(f"hook_bottom_left 的位置: {bottom_left}")
print(f"hook_bottom_right 的位置: {bottom_right}")
print(f"hook_top_left 的位置: {top_left}")
print(f"hook_top_right 的位置: {top_right}")    
dist_frame = np.linalg.norm(frame_corner_fr - frame_corner_br)
print(f"frame_corner_fr 和 frame_corner_br 的距离为: {dist_frame}")
print(f"frame_corner_fr 的位置: {frame_corner_fr}")
print(f"frame_corner_br 的位置: {frame_corner_br}")
print(f"frame_corner_bl 的位置: {frame_corner_bl}")
print(f"frame_corner_fl 的位置: {frame_corner_fl}")
