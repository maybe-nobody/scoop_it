import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. 加载模型
# 请确保你的 test_old.xml 与此脚本在同一目录下
try:
    model = mujoco.MjModel.from_xml_path('test_old.xml')
    data = mujoco.MjData(model)
except ValueError as e:
    print(f"加载模型失败，请检查文件名或路径: {e}")
    exit()

# 2. 准备初始数据
# 这里的 6 个值分别对应每个手臂的 6 个关节
arm_initial_angles = np.array([0.0, 17, -1.2, 0.0, 0.0, 0.0])

# --- 自动获取索引 ---
try:
    # 获取第一个手臂的起始索引 (arm_joint1 ... arm_joint6)
    arm1_qpos_start = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint1")
    # 获取第二个手臂的起始索引 (arm_joint1_copy ... arm_joint6_copy)
    arm2_qpos_start = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint1_copy")
    
    # 获取执行器起始 ID
    act1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_actuator_1")
    act2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_actuator_1_copy")

    print(f"成功获取索引: Arm1_qpos={arm1_qpos_start}, Arm2_qpos={arm2_qpos_start}")
    print(f"成功获取控制: Act1_ID={act1_id}, Act2_ID={act2_id}")

except Exception as e:
    print(f"错误：无法在 XML 中找到对应的名称，请检查命名是否完全一致。具体错误: {e}")
    exit()

# --- 执行初始化 (类似训练中的 reset) ---

# A. 设置初始位置 (qpos) - 瞬移过程
data.qpos[arm1_qpos_start : arm1_qpos_start + 6] = arm_initial_angles
data.qpos[arm2_qpos_start : arm2_qpos_start + 6] = arm_initial_angles

# B. 设置初始控制目标 (ctrl) - 确保一启动力控就维持在这个位置
# 假设你的执行器是位置控制 (Position Control) 或增益控制
data.ctrl[act1_id : act1_id + 6] = arm_initial_angles
data.ctrl[act2_id : act2_id + 6] = arm_initial_angles

# C. 速度清零
data.qvel[:] = 0

# D. 物理状态同步
mujoco.mj_forward(model, data)

# 3. 开启可视化运行
print("启动可视化窗口... 此时机器人应处于你设定的初始角度。")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 物理步进
        # 如果你的 XML 中定义了 kp (增益)，机器人会努力维持在 ctrl 设定的位置
        mujoco.mj_step(model, data)
        
        # 画面同步
        viewer.sync()
        
        # 严格控制仿真步长，保持与现实时间一致
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
# import mujoco
# import mujoco.viewer
# import numpy as np
# import time

# # 1. 加载模型
# model = mujoco.MjModel.from_xml_path('test_old.xml')
# data = mujoco.MjData(model)

# # 2. 准备初始数据
# # 假设每个臂有 6 个关节，初始角度相同
# arm_initial_angles = np.array([0.0, 1.2, -1.2, 0, 0, 0.0])

# # --- 动态获取索引 (防止数错) ---
# # 获取第一个臂的 qpos 索引 (假设第一个关节名叫 arm_joint1)
# arm1_qpos_start = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint1")
# arm1_qpos_idx = slice(arm1_qpos_start, arm1_qpos_start + 6)

# # 获取第二个臂的 qpos 索引 (假设第一个关节名叫 arm_joint1_copy)
# arm2_qpos_start = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_joint1_copy")
# arm2_qpos_idx = slice(arm2_qpos_start, arm2_qpos_start + 6)

# # 获取执行器索引 (假设执行器按 arm_actuator1... 命名)
# # 如果你是按顺序定义的，通常 ctrl[0:6] 是臂1，ctrl[6:12] 是臂2
# act1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_actuator_1")
# act2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_actuator_1_copy") # 根据你实际命名的第一个执行器名获取

# # --- 设置初始状态 ---
# data.qpos[arm1_qpos_idx] = arm_initial_angles
# data.qpos[arm2_qpos_idx] = arm_initial_angles

# # 设置初始控制目标
# data.ctrl[act1_id : act1_id + 6] = arm_initial_angles
# data.ctrl[act2_id : act2_id + 6] = arm_initial_angles

# mujoco.mj_forward(model, data)

# # 3. 运行
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         step_start = time.time()

#         # 只要不改变 data.ctrl，PD控制器就会尝试维持在 initial_angles
#         mujoco.mj_step(model, data)
        
#         viewer.sync()
        
#         # 保持仿真频率
#         time_until_next_step = model.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)


# import mujoco
# import mujoco.viewer
# import numpy as np
# import time

# model = mujoco.MjModel.from_xml_path('test_old.xml')
# data = mujoco.MjData(model)

# # 目标锁定角度
# lock_angles = np.array([0.0, 1.2, -1.2, 0, 0, 0.0])

# # 获取两个臂的索引 (这里演示另一种获取方式，如果你知道确切索引)
# # 假设臂1是 15:21，臂2是 21:27（具体看你的 xml 顺序）
# arm1_idx = slice(15, 21)
# arm2_idx = slice(21, 27) 

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         step_start = time.time()

#         # 【强制锁定】
#         # 每一帧步进前，强行改写两个臂的状态
#         data.qpos[arm1_idx] = lock_angles
#         data.qpos[arm2_idx] = lock_angles
        
#         # 速度清零非常重要，否则会积累由于重力产生的“虚拟速度”导致画面抖动
#         data.qvel[arm1_idx] = 0.0
#         data.qvel[arm2_idx] = 0.0

#         mujoco.mj_step(model, data)
        
#         viewer.sync()

#         time_until_next_step = model.opt.timestep - (time.time() - step_start)
#         if time_until_next_step > 0:
#             time.sleep(time_until_next_step)
import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. 加载你的 XML 模型文件
# 请确保 your_robot.xml 路径正确，或者直接放入你的模型内容
model = mujoco.MjModel.from_xml_path("your_robot.xml")
data = mujoco.MjData(model)

def show_initial_pose(target_qpos):
    """
    模拟训练中的 reset 过程，将初始角度传进去并显示
    """
    # 验证输入长度是否匹配关节数量 (nq)
    if len(target_qpos) != model.nq:
        print(f"警告：输入维度({len(target_qpos)})与模型关节数({model.nq})不符！")
    
    # --- 核心操作：模拟训练中的初始化 ---
    # 这就是你说的“直接把位置给各个关节”，不涉及力控
    data.qpos[:len(target_qpos)] = target_qpos
    
    # 将速度清零，防止机器人有“惯性”
    data.qvel[:] = 0
    
    # 必须执行这一步！它会根据 qpos 算出所有零件在空间中的实际位置
    # 否则渲染出来的画面可能还是 0 位，或者出现贴图错位
    mujoco.mj_forward(model, data)
    # -----------------------------------

    print("已成功加载初始角度，正在打开预览窗口...")
    
    # 启动一个被动查看器，它不会自动运行物理模拟（mj_step）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 保持窗口打开
        while viewer.is_running():
            # 持续同步数据到画面
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    # 在这里输入你想要查看的初始角度（弧度制）
    # 假设你的机器人有 7 个关节，请根据实际数量修改列表长度
    my_init_angles = [0.5, -0.2, 1.0, -1.5, 0.0, 0.8, 0.0] 
    
    show_initial_pose(my_init_angles)