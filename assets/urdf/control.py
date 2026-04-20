import mujoco
import mujoco.viewer as viewer
import time

# 加载模型
model = mujoco.MjModel.from_xml_path("a1arm.xml")
data = mujoco.MjData(model)

# 打开可视化窗口
with viewer.launch_passive(model, data) as v:
    
    # 持续仿真
    while v.is_running():
        
        # 给机械臂六个关节输入力矩 (示例值)
        data.ctrl[0] = 0.5      # joint 1
        data.ctrl[1] = -0.2     # joint 2
        data.ctrl[2] = -0.2     # joint 3
        data.ctrl[3] = -0.2     # joint 4
        data.ctrl[4] = -0.2     # joint 5
        data.ctrl[5] = -0.2     # joint 6

        # 执行一步仿真
        mujoco.mj_step(model, data)

        # 控制刷新频率
        time.sleep(0.01)
