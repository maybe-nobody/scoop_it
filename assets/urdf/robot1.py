import mujoco
import mujoco.viewer
import os
import time

def main():
    # 获取 xml 文件的绝对路径
    xml_path = os.path.join(os.path.dirname(__file__), 'test.xml')
    
    if not os.path.exists(xml_path):
        print(f"错误: 找不到文件 {xml_path}")
        return

    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"加载 XML 出错: {e}")
        return

    print("正在启动 MuJoCo 仿真器...")
    print("提示: 在窗口中按 'Space' (空格) 开始仿真，按 'R' 重置。")

    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 保持窗口开启
        while viewer.is_running():
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(model, data)

            # 同步查看器
            viewer.sync()

            # 控制仿真频率（尽量匹配实时）
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()