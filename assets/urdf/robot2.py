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

    # --- 核心修复：正确使用 MjSpec 实例方法 ---
    try:
        print("正在初始化 MjSpec 并加载 XML...")
        
        # 1. 先创建一个空的 Spec 实例
        spec = mujoco.MjSpec()
        
        # 2. 调用实例方法加载文件 (这里是之前报错的地方)
        spec.from_file(xml_path)
        
        # 3. 强行设定超大雅可比矩阵空间和碰撞空间
        # 针对 15x15 或 10x10 的网格，20000 绰绰有余
        spec.size.njmax = 20000 
        spec.size.nconmax = 4000
        spec.size.nstack = 2000000
        
        # 4. 编译成最终模型
        model = spec.compile()
        data = mujoco.MjData(model)
        print(f"模型编译成功！当前 njmax: {model.njmax}")

    except AttributeError as e:
        # 如果版本不支持 MjSpec (低于 3.0)
        print(f"版本兼容性提示: {e}")
        print("正在尝试传统加载方式...")
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
        except Exception as e2:
            print(f"传统加载失败: {e2}")
            return
            
    except Exception as e:
        print(f"加载 XML 过程中发生错误: {e}")
        return

    # --- 仿真运行部分 ---
    print("\n正在启动 MuJoCo 仿真器...")
    print("提示: 在窗口中按 'Space' (空格) 开始仿真，按 'R' 重置。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 物理仿真步进
            mujoco.mj_step(model, data)

            # 同步渲染
            viewer.sync()

            # 尽量匹配实时频率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()