import os
import mujoco
import mujoco.viewer
import re
#"/home/zwc/scoop_it2/assets/urdf/A1_RGM3.xml" 
# 1. 解决 Ubuntu 界面模糊：强制禁用系统缩放干扰
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['GDK_SCALE'] = '1'

def main():
    xml_path = "/home/zwc/scoop_it2/assets/urdf/A1_RGM3.xml"
    
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 彻底移除原有的 size 标签，防止冲突
    content = re.sub(r'<size\s+[^>]*/>', '', content)
    
    # 2. 插入一个绝对足够大的数值
    # 强制在 <mujoco> 之后的第一行插入
    new_size = '\n  <size njmax="8000" nconmax="4000"/>'
    content = re.sub(r'<mujoco([^>]*)>', r'<mujoco\1>' + new_size, content)

    try:
        # 3. 【核心修复】尝试使用底层 VFS 方式加载字符串
        # 这样可以绕过某些特定的文件解析缓存 Bug
        model = mujoco.MjModel.from_xml_string(content)
        data = mujoco.MjData(model)
        
        print(f"验证分配结果: njmax={model.njmax}") 
        
        if model.njmax < 10000:
            print("警告：内存分配仍被插件锁定。尝试降低布料密度进行测试。")

        mujoco.viewer.launch(model, data)
        
    except Exception as e:
        # 4. 如果依然报 5238，说明该版本的插件忽略了 size 标签
        # 此时唯一的办法是临时减小布料 count 来让它先跑通
        print(f"报错详情: {e}")
        if "allocated 5238" in str(e):
            print("\n检测到硬编码内存锁定。请尝试修改 XML 将布料 count 改为 '10 10 1' 再次运行。")

if __name__ == "__main__":
    main()