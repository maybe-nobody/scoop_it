"""
Author: Yuanhang Zhang 李思岐
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the RL environment for the DCMM task
"""
import os, sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./gym_dcmm/'))
import argparse
import math
from collections import OrderedDict
#print(os.getcwd())
import configs.env.DcmmCfg as DcmmCfg
import cv2 as cv
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM
from gym_dcmm.utils.ik_pkg.ik_base import IKBase
import copy
from termcolor import colored
from decorators import *
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from utils.util import *
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from collections import deque
import os
#print("CWD:", os.getcwd())
# os.environ['MUJOCO_GL'] = 'egl'
np.set_printoptions(precision=8)#打印浮点数时，小数点后的位数最多保留 8 位（四舍五入）

paused = True
cmd_lin_y = 0.0
cmd_lin_x = 0.0
cmd_ang = 0.0
trigger_delta = False
trigger_delta_hand = False
print_distance = False
def env_key_callback(keycode):#键盘事件回调函数
  print("chr(keycode): ", (keycode))
  global cmd_lin_y, cmd_lin_x, cmd_ang, paused, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
  if keycode == 265: # AKA: up上箭头
    cmd_lin_y += 1
    print("up %f" % cmd_lin_y)
  if keycode == 264: # AKA: down下箭头
    cmd_lin_y -= 1
    print("down %f" % cmd_lin_y)
  if keycode == 263: # AKA: left左箭头
    cmd_lin_x -= 1
    print("left: %f" % cmd_lin_x)
  if keycode == 262: # AKA: right右箭头
    cmd_lin_x += 1
    print("right %f" % cmd_lin_x) 
  if keycode == 52: # AKA: 4
    cmd_ang -= 0.2
    print("turn left %f" % cmd_ang)
  if keycode == 54: # AKA: 6
    cmd_ang += 0.2
    print("turn right %f" % cmd_ang)
  if chr(keycode) == ' ': # AKA: space，chr是把一个整数（Unicode 码点）转换成对应的字符
    if paused: paused = not paused#:后面的东西相当于重新起一行，paused初始值是true
  if keycode == 334: # AKA + (on the numpad)小键盘上的加号
    trigger_delta = True
    delta_xyz = 0.1
  if keycode == 333: # AKA - (on the numpad)
    trigger_delta = True
    delta_xyz = -0.1
  if keycode == 327: # AKA 7 (on the numpad)
    trigger_delta_hand = True
    delta_xyz_hand = 0.2
  if keycode == 329: # AKA 9 (on the numpad)
    trigger_delta_hand = True
    delta_xyz_hand = -0.2

class DcmmVecEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "depth_array", "depth_rgb_array"]}
    """
    Args:
        render_mode: str
            The mode of rendering, including "rgb_array", "depth_array".
        render_per_step: bool
            Whether to render the mujoco model per simulation step.是否在每次模拟 step 时渲染一次 mujoco 模型
        viewer: bool
            Whether to show the mujoco viewer.是否显示 mujoco 自带的 3D 可视化窗口
        imshow_cam: bool
            Whether to show the camera image.是否显示相机捕获的图像
        object_eval: bool
            Use the evaluation object.是否使用“评估对象”（比如用来测试模型表现时的特定物体）。
        camera_name: str
            The name of the camera.相机的名字（在 mujoco 模型文件里定义的）。
        object_name: str
            The name of the object.物体的名字（在 mujoco 模型里定义的）
        env_time: float
            The maximum time of the environment.环境的最大运行时间（秒）
        steps_per_policy: int
            The number of steps per action.每个动作持续多少个仿真步。
        img_size: tuple
            The size of the image.输出图像的大小
    """
    '''    
    env = DcmmVecEnv(task='Catching', object_name='object', render_per_step=False, 
                    print_reward=False, print_info=False, 
                    print_contacts=False, print_ctrl=False, 
                    print_obs=False, camera_name = ["top"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam, 
                    viewer = args.viewer, object_eval=False,
                    env_time = 2.5, steps_per_policy=20)
    env.run_test()
    '''
    def __init__(
        self,
        task="Tracking",
        render_mode="depth_array",
        render_per_step=False,#是否在每次模拟 step 时渲染一次 mujoco 模型
        viewer=False,#是否显示 mujoco 自带的 3D 可视化窗口
        imshow_cam=False,#是否显示相机捕获的图像
        object_eval=False,#是否使用“评估对象”（比如用来测试模型表现时的特定物体）。
        camera_name=["top", "wrist"],
        object_name="object",
        env_time=2.5,#强化学习中的一个episode的最大运行时间
        steps_per_policy=20,#每个动作持续多少个仿真步。每执行一次策略动作，智能体会根据当前环境状态输出一个动作
        img_size=(480, 640),
        device='cuda:0',#运算在gpu上跑,在第一块显卡上
        print_obs=True,
        print_reward=False,
        print_ctrl=True,
        print_info=False,
        print_contacts=False,
        closed=False,
        print_hand=False,
    ):
        if task not in ["Tracking", "Catching"]:
            raise ValueError("Invalid task: {}".format(task))#检测task参数是否合法
        assert render_mode is None or render_mode in self.metadata["render_modes"]#断言通过程序就会接着执行，不然就报错，字典可以用方括号来放映键所对应的值
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.object_name = object_name
        self.imshow_cam = imshow_cam
        self.task = task
        self.closed = closed
        self.img_size = img_size
        self.device = device
        self.print_hand = print_hand
        self.steps_per_policy = steps_per_policy#每个动作持续多少个仿真步
        self.render_per_step = render_per_step
        # Print Settings
        self.print_obs = print_obs
        self.print_reward = print_reward
        self.print_ctrl = print_ctrl
        self.print_info = print_info
        self.print_contacts = print_contacts
        # 在 __init__ 适当位置添加
        self.grasp_threshold = 0.05  # 进入抓取阶段的距离阈值（米）
        self.gripper_width_max = 0.05  # 夹爪最大开度
        # 成功阈值：盘子中心距离“物体下方目标点”小于 10cm
# 先用 0.10，等成功率上来后再改成 0.08 或 0.06
        #self.traj_success_threshold = 0.10
        # Initialize the environment
        # 在 self.Dcmm = MJ_DCMM(...) 之前
        # print("="*30)
        # print(f"当前 Python 工作目录: {os.getcwd()}")
        # import configs.env.DcmmCfg as DcmmCfg
        # # 打印配置里拼接后的绝对路径
        # absolute_xml = os.path.abspath(os.path.join(os.getcwd(), DcmmCfg.XML_DCMM_LEAP_OBJECT_PATH))
        # print(f"程序试图打开的 XML 绝对路径: {absolute_xml}")
        # if not os.path.exists(absolute_xml):
        #     print("警告：该路径下的文件不存在！")
        # print("="*30)
        self.Dcmm = MJ_DCMM(viewer=viewer, object_name=object_name, object_eval=object_eval)
        # self.Dcmm.show_model_info()
        self.fps = 1 / (self.steps_per_policy * self.Dcmm.model.opt.timestep)#self.model.opt.timestep = timestep=0.002
        # ================== 【修改步骤 1：初始化历史参数】 ==================
        # 建议放在这里，在 observation_space 定义之前
        self.obj_history_len = 3  # 记录过去3帧
        self.obj_pos_history = deque(maxlen=self.obj_history_len)
        # ==============================================================
        # Randomize the Object Info
        self.random_mass = 0.25#
        self.object_static_time = 0.5#物体在初始状态下静止不动的时间。仿真开始的前 0.75 秒内，物体保持静止
        self.object_throw = False#是否让物体在环境中被抛掷或动态移动
        self.object_train = True#指示当前环境处于训练阶段还是评估阶段。
        if object_eval: self.set_object_eval()
        self.arm_limit_left = True
        self.arm_limit_right = True
        self.arm_limit = True
        '''    
        def set_object_eval(self):
            self.object_train = False
        '''
        self.Dcmm.model_xml_string = self._reset_object()#xml树的str
        self.Dcmm.model = mujoco.MjModel.from_xml_string(self.Dcmm.model_xml_string)#就是把 self.Dcmm.model_xml_string（XML 的字符串形式）传给 MuJoCo，然后由 from_xml_string 方法解析它，并生成一个 MjModel 物理模型对象
        self.Dcmm.data = mujoco.MjData(self.Dcmm.model)#创建一个新的 MjData 实例，并把它保存到 self.Dcmm.data。MjData存放动态信息
        # Get the geom id of the hand, the floor and the object
        #self.hand_start_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper1') - 1#mujoco.mj_name2id(model, type, name)根据名字查找id的索引
        #print("self.hand_start_id: ", self.hand_start_id)
        self.base_link_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'base_link')
        self.arm1_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg1')
        self.arm2_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg2')
        self.arm3_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg3')
        self.arm4_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg4')
        self.arm5_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg5')
        self.arm6_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg6')
        self.arm1_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg1_copy')
        self.arm2_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg2_copy')
        self.arm3_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg3_copy')
        self.arm4_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg4_copy')
        self.arm5_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg5_copy')
        self.arm6_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg6_copy')
        self.base_link_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'base_link_copy')
        self.plate_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'frame_bottom')
        #print("#############################id")
        #print(self.arm1_id)
        #print(self.arm2_id)
# ==========================================================
# --- 夹爪 ID 全能检测模块 (Debug Hand IDs) ---
# ==========================================================
# --- 改为以下精确获取 ---
        # # 1. 用于 Observation (qposadr): 对应调试输出的 21
        # self.hand_qpos_addr = self.Dcmm.model.joint('gripper1_axis').qposadr[0]
        
        # # 2. 用于 Step 控制 (actuator): 对应调试输出的 14
        # self.hand_ctrl_id = self.Dcmm.model.actuator('hand_actuator_0').id

        # # 3. 用于碰撞检测 (geom): 对应调试输出的 57 和 59
        self.f1_geom_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper1')
        self.f2_geom_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper2')

        # # 顺便获取手臂起始 ID，用于 mask_coll 判定 (假设 link1 是手臂起始)
        self.arm_start_geom_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm_seg1')
        # ------------------------------
        self.floor_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)
        self.base_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'ranger_base')
        self.base_id_copy = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'ranger_base_copy')
        # Set the camera configuration
        self.Dcmm.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]#离屏渲染
        self.Dcmm.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.mujoco_renderer = MujocoRenderer(
            self.Dcmm.model, self.Dcmm.data
        )#给渲染器提供 需要渲染的场景（模型和数据），内部会用到offscreen rendering API，初始化了渲染，告诉应该用什么模型和数据进行渲染
        if self.Dcmm.open_viewer:# 如果配置里要求打开可视化窗口
            if self.Dcmm.viewer:#如果之前已经有一个 viewer 窗口存在
                print("Close the previous viewer")
                self.Dcmm.viewer.close()#把之前的窗口关闭掉
            self.Dcmm.viewer = mujoco.viewer.launch_passive(self.Dcmm.model, self.Dcmm.data, key_callback=env_key_callback)#被动渲染，viewer 只是一个“画布”，不会自动推进仿真，被动渲染通过代码控制，适合强化学习
            #key_callback=env_key_callback允许与键盘交互
            # Modify the view position and orientation
            self.Dcmm.viewer.cam.lookat[0:2] = [0, 1]#lookat是长度为3的数组，意思是相机注视（0，1，0）
            self.Dcmm.viewer.cam.distance = 5.0#相机与注视点的距离
            self.Dcmm.viewer.cam.azimuth = 180#相机从正后方看目标
            # self.viewer.cam.elevation = -1.57
        else: self.Dcmm.viewer = None

        # Observations are dictionaries with the agent's and the object's state. (dim = 44)
        hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15#hand_mask是一个数组，等于1表示是手
        #部的关节，==1会得到一个布尔数组，表示哪些位置等于 1，np.where 返回的是一个 元组，[0]是为了把array取出来
        #手的关节编号在模型里是 从第 15 个关节以后开始的，所以要整体偏移。
        self.observation_space = spaces.Dict(#定义机器人环境的观测空间，一共30维 ee pos,ee vel,joint pos,obj pos vel,
            {
                "base1": spaces.Dict({
                    "v_lin_3d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),#spaces.Box(low, high, shape, dtype)
                    "base_pos": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                }),
                "base2": spaces.Dict({
                    "v_lin_3d": spaces.Box(-4,  4, shape=(2,), dtype=np.float32),#spaces.Box(low, high, shape, dtype)
                    "base_pos": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                }),
                "arm1": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(9, 15)]),#（9,15)代表的是6个关节，这是给6个关节找上下限
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(9, 15)]),
                                            dtype=np.float32),
                }),
                "arm2": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-5, 5, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(26, 32)]),#（9,15)代表的是6个关节，这是给6个关节找上下限
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(26, 32)]),
                                            dtype=np.float32),
                }),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-5, 5, shape=(3,), dtype=np.float32),
                    ## TODO: to be determined
                    # "shape": spaces.Box(-5, 5, shape=(2,), dtype=np.float32),
                }),
                "plate": spaces.Dict({
                    # 目标点 - 当前盘子中心，3 维向量
                    # 这个量告诉策略：盘子应该往哪个方向移动
                    "plate_pos": spaces.Box(
                        low=-10.0,
                        high=10.0,
                        shape=(3,),
                        dtype=np.float32
                    ),
                    }),
                # "trajectory": spaces.Dict({
                #     # 当前小目标 ref - 当前盘子位置
                #     "plate_ref_error": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),

                #     # # 最终预测接球点 target - 当前盘子位置
                #     # "plate_target_error": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),

                #     # # 当前轨迹 alpha，范围 0~1
                #     # "traj_alpha": spaces.Box(0, 1, shape=(1,), dtype=np.float32),

                #     # # 从当前时刻开始，预计还有多少秒到接球高度
                #     # "time_to_catch": spaces.Box(0, 2, shape=(1,), dtype=np.float32),

                #     # # 球是否已经抛出
                #     # "object_throw": spaces.Box(0, 1, shape=(1,), dtype=np.float32),

                #     # 当前 episode 进度
                #     "progress": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                # }),
            }
        )#如果手是12维的话，这里就是36维
        # Define the limit for the mobile base action
        # base_low = np.array([-4, -4])#两个方向的速度，所以是2维
        # base_high = np.array([4, 4])
        # === 修改后 (DcmmVecEnv.py) ===
        # 增加第3维作为模式选择信号 (假设范围 -1 到 1)
        base_low = np.array([-4, -4]) 
        base_high = np.array([4, 4])
        # Define the limit for the arm action
        arm_low = -0.025*np.ones(3)
        arm_high = 0.025*np.ones(3)
        # Define the limit for the hand action
        hand_low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices])#hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15
        hand_high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices])#jnt_range是固定的表述，表示的就是第几个joint
        #传入hand活动范围的最大最小值，最大最小值是存储在.xml文件中的
        # Get initial ee_pos3d
        self.init_pos = True
        self.initial_ee_pos3d = self._get_relative_ee_pos3d()
        self.initial_ee_pos3d_copy = self._get_relative_ee_pos3d_copy()
        '''return np.array([x, y, 
                         self.Dcmm.data.body("link6").xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#末端执行器(link6)在垂直方向(Z轴)上相对于机械臂底座(arm_base)的高度差。
        x,y是末端执行器的位置'''
        self.initial_obj_pos3d = self._get_object_pos3d()
        self.initial_obj_pos3d_copy = self._get_relative_object_pos3d_copy()
        '''return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#物体在 Z 方向（垂直方向）相对于机械臂基座的高度差。
        x,y是机械臂基地的位置'''
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_obj_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d_copy = np.array([0.0, 0.0, 0.0])
        self.prev_obj_pos3d_copy = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d[:] = self.initial_obj_pos3d[:]
        self.prev_ee_pos3d_copy[:] = self.initial_ee_pos3d_copy[:]
        self.prev_obj_pos3d_copy[:] = self.initial_obj_pos3d_copy[:]

        # Actions (dim = 20)
        self.action_space = spaces.Dict(
            {
                "base": spaces.Box(base_low, base_high, shape=(2,), dtype=np.float32),#定义机器人底座的控制动作。
                "arm": spaces.Box(arm_low, arm_high, shape=(3,), dtype=np.float32),#定义机器人机械臂的控制动作
                "hand": spaces.Box(low = hand_low,
                                   high = hand_high,
                                   dtype = np.float32),#这里的hand的维数是由上面的high和low确定的，他们是几维的hand就是几维的
                "base_copy": spaces.Box(base_low, base_high, shape=(2,), dtype=np.float32),#定义机器人底座的控制动作。
                "arm_copy": spaces.Box(arm_low, arm_high, shape=(3,), dtype=np.float32),#定义机器人机械臂的控制动作
                "hand_copy": spaces.Box(low = hand_low,
                                   high = hand_high,
                                   dtype = np.float32),
            }
        )#2+2+3+3+2+2=
        self.action_buffer = {
            "base": DynamicDelayBuffer(maxlen=2),
            "arm": DynamicDelayBuffer(maxlen=2),
            "hand": DynamicDelayBuffer(maxlen=2),#hand原来是12维
            "base_copy": DynamicDelayBuffer(maxlen=2),
            "arm_copy": DynamicDelayBuffer(maxlen=2),
            "hand_copy": DynamicDelayBuffer(maxlen=2),
        }#定义一个字典来保存动作缓冲区，每一帧都会接受一个新动作，但是由于延迟和惯性，动作不会立即生效
        # Combine the limits of the action space
        self.actions_low = np.concatenate([base_low, arm_low, hand_low])#拼成一个大的数组
        self.actions_high = np.concatenate([base_high, arm_high, hand_high])

        self.obs_dim = get_total_dimension(self.observation_space)
        self.act_dim = get_total_dimension(self.action_space)#计算一个嵌套的观测空间（observation space）或动作空间（action space）的总维度
        '''
        def get_total_dimension(data):
            # print("type data: ", type(data))
            total_dimension = 0
            # If it is a dictionary, recursively process its values.
            if isinstance(data, spaces.Dict) or isinstance(data, dict):
                for value in data.values():
                    total_dimension += get_total_dimension(value)
            # If it is a box, return the size of the box.
            elif isinstance(data, spaces.Box):
                return data.shape[0]
            # If it is an array, return the size of the array.
            elif isinstance(data, np.ndarray):
                return data.size
            # If it is a single element, return 1.
            else:
                return 1
            
            return total_dimension
        '''
        #act里面的arm都是4维的，只要给定末端的就可以算出六个机械臂关节的角度
        #obs里面的arm就是6维的机械臂角度
        self.obs_t_dim = self.obs_dim   # dim = 18, 12 for the hand, 6 for the arm joint positions
        #还要-6的原因：Tracking Task（跟踪任务）不需要关节角度，只需要末端（EE）
        self.act_t_dim = self.act_dim - 4 # dim = 6, 12 for the hand
        self.obs_c_dim = self.obs_dim - 6  # dim = 30, 6 for the arm joint positions
        #维度要-6的原因：Catching 任务只需要知道末端的位置，不需要知道每个关节角度
        self.act_c_dim = self.act_dim # dim = 18,现在的dim是8维
        ########################################之前的打印信息##########################################################################################
        #print("##### Tracking Task \n obs_dim: {}, act_dim: {}".format(self.obs_t_dim, self.act_t_dim))
        #print("##### Catching Task \n obs_dim: {}, act_dim: {}\n".format(self.obs_c_dim, self.act_c_dim))
        ######################################################################################################################
        # Init env params
        self.arm_limit = True#：表示是否对机械臂的动作施加限制
        self.terminated = False#标记当前环境是否已经结束（episode 是否终止）。
        self.start_time = self.Dcmm.data.time#环境开始的仿真时间。
        self.catch_time = self.Dcmm.data.time - self.start_time#
        self.reward_touch = 0#触碰到给多少分
        self.reward_stabilitspaces = 0#物体没有掉落或者晃动，初始化奖励值
        self.env_time = env_time#环境允许的最大运行时间
        self.stage_list = ["tracking", "grasping"]#定义环境中的任务阶段，通常机器人抓取任务有多个阶段：，跟踪，抓取
        # Default stage is "tracking"当前任务是跟踪
        self.stage = self.stage_list[0]
        self.steps = 0#记录当前 episode 已经执行的步数
        # ================== 轨迹跟踪参数 ==================
        # 盘子目标点：物体下方多少米
        # 例如 object_z = 0.9，则 plate_target_z = 0.9 - 0.18 = 0.72
        self.catch_z_offset = 0.0
        self.traj_reach_ratio = 0.5# 前 65% 的 episode 用来走到目标点，后 35% 用来保持
        self.traj_success_threshold = 0.1# 成功阈值：盘子中心距离最终目标点小于 6cm，认为轨迹跟踪成功
        self.traj_total_steps = max(1, int(self.env_time * self.fps))#整个episode总共有多少step，policy step，fps就是策略步的频率
        self.traj_reach_steps = max(1, int(self.traj_total_steps * self.traj_reach_ratio))
        # 每个 episode reset 的时候会重新赋值
        self.traj_start_plate_pos = np.zeros(3, dtype=np.float64)
        self.traj_target_plate_pos = np.zeros(3, dtype=np.float64)
        self.traj_ref_plate_pos = np.zeros(3, dtype=np.float64)
        self.prev_traj_error = 0.0
        # 非滚动轨迹的计时起点。
        # 对抛物任务来说，轨迹 alpha 不应该从 episode 第 0 步开始算，
        # 而应该从物体真正开始抛出之后算。
        self.traj_start_step = 0

        # 盘子最好比物体提前一点到达预测接物点，留安全余量。
        # 例如物体还有 0.40s 到接物高度，盘子希望 0.35s 左右到。
        self.catch_arrive_early_time = 0.1

        # 防止 traj_reach_steps 太小，导致 alpha 一下子跳到 1。
        self.min_traj_reach_steps = 3
        # 每个 episode reset 的时候会重新赋值
        # traj_start_plate_pos：这个 episode 一开始盘子在哪里
        # traj_target_plate_pos：最终希望盘子去哪里，一般是物体下方
        # traj_ref_plate_pos：当前 step 应该追的中间轨迹点
        # traj_reach_steps：多少步内走到最终目标点
        # 上一步的轨迹误差，用于计算“有没有变近”

        # ==================================================
        # ================== 动态物体预测 / 滚动规划参数 ==================
        # 是否使用动态物体预测。静态物体时可以关掉。
        self.use_dynamic_prediction = True

        # 预测未来多少秒后的物体位置。
        # 第一版建议 0.6~0.9 秒之间调。
        self.catch_predict_time = 0.4#t_hit解不出来，就先用固定的预测时间点来预测物体位置。
        self.prethrow_reach_time = 2
        self.prethrow_reach_steps = max(1, int(self.prethrow_reach_time * self.fps))#在物体u没有被抛出之前，希望盘子可以先往物体下面靠近，预热一下
        # 每隔多少个 policy step 重新规划一次目标点。
        # 不建议每一步都重规划，容易抖；3~5 比较合适。
        self.replan_interval = 3

        # 目标点平滑系数。越小越平滑，越大反应越快。
        self.target_smooth_beta = 0.2#new_target = (1.0 - beta) * old_target + beta * limited_target

        # 每次重规划目标点最多允许移动多少米，防止目标跳变过大。
        self.max_target_shift_per_replan = 10
        # 滚动规划用：每次 replan 后重新从当前盘子位置走向新目标
        self.steps_since_replan = 0
        # 用最近几帧物体世界坐标估计速度，方便以后迁移到真实系统。
        # success 连续保持计数
        self.success_counter = 0
        self.success_hold_steps = 5
        self.has_success = False
        self.success_bonus_given = False
        self.obj_world_history_len = 5
        self.obj_world_pos_history = deque(maxlen=self.obj_world_history_len)
        self.obj_world_time_history = deque(maxlen=self.obj_world_history_len)
        # ==================================================
        # waypoint path 参数
        # ==================================================
        # 保存 start -> wp1 -> wp2 -> target 这条路径
        self.plate_waypoints = None

        # 上一步沿 waypoint 路径的进度，用来奖励是否往前走
        self.prev_path_progress = 0.0
        self.num_path_midpoints = 0
        # 盘子离路径多远算偏离，单位 m
        # 越小，要求越贴着路径走；越大，路径约束越宽松
        self.path_sigma = 0.20

        # 中间 waypoint 的高度抬升量
        # 作用：让盘子中心路径更自然，不是直接低平地硬拉过去
        self.path_lift_z = 0.0

        # path_progress 奖励每步最大允许变化
        # 防止某一步 progress 跳变导致 reward 爆炸
        self.path_progress_clip = 0.05

        # ==============================================================
        self.prev_ctrl = np.zeros(8)#上一时刻机器人控制输入（动作指令base(2) + arm(4) + hand(12) = 18）
        self.init_ctrl = True#表示当前是否处于 初始化控制状态。当第一次控制指令被应用之后，代码会把这个标志位改成false
        self.vel_init = False#速度是否已经初始化。
        self.vel_history = deque(maxlen=4)#一个 速度历史缓存队列，最多存储 4 个最近的速度值。
        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("arm_seg6").xpos - 
                                          self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),#计算手与目标物体的距离
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] - 
                                            self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),#机械臂基座和目标物体在平面上的距离，判断底盘是否需要移动
            "base_distance_copy": np.linalg.norm(self.Dcmm.data.body("arm_base_copy").xpos[0:2] - 
                                            self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "env_time": self.Dcmm.data.time - self.start_time,#计算环境运行了多长时间
            "imgs": {},#用来存取拍摄帧
            "qpos_sum": self.Dcmm.data.joint("gripper2_axis").qpos[0]+self.Dcmm.data.joint("gripper1_axis").qpos[0], # 机械臂关节位置的总和（用于调试）
            "plate_distance":np.linalg.norm(self.Dcmm.data.site("frame_bottom_center").xpos - self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "plate_z_diff" : np.abs(self.Dcmm.data.site("frame_bottom_center").xpos[2] - 
                      self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]),
            "ee_position":self.Dcmm.data.body("arm_seg6").xpos,
            "ee_position_copy":self.Dcmm.data.body("arm_seg6_copy").xpos,
        }
        self.contacts = {
            "any_base_collision": False,
            "any_arm_collision": False,
            "object_failed": False,      # 物体非法碰撞或掉地
            "object_on_plate": False     # 物体是否成功落在盘子上（用于奖励）
        }

        self.object_q = np.array([1, 0, 0, 0])#物体的四元数，表示物体在空间中的旋转姿态
        self.object_pos3d = np.array([0, 0, 1.5])#物体的坐标
        self.object_vel6d = np.array([0., 0., 1.25, 0.0, 0.0, 0.0])#前三线速度，后三角速度
        self.step_touch = False#手是否与物体接触

        self.imgs = np.zeros((0, self.img_size[0], self.img_size[1], 1))#        img_size=(480, 640),
        #第一个维度代表的是有几张照片，后面三个维度是分辨率和通道数
        # Random PID Params让 RL 训练更鲁棒，不怕噪声，使模型在真实环境更稳定。
        self.k_arm = np.ones(6)
        self.k_drive = np.ones(4)#电机驱动轮
        self.k_steer = np.ones(4)#电机转向轮
        self.k_hand = np.ones(1)
        # Random Obs & Act Params，观测和动作的随机化尺度”
        self.k_obs_base = DcmmCfg.k_obs_base
        self.k_obs_arm = DcmmCfg.k_obs_arm
        self.k_obs_hand = DcmmCfg.k_obs_hand
        self.k_obs_object = DcmmCfg.k_obs_object
        self.k_act = DcmmCfg.k_act
        #这些 k_xxx 参数是随机化噪声系数，用于增强训练鲁棒性，分别控制底座、机械臂、手、物体、观测和动作的噪声大小。
        '''
        k_obs_base = 0.01
        k_obs_arm = 0.001
        k_obs_object = 0.01
        k_obs_hand = 0.01
        '''
        self.keep_ee_level = False

        # 保存 reset 时的末端姿态，作为 IK 的目标姿态
        self.target_ee_quat_wxyz = None
        self.target_ee_quat_wxyz_copy = None
    def set_object_eval(self):
        self.object_train = False

    def update_render_state(self, render_per_step):
        self.render_per_step = render_per_step

    def update_stage(self, stage):
        if stage in self.stage_list:
            self.stage = stage
        else:
            raise ValueError("Invalid stage: {}".format(stage))

    def _get_contacts(self):
        # 1. 获取当前所有碰撞对数据
        geom_ids = self.Dcmm.data.contact.geom
        
        # 初始化返回字典
        results = {
            "any_base_collision": False,
            "any_arm_collision": False,
            "object_failed": False,      # 物体非法碰撞或掉地

            # 只有碰到 frame_bottom，才认为真正落在盘子底面上
            "object_on_plate": False,

            # 只碰到 edge_left / edge_right / edge_back / edge_front
            # 不算失败，但也不算真正落盘
            "object_on_edge": False
        }

        if geom_ids.size == 0:
            return results

        geom1_ids = geom_ids[:, 0]
        geom2_ids = geom_ids[:, 1]

        # --- 【ID 组定义】 ---
        # 12个机械臂连杆 ID
        arm_all_ids = [
            self.arm1_id, self.arm2_id, self.arm3_id, self.arm4_id, self.arm5_id, self.arm6_id,
            self.arm1_id_copy, self.arm2_id_copy, self.arm3_id_copy, self.arm4_id_copy, self.arm5_id_copy, self.arm6_id_copy,
        ]
        # 两个底座 ID
        bases_all_ids = [self.base_link_id, self.base_link_id_copy]
        # 盘子底面 ID：只有 frame_bottom 才算真正落盘成功
        plate_id = self.plate_id

        # 真正可以触发 object_on_plate 的 geom
        # 目前只允许 frame_bottom 触发成功
        plate_success_ids = set([plate_id])

        # 四个边缘 geom：
        # 碰到这些不算失败，但是也不算 object_on_plate
        plate_edge_ids = []

        for name in ["edge_left", "edge_right", "edge_back", "edge_front"]:
            gid = mujoco.mj_name2id(
                self.Dcmm.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name
            )
            if gid != -1:
                plate_edge_ids.append(gid)

        plate_edge_ids = set(plate_edge_ids)

        # 允许接触的盘子相关 geom：
        # frame_bottom + 四个边缘
        # 作用：这些接触都不算 object_failed
        plate_allowed_ids = plate_success_ids | plate_edge_ids

        # 被接的物体 ID
        object_id = self.object_id
        # 地面 ID
        floor_id = self.floor_id

        # --- 【辅助函数：查找谁碰到了谁】 ---
        def get_partners(target_group):
            """获取与目标组发生碰撞的所有对方物体 ID"""
            idx1 = np.where(np.isin(geom1_ids, target_group))[0]
            idx2 = np.where(np.isin(geom2_ids, target_group))[0]
            partners = np.concatenate((geom2_ids[idx1], geom1_ids[idx2]))
            return partners.astype(int)

        # 2. 提取关键部件的碰撞伙伴
        base_partners = get_partners(bases_all_ids)
        arm_partners = get_partners(arm_all_ids)
        object_partners = get_partners([object_id])

        # --- 【判定核心逻辑】 ---
        
        # A. 底座判定：撞到非地面物体即失败
        if base_partners.size > 0:
            results["any_base_collision"] = np.any(base_partners != floor_id)

        # B. 机械臂判定：撞到非盘子、非地面的物体即失败
        # (假设手臂碰盘子是允许的，如果不允许，把 p_id != plate_id 删掉)
        if arm_partners.size > 0:
            for p_id in arm_partners:
                if p_id != plate_id and p_id != floor_id:
                    results["any_arm_collision"] = True
                    break

        # # C. 外部物体判定（最关键）：只能碰到盘子
        # if object_partners.size > 0:
        #     for p_id in object_partners:
        #         # 如果碰到了不是盘子的东西（包括地面、底座、手臂等）
        #         if p_id != plate_id:
        #             results["object_failed"] = True
        #             break
            
        #     # 顺便判定物体是否当前正落在盘子上（可用于计算 Reward）
        #     if plate_id in object_partners:
        #         results["object_on_plate"] = True
        if object_partners.size > 0:
            has_success_contact = False
            has_edge_contact = False
            has_illegal_contact = False

            for p_id in object_partners:
                p_id = int(p_id)

                # 1. 碰到 frame_bottom：
                #    真正落盘，算 object_on_plate
                if p_id in plate_success_ids:
                    has_success_contact = True

                # 2. 碰到四个边：
                #    不算失败，但也不算真正落盘
                elif p_id in plate_edge_ids:
                    has_edge_contact = True

                # 3. 保险分支：
                #    如果以后 plate_allowed_ids 里加入了其他允许接触的盘子结构，
                #    这里也不算失败，但默认不算成功、不算边缘奖励
                elif p_id in plate_allowed_ids:
                    pass

                # 4. 碰到其他任何东西：
                #    floor / ranger_base / gripper / arm / 其他结构，都算失败
                else:
                    has_illegal_contact = True

                    partner_name = mujoco.mj_id2name(
                        self.Dcmm.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        p_id
                    )

                    partner_body_id = self.Dcmm.model.geom_bodyid[p_id]
                    partner_body_name = mujoco.mj_id2name(
                        self.Dcmm.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(partner_body_id)
                    )
                    if self.print_contacts:
                        print(
                            "[OBJECT FAILED CONTACT] "
                            f"partner_id={p_id}, "
                            f"partner_geom_name={partner_name}, "
                            f"partner_body_id={int(partner_body_id)}, "
                            f"partner_body_name={partner_body_name}, "
                            f"object_z={self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]:.3f}"
                        )

            # 只有碰到 frame_bottom，才算真正落盘
            if has_success_contact:
                results["object_on_plate"] = True

            # 只碰到四个边缘，记录为 object_on_edge
            # 注意：object_on_edge 不会增加 success_counter
            if has_edge_contact:
                results["object_on_edge"] = True

            # 碰到非法物体，判失败
            if has_illegal_contact:
                results["object_failed"] = True
        return results
    def _get_base_vel(self):
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        #获取机器人底座的 yaw（朝向）世界坐标系
        global_base_vel = self.Dcmm.data.qvel[0:2]#读取底座的线速度（世界坐标系）直接从data中获取的数据都是相对于世界坐标系来说的
        base_vel_x = math.cos(base_yaw) * global_base_vel[0] + math.sin(base_yaw) * global_base_vel[1]
        base_vel_y = -math.sin(base_yaw) * global_base_vel[0] + math.cos(base_yaw) * global_base_vel[1]
        return np.array([base_vel_x, base_vel_y])#机器人前进后退速度，左右速度，这个速度是相对于机器人自身来说的
    def _get_base_vel_copy(self):
        base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link_copy").xquat[3])
        #获取机器人底座的 yaw（朝向）世界坐标系
        global_base_vel = self.Dcmm.data.qvel[22:24]#读取底座的线速度（世界坐标系）直接从data中获取的数据都是相对于世界坐标系来说的
        base_vel_x = math.cos(base_yaw) * global_base_vel[0] + math.sin(base_yaw) * global_base_vel[1]
        base_vel_y = -math.sin(base_yaw) * global_base_vel[0] + math.cos(base_yaw) * global_base_vel[1]
        return np.array([base_vel_x, base_vel_y])#机器人前进后退速度，左右速度，这个速度是相对于机器人自身来说的

    def _get_relative_ee_pos3d(self):#ee：末端执行器
        # Caclulate the ee_pos3d w.r.t. the arm_base(原来写的是base_link)感觉原来写错了
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])#确定底座朝向，相对于世界坐标系
        x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
                                self.Dcmm.data.body("arm_seg6").xpos[0:2], 
                                base_yaw)#得到的 x, y 是 末端执行器在arm_base平面局部坐标系下的位置
        #x = arm_seg6 在 arm_base 坐标系中的前向距离
        #y = arm_seg6 在 arm_base 坐标系中的左向距离
        return np.array([x, y, 
                         self.Dcmm.data.body("arm_seg6").xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#末端执行器（link6）在垂直方向（Z轴）上相对于机械臂底座（arm_base）的高度差。
    def _get_relative_ee_pos3d_copy(self):#ee：末端执行器
        # Caclulate the ee_pos3d w.r.t. the arm_base(原来写的是base_link)感觉原来写错了
        base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link_copy").xquat[3])#确定底座朝向，相对于世界坐标系
        x,y = relative_position(self.Dcmm.data.body("arm_base_copy").xpos[0:2], 
                                self.Dcmm.data.body("arm_seg6_copy").xpos[0:2], 
                                base_yaw)#得到的 x, y 是 末端执行器在arm_base平面局部坐标系下的位置
        #x = arm_seg6 在 arm_base 坐标系中的前向距离
        #y = arm_seg6 在 arm_base 坐标系中的左向距离
        return np.array([x, y, 
                         self.Dcmm.data.body("arm_seg6_copy").xpos[2]-self.Dcmm.data.body("arm_base_copy").xpos[2]])
    
    def _get_relative_ee_quat(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        quat = relative_quaternion(self.Dcmm.data.body("base_link").xquat, self.Dcmm.data.body("arm_seg6").xquat)
        return np.array(quat)#在这里获取base_link的位姿是因为arm_base的位姿和base_link的位姿完全相同 
    def _get_relative_ee_quat_copy(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        quat = relative_quaternion(self.Dcmm.data.body("base_link_copy").xquat, self.Dcmm.data.body("arm_seg6_copy").xquat)
        return np.array(quat)#在这里获取base_link的位姿是因为arm_base的位姿和base_link的位姿完全相同
    
    def _get_relative_ee_v_lin_3d(self):
        # Caclulate the ee_v_lin3d w.r.t. the base_link
        # In simulation, we can directly get the velocity of the end-effector
        base_vel = self.Dcmm.data.body("arm_base").cvel[3:6]
        global_ee_v_lin = self.Dcmm.data.body("arm_seg6").cvel[3:6]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        ee_v_lin_x = math.cos(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        ee_v_lin_y = -math.sin(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        # TODO: In the real world, we can only estimate it by differentiating the position
        return np.array([ee_v_lin_x, ee_v_lin_y, global_ee_v_lin[2]-base_vel[2]])#机械臂末端相对于底座的速度
    def _get_relative_ee_v_lin_3d_copy(self):
        # Caclulate the ee_v_lin3d w.r.t. the base_link
        # In simulation, we can directly get the velocity of the end-effector
        base_vel = self.Dcmm.data.body("arm_base_copy").cvel[3:6]
        global_ee_v_lin = self.Dcmm.data.body("arm_seg6_copy").cvel[3:6]
        base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link_copy").xquat[3])
        ee_v_lin_x = math.cos(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        ee_v_lin_y = -math.sin(base_yaw) * (global_ee_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_ee_v_lin[1]-base_vel[1])
        # TODO: In the real world, we can only estimate it by differentiating the position
        return np.array([ee_v_lin_x, ee_v_lin_y, global_ee_v_lin[2]-base_vel[2]])#机械臂末端相对于底座的速度
    
    # def _get_relative_object_pos3d(self):
    #     # Caclulate the object_pos3d w.r.t. the base_link
    #     base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])#底座在水平平面上的旋转角度。
    #     x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
    #                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
    #                             base_yaw)#物体在机械臂基座（arm_base）坐标系下的相对位置，但只针对 平面 XY 方
    #     return np.array([x, y, 
    #                      self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#物体在 Z 方向（垂直方向）相对于机械臂基座的高度差。
    #     #小球相对于arm_base的位置
    # def _get_relative_object_pos3d_copy(self):
    #     # Caclulate the object_pos3d w.r.t. the base_link
    #     base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link_copy").xquat[3])#底座在水平平面上的旋转角度。
    #     x,y = relative_position(self.Dcmm.data.body("arm_base_copy").xpos[0:2], 
    #                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
    #                             base_yaw)#物体在机械臂基座（arm_base）坐标系下的相对位置，但只针对 平面 XY 方
    #     return np.array([x, y, 
    #                      self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base_copy").xpos[2]])#物体在 Z 方向（垂直方向）相对于机械臂基座的高度差。
    #     #小球相对于arm_base的位置
    def _get_object_pos3d(self):
        # 获取物体在世界坐标系下的位置
        # 返回 [x, y, z]，单位是米
        return self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3].copy()
    def _get_relative_object_pos3d_copy(self):
        # 获取物体在世界坐标系下的位置
        # 返回 [x, y, z]，单位是米
        return self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3].copy()
    # def _get_relative_object_v_lin_3d(self):
    #     # Caclulate the object_v_lin3d w.r.t. the base_link
    #     base_vel = self.Dcmm.data.body("arm_base").cvel[3:6]#arm_base底座刚体在当前仿真状态下的数据，cvel代表速度[3:6]是线速度，[0:3]是角速度
    #     global_object_v_lin = self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3]#.joint(name) 返回 仿真中名为 name 的关节对象的数据，对于关节来说[0:3]是线速度
    #     base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])#返回该刚体在世界坐标系下的 四元数旋转，将 四元数中的 w 和 z 分量 转换为 Yaw 角
    #     #移动机器人底座在水平平面（XY 平面）上的旋转角，也就是 机器人“朝向”的角度。
    #     object_v_lin_x = math.cos(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_object_v_lin[1]-base_vel[1])
    #     object_v_lin_y = -math.sin(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_object_v_lin[1]-base_vel[1])#把物体的线速度从全局坐标系转换到机器人底座的局部坐标系
    #     return np.array([object_v_lin_x, object_v_lin_y, global_object_v_lin[2]-base_vel[2]])#物体相对于机器人底座在竖直方向（z轴）的线速度
    #     #小球相对于base_link的速度，其实就是相对于aem_base的速度
    # def _get_relative_object_v_lin_3d_copy(self):
    #     # Caclulate the object_v_lin3d w.r.t. the base_link
    #     base_vel = self.Dcmm.data.body("arm_base_copy").cvel[3:6]#arm_base底座刚体在当前仿真状态下的数据，cvel代表速度[3:6]是线速度，[0:3]是角速度
    #     global_object_v_lin = self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3]#.joint(name) 返回 仿真中名为 name 的关节对象的数据，对于关节来说[0:3]是线速度
    #     base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link_copy").xquat[3])#返回该刚体在世界坐标系下的 四元数旋转，将 四元数中的 w 和 z 分量 转换为 Yaw 角
    #     #移动机器人底座在水平平面（XY 平面）上的旋转角，也就是 机器人“朝向”的角度。
    #     object_v_lin_x = math.cos(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_object_v_lin[1]-base_vel[1])
    #     object_v_lin_y = -math.sin(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_object_v_lin[1]-base_vel[1])#把物体的线速度从全局坐标系转换到机器人底座的局部坐标系
    #     return np.array([object_v_lin_x, object_v_lin_y, global_object_v_lin[2]-base_vel[2]])#物体相对于机器人底座在竖直方向（z轴）的线速度
    def _get_relative_object_v_lin_3d(self):
        # 获取物体在世界坐标系下的线速度
        # 返回 [vx, vy, vz]，单位通常是 m/s
        return self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3].copy()
    def _get_relative_object_v_lin_3d_copy(self):
        # 获取物体在世界坐标系下的线速度
        # 返回 [vx, vy, vz]，单位通常是 m/s
        return self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3].copy()
    def _get_plate_pos(self):
        """
        获取盘子中心在世界坐标系下的位置。
        你现在 XML 里已经用了 site: frame_bottom_center。
        """
        return self.Dcmm.data.site("frame_bottom_center").xpos.copy()
    def _update_plate_target_from_object(self):
        """
        根据当前物体位置更新最终目标点。

        target 定义：
            x = object_x
            y = object_y
            z = object_z - catch_z_offset

        当前版本：
            不再裁剪 target_pos。
        """

        obj_pos = self.Dcmm.data.body(self.Dcmm.object_name).xpos.copy()

        target_pos = obj_pos.copy()
        target_pos[2] = obj_pos[2] - self.catch_z_offset

        # 不再裁剪目标高度
        # target_pos[2] = np.clip(target_pos[2], 0.0, 6)

        self.traj_target_plate_pos = target_pos.copy()

        return self.traj_target_plate_pos.copy()
    
    def _build_plate_waypoints(self):#会返回起点喝终点以及n个路径点的坐标,形状是 (num_midpoints+2, 3)
        """
        根据当前盘子起点和最终目标点生成 waypoint 路径。

        默认：
            start -> wp1 -> wp2 -> target

        其中 wp1/wp2 是 start-target 之间的中间点，
        并且 z 方向稍微抬高 path_lift_z。
        """

        start = self.traj_start_plate_pos.copy()
        target = self.traj_target_plate_pos.copy()

        num_midpoints = self.num_path_midpoints
        lift = float(getattr(self, "path_lift_z", 0.08))

        waypoints = [start.copy()]

        for i in range(num_midpoints):
            # 如果 num_midpoints=2:
            # i=0 -> ratio=1/3
            # i=1 -> ratio=2/3
            ratio = float(i + 1) / float(num_midpoints + 1)

            wp = start + ratio * (target - start)

            # 中间点 z 稍微抬高
            wp[2] += lift
            waypoints.append(wp.copy())

        waypoints.append(target.copy())

        self.plate_waypoints = np.stack(
            waypoints,
            axis=0
        ).astype(np.float64)

        return self.plate_waypoints.copy()
    # def _project_point_to_waypoint_path(self, point):#输入的参数是当前盘子中心的位置
    #     """
    #     把当前盘子位置投影到 waypoint 路径上。

    #     输入:
    #         point:
    #             当前盘子中心位置shape=(3,)

    #     返回:
    #         path_error:
    #             point 到整条 waypoint 路径的最近距离。

    #         path_progress:
    #             point 沿路径的归一化进度，范围 0~1。

    #         closest_point:
    #             路径上距离 point 最近的点。
    #     """

    #     if self.plate_waypoints is None or len(self.plate_waypoints) < 2:
    #         return 0.0, 0.0, point.copy()

    #     waypoints = self.plate_waypoints#之前函数返回的路径点坐标，形状是 (num_midpoints+2, 3)，包含起点、终点和中间点

    #     # 先计算每一段长度
    #     seg_lengths = []#保存两个路径点之间的长度
    #     total_len = 0.0#计算整条路径的总长度

    #     for i in range(len(waypoints) - 1):#遍历路径里的每一段线段
    #         a = waypoints[i]#路径的起点
    #         b = waypoints[i + 1]#当前线段的终点
    #         seg_len = float(np.linalg.norm(b - a))#计算a到b的长度
    #         seg_lengths.append(seg_len)
    #         total_len += seg_len

    #     total_len = max(total_len, 1e-6)

    #     best_dist = float("inf")
    #     best_progress_len = 0.0
    #     best_closest = waypoints[0].copy()

    #     accumulated_len = 0.0

    #     for i in range(len(waypoints) - 1):#遍历每一条线段，但是现在的线段都是两个相邻点之间的线段
    #         a = waypoints[i]
    #         b = waypoints[i + 1]
    #         ab = b - a#当前线段的方向向量

    #         seg_len = max(seg_lengths[i], 1e-6)#当前线段的长度

    #         # point 在当前线段上的投影比例 t
    #         t = float(np.dot(point - a, ab) / (seg_len ** 2))
    #         t = float(np.clip(t, 0.0, 1.0))#计算point在当前线段上的投影比例，表示point投影到线段上面以后占比是多少

    #         closest = a + t * ab#计算当前线段上距离 point 最近的点坐标
    #         dist = float(np.linalg.norm(point - closest))

    #         if dist < best_dist:
    #             best_dist = dist
    #             best_closest = closest.copy()
    #             best_progress_len = accumulated_len + t * seg_len

    #         accumulated_len += seg_lengths[i]

    #     path_progress = float(np.clip(best_progress_len / total_len, 0.0, 1.0))

    #     return best_dist, path_progress, best_closest
    def _project_point_to_waypoint_path(self, point):
        """
        旧版逻辑：point 投影到整条 waypoint 路径的最近点。

        当前修改版：
            不再投影到线段内部；
            每一段只考虑该线段的终点 b；
            最后选择所有线段终点中离 point 最近的那个点。

        返回:
            path_error:
                point 到最近 waypoint 终点的距离。

            path_progress:
                最近 waypoint 终点对应的路径进度，范围 0~1。

            closest_point:
                离 point 最近的 waypoint 终点。
        """

        if self.plate_waypoints is None or len(self.plate_waypoints) < 2:
            return 0.0, 0.0, point.copy()

        waypoints = self.plate_waypoints

        # 先计算每一段长度
        seg_lengths = []
        total_len = 0.0

        for i in range(len(waypoints) - 1):
            a = waypoints[i]
            b = waypoints[i + 1]
            seg_len = float(np.linalg.norm(b - a))
            seg_lengths.append(seg_len)
            total_len += seg_len

        total_len = max(total_len, 1e-6)

        best_dist = float("inf")
        best_progress_len = 0.0
        best_closest = waypoints[0].copy()

        accumulated_len = 0.0

        for i in range(len(waypoints) - 1):
            a = waypoints[i]
            b = waypoints[i + 1]

            seg_len = max(seg_lengths[i], 1e-6)

            # =========================================================
            # 修改点：
            # 不再计算 t，不再投影到线段内部。
            # 当前线段的参考点直接取终点 b。
            # =========================================================
            closest = b.copy()

            # 当前盘子位置 point 到当前线段终点 b 的距离
            dist = float(np.linalg.norm(point - closest))

            if dist < best_dist:
                best_dist = dist
                best_closest = closest.copy()

                # 因为 closest 是当前线段终点，
                # 所以进度就是当前线段结束时的累计路径长度
                best_progress_len = accumulated_len + seg_len

            accumulated_len += seg_lengths[i]

        path_progress = float(np.clip(best_progress_len / total_len, 0.0, 1.0))

        return best_dist, path_progress, best_closest

    def _reset_plate_trajectory(self):
        """
        每个 episode reset 时调用一次：
        生成一条从当前盘子位置到物体下方目标点的 waypoint 路径。

        注意：
        这里不再依赖 alpha 生成 reward 目标。
        alpha/ref 可以保留做 debug，但 reward 主要使用 waypoint path。
        """

        self.traj_total_steps = max(1, int(self.env_time * self.fps))
        self.traj_reach_steps = max(1, int(self.traj_total_steps * self.traj_reach_ratio))

        # 起点：当前盘子中心
        self.traj_start_plate_pos = self._get_plate_pos()

        # 终点：物体下方目标点
        self.traj_target_plate_pos = self._update_plate_target_from_object()

        # 不再裁剪目标高度
        # self.traj_target_plate_pos[2] = np.clip(
        #     self.traj_target_plate_pos[2],
        #     0.0,
        #     8
        # )

        # 生成 waypoint path
        if self.task == "Tracking":
            # Tracking 阶段保留 waypoint path
            self._build_plate_waypoints()

            self.traj_ref_plate_pos = self.traj_start_plate_pos.copy()

            plate_pos = self._get_plate_pos()
            path_error, path_progress, path_closest_point = self._project_point_to_waypoint_path(
                plate_pos
            )

            self.prev_traj_error = float(path_error)
            self.prev_path_progress = float(path_progress)

        else:
            # Catching 阶段不使用路径点
            self.plate_waypoints = None
            self.traj_ref_plate_pos = self.traj_target_plate_pos.copy()

            plate_pos = self._get_plate_pos()
            target_error = float(np.linalg.norm(self.traj_target_plate_pos - plate_pos))

            self.prev_traj_error = target_error
            self.prev_path_progress = 0.0

    def _traj_alpha(self):
        """
        计算当前轨迹插值比例 alpha，范围 0~1。

        alpha = 0：参考点在起点
        alpha = 1：参考点在终点

        这里用了 smoothstep，比线性插值更平滑：
        alpha = 3s^2 - 2s^3
        """
        s = np.clip(self.steps / float(self.traj_reach_steps), 0.0, 1.0)
        return s * s * (3.0 - 2.0 * s)


    def _traj_episode_progress(self):
        """
        当前 episode 进度，范围 0~1。
        """
        return np.clip(self.steps / float(self.traj_total_steps), 0.0, 1.0)


    def _get_ref_plate_pos(self):
        """
        当前 step 的参考轨迹点。

        ref = start + alpha * (target - start)
        """
        alpha = self._traj_alpha()
        self.traj_ref_plate_pos = (
            self.traj_start_plate_pos
            + alpha * (self.traj_target_plate_pos - self.traj_start_plate_pos)
        )
        return self.traj_ref_plate_pos.copy()


    def _get_traj_phase(self):
        """
        简单分三个阶段：
        0:粗跟踪阶段
        1:精确靠近阶段
        2:保持阶段
        """
        p = self._traj_episode_progress()

        if p < 0.55:
            return 0
        elif p < self.traj_reach_ratio:
            return 1
        else:
            return 2

    def _check_tracking_success(self, info):
        """
        静止物体 Tracking 任务的 success 判定。

        当前目标：
        1. 不要求物体一定在盘子里。
        2. 不要求 prediction_ok。
        3. 不要求 object_throw。
        4. 只要求盘子中心靠近最终目标点，也就是：
            target_error < self.traj_success_threshold
        5. 为了避免某一瞬间误判成功，需要连续保持 success_hold_steps 步。
        6. 同时要求没有 object_failed，没有 base collision。

        适用场景：
        - 物体是静止的；
        - 你的任务目标是让盘子移动到物体下方目标点附近；
        - 成功标准是进入你设置的距离门槛，而不是必须发生物体落盘接触。
        """

        # =========================================================
        # 1. 读取当前目标误差和轨迹进度
        # =========================================================
        target_error = float(info["target_error"])
        traj_error = float(info.get("traj_error", 0.0))
        traj_alpha = float(self._traj_alpha())

        # 盘子是否已经进入最终目标点附近
        near_target = target_error < self.traj_success_threshold

        # =========================================================
        # 2. 安全条件
        # =========================================================
        # 静止 tracking 任务里，只要没有物体失败、没有底盘碰撞，就认为 safe。
        safe = (
            not self.contacts.get("object_failed", False)
            and not self.contacts.get("any_base_collision", False)
        )

        # =========================================================
        # 3. 这些信息只记录，不作为当前 success 的强制条件
        # =========================================================
        object_on_plate = bool(self.contacts.get("object_on_plate", False))

        pred_valid = bool(getattr(self, "last_prediction_valid", False))
        too_late = bool(getattr(self, "last_prediction_too_late", False))
        prediction_ok = pred_valid and (not too_late)

        # =========================================================
        # 4. 时间 / 轨迹阶段条件
        # =========================================================
        # 防止刚 reset 后目标刚好很近，立刻误判成功。
        # 这里用 self.steps >= 8，而不是 steps_after_throw，
        # 因为静止 tracking 不依赖 object_throw。
        time_ok = self.steps >= 8

        # 你的 reward_success 当前是 alpha >= 0.99 时才给成功奖励。
        # 所以这里也用 0.99，让 success 判定和 reward_success 对齐。
        alpha_ok = traj_alpha >= 0.99

        # =========================================================
        # 5. 静止 tracking 的 raw_success
        # =========================================================
        # 核心修改：
        # 不再强制 self.object_throw。
        # 不再强制 object_on_plate。
        # 不再强制 prediction_ok。
        raw_success = (
            time_ok
            #and alpha_ok
            and near_target
            and safe
        )

        # =========================================================
        # 6. 连续保持计数
        # =========================================================
        if raw_success:
            self.success_counter += 1
        else:
            self.success_counter = 0

        tracking_success = self.success_counter >= self.success_hold_steps

        if tracking_success:
            self.has_success = True

        # =========================================================
        # 7. 写入 info，方便 TensorBoard / wandb 查看
        # =========================================================
        info["is_success"] = float(tracking_success)
        info["raw_success"] = float(raw_success)

        info["success_target_error"] = float(target_error)
        info["success_traj_error"] = float(traj_error)
        info["success_threshold"] = float(self.traj_success_threshold)

        info["success_traj_alpha"] = float(traj_alpha)
        info["success_near_target"] = float(near_target)
        info["success_safe"] = float(safe)
        info["success_counter"] = float(self.success_counter)

        info["success_time_ok"] = float(time_ok)
        info["success_alpha_ok"] = float(alpha_ok)

        # 下面这些只是观察指标，不参与当前 success 判定
        info["success_object_on_plate"] = float(object_on_plate)
        info["success_prediction_ok"] = float(prediction_ok)
        info["success_pred_valid"] = float(pred_valid)
        info["success_pred_too_late"] = float(too_late)
        info["success_object_throw"] = float(self.object_throw)

        # 额外调试项：
        # 用来看如果以后加 object_on_plate / prediction_ok，成功率会不会被卡住。
        info["success_raw_with_object_on_plate"] = float(
            raw_success and object_on_plate
        )

        info["success_raw_with_prediction"] = float(
            raw_success and prediction_ok
        )

        info["success_raw_with_object_and_prediction"] = float(
            raw_success and object_on_plate and prediction_ok
        )

        return tracking_success
    
    def _get_obs(self):
        """
        获取当前观测。

        关键点：
        1. 物体位置仍然使用相对于两个 arm_base 的相对坐标。
        2. 物体速度不再直接读取 MuJoCo qvel。
        3. 物体速度改成用前后两帧相对位置差分估计：
              v = (pos_now - pos_prev) / dt
        4. 第一帧没有上一帧，所以第一帧速度自然为 0。
        """

        # =========================================================
        # 1. 获取当前帧的末端、物体、盘子、轨迹点位置
        # =========================================================
        ee_pos3d = self._get_relative_ee_pos3d()
        ee_pos3d_copy = self._get_relative_ee_pos3d_copy()

        obj_pos3d = self._get_object_pos3d()
        obj_pos3d_copy = self._get_relative_object_pos3d_copy()


        # =========================================================
        # 2. 第一帧初始化 prev 位置
        # =========================================================
        # 第一次进入 _get_obs() 时，还没有真正的上一帧。
        # 所以把上一帧位置初始化为当前帧位置。
        # 这样第一帧速度 = 当前帧 - 当前帧 = 0，是合理的。
        if self.init_pos:
            self.prev_ee_pos3d = ee_pos3d.copy()
            self.prev_ee_pos3d_copy = ee_pos3d_copy.copy()

            self.prev_obj_pos3d = obj_pos3d.copy()
            self.prev_obj_pos3d_copy = obj_pos3d_copy.copy()

            self.init_pos = False

        # =========================================================
        # 3. 用前后两帧位置差分估计速度
        # =========================================================
        # 一个 policy step 的时间间隔：
        # self.fps = 1 / (steps_per_policy * mujoco_timestep)
        # 所以 dt = 1 / self.fps
        dt = max(1.0 / float(self.fps), 1e-6)

        ee_v_lin_3d_est = (ee_pos3d - self.prev_ee_pos3d) / dt
        ee_v_lin_3d_copy_est = (ee_pos3d_copy - self.prev_ee_pos3d_copy) / dt

        obj_v_lin_3d_est = (obj_pos3d - self.prev_obj_pos3d) / dt
        obj_v_lin_3d_copy_est = (obj_pos3d_copy - self.prev_obj_pos3d_copy) / dt
        plate_pos = self._get_plate_pos()

        # 统一更新最终目标点
        # obs / info / reward / success 都应该使用这个 self.traj_target_plate_pos
        target_plate_pos = self._update_plate_target_from_object()

        # ref 只在环境内部使用，不放进 obs
        # waypoint path 不需要在 obs 里提供 ref。
        # 这里保留一个 ref 变量只是为了兼容后面的调试代码。
        if self.plate_waypoints is not None:
            _, _, ref_plate_pos = self._project_point_to_waypoint_path(plate_pos)
        else:
            ref_plate_pos = plate_pos.copy()

        # obs 只放最终目标点相关信息
        plate_target_error = target_plate_pos - plate_pos

        plate_target_distance = np.array(
            [np.linalg.norm(plate_target_error)],
            dtype=np.float32
        )
        # =========================================================
        # 4. 轨迹观测
        # =========================================================
        # 当前轨迹点 - 当前盘子位置
        # 这个量告诉策略：盘子应该往哪个方向运动。
        plate_ref_error = ref_plate_pos - plate_pos
        plate_target_error = target_plate_pos - plate_pos
        # 两个 bar 的高度差，和 compute_reward() 里的 reward_bar_level 保持一致
        bar1_z = float(self.Dcmm.data.body("bar_left").xpos[2])
        bar2_z = float(self.Dcmm.data.body("bar_right").xpos[2])

        bar_height_error = np.array(
            [abs(bar1_z - bar2_z)],
            dtype=np.float32
        )
        # =========================================================
        # 两个底盘的位置和距离
        # =========================================================
        base1_xy = self.Dcmm.data.body("base_link").xpos[:2].copy()
        base2_xy = self.Dcmm.data.body("base_link_copy").xpos[:2].copy()

        # 两个底盘的位置，4 维：
        # [base1_x, base1_y, base2_x, base2_y]
        base_pair_pos = np.concatenate(
            [base1_xy, base2_xy],
            axis=0
        ).astype(np.float32)

        # 两个底盘之间的 xy 平面距离，1 维
        base_pair_distance = np.array(
            [np.linalg.norm(base1_xy - base2_xy)],
            dtype=np.float32
        )
        traj_alpha = np.array(
            [float(self._traj_alpha())],
            dtype=np.float32
        )

        time_to_catch = np.array(
            [float(np.clip(getattr(self, "last_pred_t_hit", 0.0), 0.0, 6.0))],
            dtype=np.float32
        )

        object_throw_obs = np.array(
            [float(self.object_throw)],
            dtype=np.float32
        )

        traj_progress = np.array(
            [self._traj_episode_progress()],
            dtype=np.float32
        )

        # 如果你后面还想用位置历史，可以保留这一行。
        # 目前 PPO 输入里没有用 pos_history，所以它只是备用。
        self.obj_pos_history.append(obj_pos3d.copy())
        object_pos_now = self.Dcmm.data.body(self.Dcmm.object_name).xpos.copy()

        # 当前盘子位置
        plate_pos_now = self._get_plate_pos()

        # 物体相对盘子的误差
        plate_target_error = object_pos_now - plate_pos_now
        # =========================================================
        # 5. 构造 obs 字典
        # =========================================================
        obs = {
            "base1": {
                "v_lin_3d": (
                    self._get_base_vel()
                    + np.random.normal(0, self.k_obs_base, 2)
                ),
                "base_pos": (
                    self.Dcmm.data.body("arm_base").xpos[0:3]
                    + np.random.normal(0, self.k_obs_base, 3)
                )
            },

            "base2": {
                "v_lin_3d": (
                    self._get_base_vel_copy()
                    + np.random.normal(0, self.k_obs_base, 2)
                ),
                "base_pos": (
                    self.Dcmm.data.body("arm_base_copy").xpos[0:3]
                    + np.random.normal(0, self.k_obs_base, 3)
                )   
            },

            "arm1": {
                "ee_pos3d": (
                    ee_pos3d
                    + np.random.normal(0, self.k_obs_arm, 3)
                ),

                "ee_quat": (
                    self._get_relative_ee_quat()
                    + np.random.normal(0, self.k_obs_arm, 4)
                ),

                # 这里也使用前后两帧位置差分速度，
                # 和物体速度估计方式保持一致。
                "ee_v_lin_3d": (
                    ee_v_lin_3d_est
                    + np.random.normal(0, self.k_obs_arm, 3)
                ),

                "joint_pos": (
                    np.array(self.Dcmm.data.qpos[15:21])
                    + np.random.normal(0, self.k_obs_arm, 6)
                ),
            },

            "arm2": {
                "ee_pos3d": (
                    ee_pos3d_copy
                    + np.random.normal(0, self.k_obs_arm, 3)
                ),

                "ee_quat": (
                    self._get_relative_ee_quat_copy()
                    + np.random.normal(0, self.k_obs_arm, 4)
                ),

                "ee_v_lin_3d": (
                    ee_v_lin_3d_copy_est
                    + np.random.normal(0, self.k_obs_arm, 3)
                ),

                "joint_pos": (
                    np.array(self.Dcmm.data.qpos[38:44])
                    + np.random.normal(0, self.k_obs_arm, 6)
                ),
            },

            "object": {
                "pos3d": (
                    obj_pos3d
                    + np.random.normal(0, self.k_obs_object, 3)
                ),

                # 重要：
                # 这里不再使用 self._get_relative_object_v_lin_3d()
                # 因为那个函数内部直接读取 MuJoCo qvel。
                # 现在改成前后两帧相对位置差分估计速度。
                "v_lin_3d": (
                    obj_v_lin_3d_est
                    + np.random.normal(0, self.k_obs_object, 3)
                ),

            },
            # "plate": {
            #     "plate_pos": plate_pos_now.astype(np.float32),
            # }
            "plate": {
                "plate_pos": plate_target_error.astype(np.float32),
            }#d底盘位置，速度，ee pos,ee vel,joint pos,obj pos vel,
                # "trajectory": {
                #     "plate_ref_error": (
                #         plate_ref_error.astype(np.float32)
                #         + np.random.normal(0, self.k_obs_object, 3)
                #     ),

                #     # "plate_target_error": (
                #     #     plate_target_error.astype(np.float32)
                #     #     + np.random.normal(0, self.k_obs_object, 3)
                #     # ),

                #     # "traj_alpha": traj_alpha,

                #     # "time_to_catch": time_to_catch,

                #     # "object_throw": object_throw_obs,

                #     "progress": traj_progress,
                # },
        }

        # =========================================================
        # 6. 最后更新上一帧位置
        # =========================================================
        # 注意：一定要放在速度计算之后。
        # 否则如果先更新 prev，再计算速度，就会变成：
        # 当前帧 - 当前帧 = 0
        self.prev_ee_pos3d = ee_pos3d.copy()
        self.prev_ee_pos3d_copy = ee_pos3d_copy.copy()

        self.prev_obj_pos3d = obj_pos3d.copy()
        self.prev_obj_pos3d_copy = obj_pos3d_copy.copy()

        if self.print_obs:
            print("##### print obs: \n", obs)

        return obs

    def _get_info(self):
        # Time of the Mujoco environment
        env_time = self.Dcmm.data.time - self.start_time
        ee_distance = np.linalg.norm(self.Dcmm.data.body("arm_seg6").xpos - 
                                    self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        base_distance = np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                        self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        base_distance_copy = np.linalg.norm(self.Dcmm.data.body("arm_base_copy").xpos[0:2] -
                                        self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        # print("base_distance: ", base_distance)
        qpos1 = self.Dcmm.data.joint("gripper1_axis").qpos[0]
        qpos2 = self.Dcmm.data.joint("gripper2_axis").qpos[0]
        qpos_sum = qpos1 + qpos2
        plate_distance=np.linalg.norm(self.Dcmm.data.site("frame_bottom_center").xpos - self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        plate_z_distance = np.abs(self.Dcmm.data.site("frame_bottom_center").xpos[2] - 
                      self.Dcmm.data.body(self.Dcmm.object_name).xpos[2])
        ee_position=self.Dcmm.data.body("arm_seg6").xpos,
        ee_position_copy=self.Dcmm.data.body("arm_seg6_copy").xpos
        plate_pos = self._get_plate_pos()
        base1_xy = self.Dcmm.data.body("base_link").xpos[:2].copy()
        base2_xy = self.Dcmm.data.body("base_link_copy").xpos[:2].copy()

        base_pair_pos = np.concatenate(
            [base1_xy, base2_xy],
            axis=0
        ).astype(np.float32)

        base_pair_distance = float(np.linalg.norm(base1_xy - base2_xy))


        # 和 obs 使用同一个最终目标点
        target_plate_pos = self._update_plate_target_from_object()

        target_error = float(np.linalg.norm(plate_pos - target_plate_pos))

        if self.task == "Tracking":
            # Tracking 阶段才使用 waypoint path
            self._build_plate_waypoints()

            path_error, path_progress, path_closest_point = self._project_point_to_waypoint_path(
                plate_pos
            )

            ref_plate_pos = path_closest_point.copy()
            traj_error = float(path_error)

        else:
            # Catching 阶段不使用路径点
            path_error = target_error
            path_progress = 0.0
            path_closest_point = target_plate_pos.copy()

            ref_plate_pos = target_plate_pos.copy()
            traj_error = target_error
        
        if self.print_info: 
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
        return {
            # Get contact point from the mujoco model
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
            "base_distance_copy": base_distance_copy,
            "qpos_sum": qpos_sum,
            "plate_distance":plate_distance,
            "plate_z_distance":plate_z_distance,
            "ee_position":self.Dcmm.data.body("arm_seg6").xpos,
            "ee_position_copy":self.Dcmm.data.body("arm_seg6_copy").xpos,
            "plate_pos": plate_pos,
            "plate_ref_pos": ref_plate_pos,
            "plate_target_pos": target_plate_pos,

            # traj_error 现在表示 path_error，不再是 alpha-ref error
            "traj_error": traj_error,
            "path_error": float(path_error),
            "path_progress": float(path_progress),
            "path_closest_point": path_closest_point,

            "target_error": target_error,
            "traj_progress": self._traj_episode_progress(),
            "traj_phase": self._get_traj_phase(),

            "plate_target_error": target_plate_pos - plate_pos,
            "plate_target_distance": target_error,
            "base_pair_pos": base_pair_pos,
            "base_pair_distance": base_pair_distance,

        }
    
    def update_target_ctrl(self):#把当前这一时刻的目标控制量（底盘速度、机械臂关节位置、夹爪关节位置）存进一个“动作缓冲区”
        self.action_buffer["base"].append(copy.deepcopy(self.Dcmm.target_base_vel[:]))
        self.action_buffer["arm"].append(copy.deepcopy(self.Dcmm.target_arm_qpos[:]))
        self.action_buffer["hand"].append(copy.deepcopy(self.Dcmm.target_hand_qpos[:]))
        self.action_buffer["base_copy"].append(copy.deepcopy(self.Dcmm.target_base_vel_copy[:]))
        self.action_buffer["arm_copy"].append(copy.deepcopy(self.Dcmm.target_arm_qpos_copy[:]))
        self.action_buffer["hand_copy"].append(copy.deepcopy(self.Dcmm.target_hand_qpos_copy[:]))

    def _get_ctrl(self):#把强化学习输出的 action 信号 → 转换成底盘、机械臂、夹爪的真实控制量（ctrl）
        #作用：把缓冲区里的目标量（速度 / 关节角）转换成 MuJoCo 中 data.ctrl 的 16 维控制向量。
        # Map the action to the control 
        mv_steer, mv_drive = self.Dcmm.move_base_vel(self.action_buffer["base"][0]) # 8 mv_steer = [steer_fl, steer_fr, steer_rl, steer_rr]
        mv_steer_copy, mv_drive_copy = self.Dcmm.move_base_vel_copy(self.action_buffer["base_copy"][0])
        mv_arm = self.Dcmm.arm_pid.update(self.action_buffer["arm"][0], self.Dcmm.data.qpos[15:21], self.Dcmm.data.time) # 6
        mv_arm_copy = self.Dcmm.arm_pid_copy.update(self.action_buffer["arm_copy"][0], self.Dcmm.data.qpos[38:44], self.Dcmm.data.time)
        #mv_hand = self.Dcmm.hand_pid.update(self.action_buffer["hand"][0], self.Dcmm.data.qpos[21:23], self.Dcmm.data.time) # 16
        mv_hand = self.action_buffer["hand"][0] #
        mv_hand_copy = self.action_buffer["hand_copy"][0]
        #########################################################################################################################################
        #ctrl = np.concatenate([mv_steer, mv_drive, mv_steer_copy, mv_drive_copy,mv_arm, mv_arm_copy, mv_hand, mv_hand_copy], axis=0)#得到的是控制的力，但是从神经网络里面输出的是位移和角度
        #ctrl[:32] *= np.random.normal(1, self.k_act, 32)
        ###########################################################################################################################
        # mv_steer = np.clip(mv_steer, -7.5, 7.5)
        # mv_steer_copy = np.clip(mv_steer_copy, -7.5, 7.5)
        # mv_drive = np.clip(mv_drive, -40.19, 40.19)
        # mv_drive_copy = np.clip(mv_drive_copy, -40.19, 40.19)  
        # mv_arm = np.clip(mv_arm, -100, 100)
        # mv_arm_copy = np.clip(mv_arm_copy, -100, 100)、
        mv_steer = np.clip(mv_steer, -10, 10)
        mv_steer_copy = np.clip(mv_steer_copy, -10, 10)
        mv_drive = np.clip(mv_drive, -50, 50)
        mv_drive_copy = np.clip(mv_drive_copy, -50, 50)  
        mv_arm = np.clip(mv_arm, -100, 100)
        mv_arm_copy = np.clip(mv_arm_copy, -100, 100)
        ctrl_list = [
                np.array(mv_steer), np.array(mv_drive),           # 0-7
                np.array(mv_steer_copy), np.array(mv_drive_copy), # 8-15
                np.array(mv_arm),                                # 16-21
                np.array(mv_arm_copy),                           # 22-27
                np.array(mv_hand),                               # 28-29
                np.array(mv_hand_copy)                            # 30-31
            ]
            
    # 执行拼接
        ctrl = np.concatenate(ctrl_list, axis=0).astype(np.float64)
        ctrl[:32] *= np.random.normal(1, self.k_act, 32)
        # 3. 【关键】捕获第一帧的 NaN 并报告源头
        if np.isnan(ctrl).any():
            print(f"\n!!! 发现 NaN !!! 时间: {self.Dcmm.data.time}")
            names = ["steer", "drive", "steer_c", "drive_c", "arm", "arm_c", "hand", "hand_c"]
            for name, arr in zip(names, ctrl_list):
                if np.isnan(arr).any():
                    print(f"源头是 -> {name}: {arr}")
            
            # 防御性修复：将 NaN 替换为 0，防止仿真崩溃
            ctrl = np.nan_to_num(ctrl)

        # 4. 噪声处理 (先确保 ctrl 没问题再加噪声)
        if self.k_act > 0:
            noise = np.random.normal(1, self.k_act, ctrl.shape)
            ctrl *= noise
        # ctrl *= np.random.normal(1, self.k_act, 16)#给 16 维的控制力矩信号 乘上一个随机因子（高斯噪声）
        if self.print_ctrl:
            print("##### ctrl:")
            print("mv_steer: {}, \nmv_drive: {}, \nmv_arm: {}, \nmv_hand: {}\n".format(mv_steer, mv_drive, mv_arm, mv_hand))
        return ctrl

    def _reset_object(self):#在每次环境 reset 时，重新随机生成一个新的“抛物体 object”
        # Parse the XML string
        root = ET.fromstring(self.Dcmm.model_xml_string)#把 XML 字符串解析成一个树状结构

        # Find the <body> element with name="object"
        object_body = root.find(".//body[@name='object']")#查找 <body> 标签中 name="object" 的节点，object_body中是整个body的内容
        geom = object_body.find(".//geom[@name='object']")
        if geom is not None:
            # print("\n========== [XML OBJECT BEFORE] ==========")
            # print("object_train =", self.object_train)
            # print("type =", geom.attrib.get("type"))
            # print("size =", geom.attrib.get("size"))
            # print("mesh =", geom.attrib.get("mesh"))
            # print("=========================================\n")
            if self.task == "Tracking":
                # 1. 把 object 改成一个小球，作为可视化目标点
                geom.set("type", "sphere")

                # 2. 设置小球半径
                # 0.01 表示 1cm，如果你想看得更明显，可以改成 0.02 或 0.03
                geom.set("size", "0.01")

                # 3. 如果原来 object 是 mesh，要删除 mesh 属性
                # 否则 MuJoCo 可能仍然按 mesh 加载
                if "mesh" in geom.attrib:
                    del geom.attrib["mesh"]

                # 4. 关键：关闭碰撞
                # contype=0 和 conaffinity=0 表示这个 geom 不参与任何碰撞
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

                # 5. 设置颜色，红色半透明，方便 viewer 里面看到目标点
                geom.set("rgba", "1 0 0 0.5")
                # print(
                #     "\n========== [TRACKING OBJECT AS POINT ENABLED] ==========\n"
                #     f"task = {self.task}\n"
                #     f"type = {geom.attrib.get('type')}\n"
                #     f"size = {geom.attrib.get('size')}\n"
                #     f"mesh = {geom.attrib.get('mesh')}\n"
                #     f"contype = {geom.attrib.get('contype')}\n"
                #     f"conaffinity = {geom.attrib.get('conaffinity')}\n"
                #     f"rgba = {geom.attrib.get('rgba')}\n"
                #     "object will be used as a visual target point without collision.\n"
                #     "=======================================================\n"
                # )

                # 6. Tracking 阶段直接返回 XML
                # 不再执行下面随机 shape / mesh 逻辑
                xml_str = ET.tostring(root, encoding='unicode')
                return xml_str
            object_id = np.random.choice([0, 1, 2, 3, ])

            if self.object_train:
                object_shape = DcmmCfg.object_shape[object_id]
                geom.set("type", object_shape)

                object_size = np.array([
                    np.random.uniform(low=low, high=high)
                    for low, high in DcmmCfg.object_size[object_shape]
                ])
                geom.set("size", np.array_str(object_size)[1:-1])

                if "mesh" in geom.attrib:
                    del geom.attrib["mesh"]

                # print("\n========== [OBJECT TRAIN GEOM BRANCH] ==========")
                # print("object_id =", object_id)
                # print("selected type =", object_shape)
                # print("selected size =", object_size)
                # print("===============================================\n")

            else:
                object_mesh = DcmmCfg.object_mesh[object_id]

                # 先只打印，不改逻辑，用来确认你现在为什么一直是 box
                geom.set("mesh", object_mesh)

            #     print("\n========== [OBJECT EVAL MESH BRANCH] ==========")
            #     print("object_id =", object_id)
            #     print("selected mesh =", object_mesh)
            #     print("type after set mesh =", geom.attrib.get("type"))
            #     print("size after set mesh =", geom.attrib.get("size"))
            #     print("mesh after set mesh =", geom.attrib.get("mesh"))
            #     print("==============================================\n")

            # print("\n========== [XML OBJECT AFTER] ==========")
            # print("type =", geom.attrib.get("type"))
            # print("size =", geom.attrib.get("size"))
            # print("mesh =", geom.attrib.get("mesh"))
            # print("========================================\n")
            # Find the <geom> element
            geom = object_body.find(".//geom[@name='object']")
            if geom is not None:#随机化集合外形
                # Modify the type and size attributes
                object_id = np.random.choice([0, 1, 2, 3, ])#四个数随机选一个
                if self.object_train:#如果是训练模式，随机生成几何体，非训练模式就用已经有的mesh文件，随机生成的几何体是mujoco可以自己生成的，只需要简单的参数就可以生成
                    object_shape = DcmmCfg.object_shape[object_id]
                    geom.set("type", object_shape)  # Replace "box" with the desired type
                    object_size = np.array([np.random.uniform(low=low, high=high) for low, high in DcmmCfg.object_size[object_shape]])
                    geom.set("size", np.array_str(object_size)[1:-1])  # Replace with the desired size
                else:
                    object_mesh = DcmmCfg.object_mesh[object_id]
                    geom.set("mesh", object_mesh)
        xml_str = ET.tostring(root, encoding='unicode')#ET.tostring() 的作用是 把 XML 树（ElementTree 节点对象）转换为字符串。
        
        return xml_str#xml树的str
        #训练模式时随机生成object，评估的时候用固定的object
    def random_object_pose(self):
        throw_from_right = np.random.rand() < 0.5

        if True:
            # ======================================================
            # 情况 1：从右侧扔，并且往中间飞
            #
            # 初始位置：
            #   y > 0
            #
            # 横向速度：
            #   v_lin_y < 0
            #   从 y 正方向往 y=0 中间靠
            
            
            #下面这个是正常的
            # ======================================================

            # # 初始位置
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(2.7, 3.1)
            height = np.random.uniform(0.8, 1.2)
            r_vel = 1 + np.random.rand() # (1, 2)
            alpha_vel = math.pi * (np.random.rand()*1/6 + 5/12) # alpha_vel = (5/12 * pi, 7/12 * pi)
            # alpha_vel = math.pi * (np.random.rand()*1/3 + 1/3) # alpha_vel = (1/3 * pi, 2/3 * pi)
            v_lin_x = r_vel * math.cos(alpha_vel) # (-0.0, -0.5)
            v_lin_y =np.random.uniform(-1.8, -1.3)
            #v_lin_y = - r_vel * math.sin(alpha_vel) # (-2, -1)
            # 初始速度
            # v_lin_x = np.random.uniform(-0.3, 0.3)
            # v_lin_y = np.random.uniform(-1.2, -0.8)
            v_lin_z = np.random.uniform(2.2, 2.5)
            if y > 2.85: v_lin_y -= 0.5
            if height < 1.0: v_lin_z += 0.8
            # ==========================================================




            # 慢速 Catching：
            # 目标：先让策略学会根据当前物体坐标移动盘子
            # ==========================================================

            # 初始位置先放近一点，不要一开始离得太远
            # x = np.random.uniform(-0.20, 0.20)
            # y = np.random.uniform(0.5, 0.6)
            # height = np.random.uniform(0.90, 1.20)

            # v_lin_x = np.random.uniform(-0.0, 0.0)
            # v_lin_y = np.random.uniform(-0.0, -0.0)
            # v_lin_z = np.random.uniform(0.10, 0.2)
            # x = np.random.uniform(-0.20, 0.20)
            # y = np.random.uniform(0.9, 1.2)
            # height = np.random.uniform(0.90, 1.20)

                #print("catching")

        # else:
        #     # ======================================================
        #     # 情况 2：从左侧扔，并且往中间飞
        #     #
        #     # 初始位置：
        #     #   y < 0
        #     #
        #     # 横向速度：
        #     #   v_lin_y > 0
        #     #   从 y 负方向往 y=0 中间靠
        #     # ======================================================

        #     # 初始位置
        #     x = np.random.uniform(-2.2, -2.5)
        #     y = np.random.uniform(0.8, 0.3)
        #     height = np.random.uniform(1.2, 1.5)

        #     # 初始速度
        #     v_lin_x = np.random.uniform(2.5, 2.0)
        #     v_lin_y = np.random.uniform(0.0, 0.7)
        #     v_lin_z = np.random.uniform(2.9, 3.2)


        # ==========================================================
        # 写入物体初始位置和速度
        # ==========================================================

        self.object_pos3d = np.array([x, y, height])

        if self.task == "Catching":
            self.object_vel6d = np.array([
                v_lin_x,
                v_lin_y,
                v_lin_z,
                0.0,
                0.0,
                0.0
            ])
        else:
            self.object_vel6d = np.array([
                0,
                0,
                0,
                0.0,
                0.0,
                0.0
            ])
            x = np.random.uniform(-0.5, 0.5)
            y = np.random.uniform(1.2, 2)
            height = np.random.uniform(0.65, 0.9)
            #print("tracking")
            self.object_pos3d = np.array([x, y, height])
        # if self.task == "Catching":
        #     self.object_vel6d = np.array([v_lin_x, v_lin_y, v_lin_z, 0.0, 0.0, 0.0])
        # else:
        #     self.object_vel6d = np.array([0, 0, 0, 0.0, 0.0, 0.0])
# ================== 【新增修改开始】 ==================
        # 1. 随机决定轨迹类型：50% 直线 (linear)，50% 曲线 (curve)
        self.trajectory_type = np.random.choice(['throw', 'curve','circle'], p=[1.0, 0,0])

        if self.trajectory_type == 'curve':
            # 2. 如果是曲线，必须在这里生成“参数”，否则后面的 step 函数不知道该怎么画曲线
            
            # 随机振幅 (Amplitude)：决定弯曲的程度
            self.curve_amp = np.random.uniform(0.05, 0.35) 
            
            # 随机频率 (Frequency)：决定摆动的快慢
            self.curve_freq = np.random.uniform(1.0, 3.0) 
            
            # 随机相位 (Phase)：决定正弦波从哪里开始 (避免每次都从0开始)
            self.curve_phase = np.random.uniform(0, 2 * np.pi)
            
            # 随机轴向：决定是在 Y轴(左右) 还是 Z轴(上下) 做曲线运动
            self.curve_axis = np.random.choice(['y', 'z']) 
            
            #print(f"Trajectory: Curve | Axis: {self.curve_axis} | Amp: {self.curve_amp:.2f} | Freq: {self.curve_freq:.2f}")
        if self.trajectory_type == 'circle':
                # 旋转半径：决定物体离车多远旋转
                self.circle_radius = np.random.uniform(0.6, 0.7) 
                # 旋转角速度：决定转多快 (rad/s)
                self.circle_omega = np.random.uniform(0.5, 1.0) * np.random.choice([-1, 1]) 
                # 初始角度：随机起始位置
                self.circle_start_angle = np.random.uniform(0.6 * math.pi, 0.6 * math.pi)
                # 旋转中心：设置为机器人的大致位置 (0, 0) 或者机械臂基座位置
                self.circle_center = np.array([0.0, 0.0])
        # else:
        #     # 如果是直线，不需要额外参数，只需打印确认即可
        #     print("Trajectory: Linear")
        # ================== 【新增修改结束】 ==================        
        # Random Static Time
        self.object_static_time = np.random.uniform(DcmmCfg.object_static[0], DcmmCfg.object_static[1])
        # Random Quaternion
        # r_obj_quat = R.from_euler('xyz', [0, np.random.rand()*1*math.pi, 0], degrees=False)
        # self.object_q = r_obj_quat.as_quat()
        fixed_obj_quat = R.from_euler('xyz', [0, math.pi/2, 0], degrees=False)
        # 转换为四元数
        self.object_q = fixed_obj_quat.as_quat()

    
    def random_PID(self):
        # Random the PID Controller Params in DCMM在每个 episode以及reset() 开始时，随机化底座、机械臂、夹爪的 PID 参数。
        self.k_arm = np.random.uniform(0, 1, size=6)
        self.k_drive = np.random.uniform(0, 1, size=4)
        self.k_steer = np.random.uniform(0, 1, size=4)
        self.k_hand = np.random.uniform(0, 1, size=2)
        # Reset the PID Controller
        self.Dcmm.arm_pid.reset(self.k_arm*(DcmmCfg.k_arm[1]-DcmmCfg.k_arm[0])+DcmmCfg.k_arm[0])
        self.Dcmm.steer_pid.reset(self.k_steer*(DcmmCfg.k_steer[1]-DcmmCfg.k_steer[0])+DcmmCfg.k_steer[0])
        self.Dcmm.drive_pid.reset(self.k_drive*(DcmmCfg.k_drive[1]-DcmmCfg.k_drive[0])+DcmmCfg.k_drive[0])
        self.Dcmm.hand_pid.reset(self.k_hand[0]*(DcmmCfg.k_hand[1]-DcmmCfg.k_hand[0])+DcmmCfg.k_hand[0])
    #只随机p因为p比较稳定，id不稳定所以就直接给定id，id在整个过程中始终不变
    '''
    # Define the Randomization Params
    ## Wheel Drive
    k_drive = np.array([0.75, 1.25])
    ## Wheel Steer
    k_steer = np.array([0.75, 1.25])
    ## Arm Joints
    k_arm = np.array([0.75, 1.25])
    ## Hand Joints
    k_hand = np.array([0.75, 1.25])
    ## Object Shape and Size
    '''
    def random_delay(self):#随机动作延迟模，拟了现实中“控制信号延迟”的效果
        # Random the Delay Buffer Params in DCMM
        base_delay = max(1, np.random.choice(DcmmCfg.act_delay['base']))
        arm_delay = max(1, np.random.choice(DcmmCfg.act_delay['arm']))
        hand_delay = max(1, np.random.choice(DcmmCfg.act_delay['hand']))
        # self.action_buffer["base"].set_maxlen(np.random.choice(DcmmCfg.act_delay['base']))
        # self.action_buffer["arm"].set_maxlen(np.random.choice(DcmmCfg.act_delay['arm']))
        # self.action_buffer["hand"].set_maxlen(np.random.choice(DcmmCfg.act_delay['hand']))
        self.action_buffer["base"].set_maxlen(base_delay)
        self.action_buffer["arm"].set_maxlen(arm_delay)
        self.action_buffer["hand"].set_maxlen(hand_delay)
        self.action_buffer["base_copy"].set_maxlen(base_delay)
        self.action_buffer["arm_copy"].set_maxlen(arm_delay)
        self.action_buffer["hand_copy"].set_maxlen(hand_delay)
        # Clear Buffer
        self.action_buffer["base"].clear()
        self.action_buffer["arm"].clear()
        self.action_buffer["hand"].clear()
        self.action_buffer["base_copy"].clear()
        self.action_buffer["arm_copy"].clear()
        self.action_buffer["hand_copy"].clear()

    def _reset_simulation(self):#重置并随机化整个仿真
        # Reset the data in Mujoco Simulation
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)#把data全部重置为 默认初始状态，上面也是，下面的是单独的机械臂模型
        mujoco.mj_resetData(self.Dcmm.model_arm_copy, self.Dcmm.data_arm_copy)
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        if self.Dcmm.model_arm_copy.na == 0:
            self.Dcmm.data_arm_copy.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)#将所有控制都清零
        self.Dcmm.data_arm_copy.ctrl = np.zeros(self.Dcmm.model_arm_copy.nu)
        self.Dcmm.data.qpos[15:21] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[38:44] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[21:23] = DcmmCfg.hand_joints[:]#手被重置
        self.Dcmm.data.qpos[44:46] = DcmmCfg.hand_joints[:]
        self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]#把关节恢复到默认初始状态
        self.Dcmm.data_arm_copy.qpos[0:6] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.body("object").xpos[0:3] = np.array([0, 1.5, 0.8])
        # Random 3D position TODO: Adjust to the fov
        self.random_object_pose()#用来随机生成：球的位置、速度、投掷方向、姿态和等待时间
        self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                    velocity=np.zeros(6))
        #self.object_pos3d = np.array([x, y, height])
        #self.object_vel6d = np.array([v_lin_x, v_lin_y, v_lin_z, 0.0, 0.0, 0.0])
        # TODO: TESTING
        # self.Dcmm.set_throw_pos_vel(pose=np.array([0.0, 0.4, 1.0, 1.0, 0.0, 0.0, 0.0]),
        #                             velocity=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # Random Gravity
        self.Dcmm.model.opt.gravity[2] = -9.81 + 0.5*np.random.uniform(-1, 1)
        # Random PID
        self.random_PID()
        # Random Delay
        self.random_delay()
        # Forward Kinematics
        mujoco.mj_forward(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_forward(self.Dcmm.model_arm, self.Dcmm.data_arm)#根据“你刚刚重置过后的 qpos、qvel、ctrl、act 等基本状态”来重新计算所有派生值。
        mujoco.mj_forward(self.Dcmm.model_arm_copy, self.Dcmm.data_arm_copy)

    def reset(self):#在每个episode前面reset一下
        # Reset the basic simulation
        self._reset_simulation()
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.closed= False
        self.object_throw = False#物体是否已经被“抛出”（throw）。
        self.steps = 0#当前 episode 已经走了多少个环境 step
        self.steps_since_replan = 0
        # success 状态清零
        self.success_counter = 0
        self.has_success = False
        self.success_bonus_given = False
        # Reset the time
        self.start_time = self.Dcmm.data.time#由于_reset_simulation()已经全部reset了，所以这个数值就是0
        #self.catch_time = self.Dcmm.data.time - self.start_time#这也不是为了真的算抓住的时间，而是在清零     
        self.catch_time = 0
        self._reset_plate_trajectory()# 每个 episode 开始时，重新生成一条盘子参考轨迹
        # 保存 reset 后的末端姿态，作为后续 IK 姿态目标
        # MuJoCo xquat 顺序是 wxyz
        self.target_ee_quat_wxyz = self.Dcmm.data.body("arm_seg6").xquat.copy()
        self.target_ee_quat_wxyz_copy = self.Dcmm.data.body("arm_seg6_copy").xquat.copy()
        ## Reset the target velocity of the mobile base
        self.Dcmm.target_base_vel = np.array([0.0, 0.0, 0.0])
        self.Dcmm.target_base_vel_copy = np.array([0.0, 0.0, 0.0])
        ## Reset the target joint positions of the arm
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        self.Dcmm.target_arm_qpos_copy[:] = DcmmCfg.arm_joints[:]
        ## Reset the target joint positions of the hand
        self.Dcmm.target_hand_qpos[:] = DcmmCfg.hand_joints[:]
        self.Dcmm.target_hand_qpos_copy[:] = DcmmCfg.hand_joints[:]
        ## Reset the reward
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.reward_stability = 0
        obj_pos_now = self.Dcmm.data.body(self.Dcmm.object_name).xpos
        # 计算指尖到物体的欧式距离
        plate_pos_now = self._get_plate_pos()
        # =========================================================
        # 初始化两个底盘上一帧世界坐标位置
        # 用于 compute_reward() 里计算每一步底盘运动量
        # =========================================================

        self.prev_base1_xy_for_motion_reward = (
            self.Dcmm.data.body("base_link").xpos[:2].copy()
        )
        self.prev_base2_xy_for_motion_reward = (
            self.Dcmm.data.body("base_link_copy").xpos[:2].copy()
        )
        if hasattr(self, "prev_ee1_world_for_sync_reward"):
            del self.prev_ee1_world_for_sync_reward

        if hasattr(self, "prev_ee2_world_for_sync_reward"):
            del self.prev_ee2_world_for_sync_reward
        # reset 后已经在 _reset_plate_trajectory() 里 build 过 waypoints
        path_error_now, path_progress_now, path_closest_now = self._project_point_to_waypoint_path(
            plate_pos_now
        )

        plate_ref_now = path_closest_now.copy()
        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("arm_seg6").xpos - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),#ee离arm_base的距离
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),#arm_base距离物体的距离
            "base_distance_copy": np.linalg.norm(self.Dcmm.data.body("arm_base_copy").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "env_time": self.Dcmm.data.time - self.start_time,
            "qpos_sum": self.Dcmm.data.joint("gripper2_axis").qpos[0]+self.Dcmm.data.joint("gripper1_axis").qpos[0],
            "plate_distance":np.linalg.norm(self.Dcmm.data.site("frame_bottom_center").xpos - self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "plate_z_distance" : np.abs(self.Dcmm.data.site("frame_bottom_center").xpos[2] - 
                      self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]),
            "ee_position":self.Dcmm.data.body("arm_seg6").xpos,
            "ee_position_copy":self.Dcmm.data.body("arm_seg6_copy").xpos,
            "plate_pos": plate_pos_now,
            "plate_ref_pos": plate_ref_now,
            "plate_target_pos": self.traj_target_plate_pos.copy(),
            "traj_error": float(path_error_now),
            "path_error": float(path_error_now),
            "path_progress": float(path_progress_now),
            "path_closest_point": path_closest_now,
            "target_error": np.linalg.norm(plate_pos_now - self.traj_target_plate_pos),
            "traj_progress": self._traj_episode_progress(),
            "traj_phase": self._get_traj_phase(),
        }
        # Get the observation and info
        
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]#self.initial_ee_pos3d = self._get_relative_ee_pos3d()ee相对于arm_base的相对位置
        self.prev_ee_pos3d_copy[:] = self.initial_ee_pos3d_copy[:]
        self.prev_obj_pos3d = self._get_object_pos3d()#物体基于arm_base的坐标
        self.prev_obj_pos3d_copy = self._get_relative_object_pos3d_copy()
        observation = self._get_obs()
        info = self._get_info()#每次调用前，前面就会用self.info把上一时间步的info存储起来，info就成了当前时间步的info
        # Rendering
        imgs = self.render() if self.render_mode is not None else None
        info['imgs'] = imgs
        #self.init_ee_distance = info["ee_distance"]
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand']),
                               len(self.action_buffer['base_copy']),
                               len(self.action_buffer['arm_copy']),
                               len(self.action_buffer['hand_copy'])
                               ])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))
        # 在 return observation, info 之前插入
        # print(f"--- RESET CHECK ---")
        # print(f"Gripper1 Qpos: {self.Dcmm.data.joint('gripper1_axis').qpos[0]}")
        # print(f"Gripper2 Qpos: {self.Dcmm.data.joint('gripper2_axis').qpos[0]}")
        # print(f"Hand Target: {self.Dcmm.target_hand_qpos}")
        # --- 关键调试代码开始 ---
        # print(f"\n{'='*20} 初始位姿检查 {'='*20}")
        # # 1. 打印全量 qpos (广义坐标)
        # print("All qpos:", self.Dcmm.data.qpos)
        
        # 2. 精确查看两辆车的位置 (假设它们是 Freejoint)
        # 索引取决于你的 XML 顺序，通常前 7 位是车1，后面是车2或物体
        # try:
        #     car1_pos = self.Dcmm.data.body("base_link").xpos
        #     car2_pos = self.Dcmm.data.body("base_link_copy").xpos
        #     print(f"第一辆车 [base_link] 位置: {car1_pos}")
        #     print(f"第二辆车 [base_link_copy] 位置: {car2_pos}")
            
        #     # 检查四元数（判断是否“侧着”）
        #     car2_quat = self.Dcmm.data.body("base_link_copy").xquat
        #     print(f"第二辆车 [base_link_copy] 姿态(四元数): {car2_quat}")
        # except Exception as e:
        #     print("无法获取具体 Body 位置:", e)
        # print(f"{'='*54}\n")
        # --- 关键调试代码结束 ---
        return observation, info

    def norm_ctrl(self, ctrl, components):
        '''
        Convert the ctrl (dict type) to the numpy array and return its norm value
        Input: ctrl, dict
        Return: norm, float
        '''
        ctrl_array = np.concatenate([ctrl[component]*DcmmCfg.reward_weights['r_ctrl'][component] for component in components])
        return np.linalg.norm(ctrl_array)


#     def compute_reward(self, obs, info, ctrl):
#         '''
#         Rewards:
#         - Object Position Reward
#         - Object Orientation Reward
#         - Object Touch Success Reward
#         - Object Catch Stability Reward
#         - Collision Penalty
#         - Constraint Penalty
#         '''

#         rewards = 0.0

#         # ================== 新增：轨迹跟踪奖励 ==================
#         traj_error = float(info["traj_error"])
#         target_error = float(info["target_error"])
#         prev_traj_error = float(self.info.get("traj_error", traj_error))

#         phase = int(info["traj_phase"])

#         # phase 0: 粗跟踪，重点是跟上轨迹
#         # phase 1: 精确靠近，轨迹和最终目标都重要
#         # phase 2: 保持阶段，重点是留在最终目标点
#         if phase == 0:
#             w_track = 8.0
#             w_target = 1.0
#         elif phase == 1:
#             w_track = 10.0
#             w_target = 4.0
#         else:
#             w_track = 3.0
#             w_target = 8.0

#         # 当前盘子中心靠近当前轨迹点
#         reward_traj_track = w_track * math.exp(-20.0 * traj_error ** 2)

#         # 比上一帧更靠近当前轨迹点，也给奖励
#         reward_traj_improve = 8.0 * np.clip(prev_traj_error - traj_error, -0.05, 0.05)

#         # 后期靠近最终接取点
#         reward_target_precision = w_target * math.exp(-25.0 * target_error ** 2)

#         # 旧的直接追物体中心奖励先关掉，避免和“物体下方目标点”冲突
#         reward_plate_distance = 0.0
#         reward_ee_precision = 0.0
#         # ======================================================
        

#         #新增中点奖励
#         p1 = self.Dcmm.data.body('base_link').xpos[:2]
#         p2 = self.Dcmm.data.body('base_link_copy').xpos[:2]
#         mid_point = (p1 + p2) / 2

#         # 底盘中点跟当前轨迹点的 xy，而不是直接追物体 xy
#         ref_xy = info["plate_ref_pos"][:2]
#         dist_xy = np.linalg.norm(mid_point - ref_xy)
#         reward_mid = np.exp(-2 * dist_xy) * 3

#         site_id = self.Dcmm.model.site('frame_bottom_center').id
#         site_rmat = self.Dcmm.data.site_xmat[site_id].reshape(3, 3)
#         current_z_axis = site_rmat[:, 2]
#         world_z_axis = np.array([0, 0, 1])
#         alignment = np.dot(current_z_axis, world_z_axis)
#         z_alignment_reward = alignment  * 3
#         speed1 = self.Dcmm.data.body("base_link").cvel[3:5]
#         speed1_x = speed1[0]
#         speed1_y = speed1[1]
#         speed2 = self.Dcmm.data.body("base_link_copy").cvel[3:5]
#         speed2_x = speed2[0]
#         speed2_y = speed2[1]
#         reward_speed = - abs(speed1_x - speed2_x)*5 - abs(speed1_y - speed2_y)*5
#         #reward_speed = -np.linalg.norm(speed1 - speed2)*0.5
#         #两个底盘中心点xy坐标到物体xy坐标的距离
#         p1 = self.Dcmm.data.body('base_link').xpos[:2] 
#         p2 = self.Dcmm.data.body('base_link_copy').xpos[:2]
#         mid_point = (p1 + p2) / 2
#         obj_xy = self.Dcmm.data.body('object').xpos[:2]
#         dist_xy = np.linalg.norm(mid_point - obj_xy)
#         reward_mid = np.exp(-2 * dist_xy)*3
#         #两个棒距离约束
#         base1 = self.Dcmm.data.body('base_link').xpos[:2]
#         base2 = self.Dcmm.data.body('base_link_copy').xpos[:2]
#         base_link_dist = np.linalg.norm(base1 - base2)
#         #print(f"base_link_dist: {base_link_dist:.4f}")
#         dist_min, dist_max = 0.9, 1.22
#         dist_error = base_link_dist - np.clip(base_link_dist, dist_min, dist_max)
#         reward_base_link_dist = -5 * np.abs(dist_error)
#         #
#         pre_height_dist = abs(self.info["ee_position"][2] - self.info["ee_position_copy"][2])
#         now_height_dist = abs(info["ee_position"][2] - info["ee_position_copy"][2])
#         reward_height_improvement = (pre_height_dist - now_height_dist) * 10.0
#         reward_height_improvement = np.clip(reward_height_improvement, -5.0, 5.0)
        
#         # reward_base_pos_dist = (self.info["base_distance"] - info["base_distance"]) * DcmmCfg.reward_weights["r_base_pos"]
#         # reward_base_pos_copy = (self.info["base_distance_copy"] - info["base_distance_copy"]) * DcmmCfg.reward_weights["r_base_pos"]
#         # reward_base_pose = reward_base_pos_dist + reward_base_pos_copy
#         bar1_height = self.Dcmm.data.body("bar_left").xpos[2]
#         bar2_height = self.Dcmm.data.body("bar_right").xpos[2]
#         height_dist = abs(bar1_height - bar2_height)
#         reward_height = math.exp(-50 * height_dist**2) * 1
        
#         reward_z_diatance = 0
#         if info["plate_distance"]<0.1:
#             reward_z_diatance = (self.info["plate_z_distance"]-info["plate_z_distance"])* DcmmCfg.reward_weights["r_base_pos"]
#             # print("##############################true#################################")
#             # print("##############################true#################################")
#             # print("##############################true#################################")

#         #elif self.task == 'Tracking':
#             ## Ctrl Penalty
#             # Compute the norm of base and arm movement through the current actions in the grasping stage
#         reward_ctrl = - self.norm_ctrl(ctrl, {"base","base_copy","arm","arm_copy","hand","hand_copy"})*0.1

#         #rewards = (reward_ctrl + reward_plate_distance + reward_z_diatance + reward_ee_precision + reward_height_improvement + z_alignment_reward + reward_base_link_dist+reward_speed+reward_mid+reward_height)
#         rewards = (
#     reward_ctrl
#     + reward_traj_track
#     + reward_traj_improve
#     + reward_target_precision
#     + reward_height_improvement
#     + z_alignment_reward
#     + reward_base_link_dist
#     + reward_speed
#     + reward_mid
#     + reward_height
# )
#         # print(f"1. 控制惩罚 (reward_ctrl):             {reward_ctrl:.6f}")
#         # print(f"2. 盘子距离奖励 (reward_plate_dist):    {reward_plate_distance:.6f}")
#         # print(f"3. 底盘速度对齐 (reward_speed):         {reward_speed:.6f}")
#         # # print(f"4. 两棒距离约束 (reward_base_link_dist):     {reward_base_link_dist:.6f}")
#         # print(f"5. Z轴距离奖励 (reward_z_diatance):     {reward_z_diatance:.6f}")
#         # print(f"6. 末端精度奖励 指数奖励(reward_ee_precision):  {reward_ee_precision:.6f}")#太大
#         # print(f"7. 末端高度一致奖励 (reward_height_improve): {reward_height_improvement:.6f}")
#         # print(f"8.盘子z轴向上奖励 (z_alignment_reward): {z_alignment_reward:.6f}")
#         # print(f"9.底座距离约束奖励 (reward_base_link_dist ): {reward_base_link_dist :.6f}")
#         # print(f"10.底座中点奖励 (reward_mid ): {reward_mid :.6f}")#很小
#         # print(f"11.两棒高度一致奖励 (reward_height ): {reward_height :.6f}")
#         # print("-" * 30)
#         # print(f"*** 总奖励 (TOTAL REWARDS):           {rewards:.6f} ***")
#             # print("="*30 + "\n")
#         return rewards

    def compute_reward(self, obs, info, ctrl):
        """
        静止物体 Tracking 任务 reward。

        当前目标：
        1. 盘子中心靠近最终目标点 plate_target_pos
        2. 盘子沿 waypoint path 往目标方向前进
        3. 防止盘子沿路径后退
        4. 盘子保持水平
        5. 两个底盘中点靠近目标 xy，并保持合理间距
        6. 两臂 / 两杆高度不要差太多
        7. 避免 IK 失败、碰撞、动作过大
        """

        rw = DcmmCfg.reward_weights["r_traj"]

        def to_float(x):
            return float(np.asarray(x).reshape(-1)[0])

        def gaussian(error, sigma):
            sigma = max(float(sigma), 1e-6)
            return math.exp(- (float(error) / sigma) ** 2)
        if self.task == "Tracking":
            # =========================================================
            # 1. 读取核心状态
            # =========================================================
            traj_error = to_float(info.get("traj_error", info.get("path_error", 0.0)))#盘子到路径点最近点的距离

            target_error = to_float(info["target_error"])
            prev_target_error = to_float(
                self.info.get("target_error", target_error)
            )

            path_error = to_float(info.get("path_error", traj_error))
            path_progress = to_float(info.get("path_progress", 0.0))

            prev_path_progress = to_float(
                self.info.get(
                    "path_progress",
                    getattr(self, "prev_path_progress", path_progress)
                )
            )

            # 原始路径进度变化
            progress_delta_raw = path_progress - prev_path_progress

            # 只奖励正向前进
            progress_delta = max(0.0, progress_delta_raw)

            # 单独记录后退量
            backward_delta = max(0.0, -progress_delta_raw)

            # =========================================================
            # 2. 最终目标奖励：far / mid / precision / improve
            # =========================================================
            reward_target_far = 8 * gaussian(
                target_error,
                rw.get("target_far_sigma", 1.5)
            )

            reward_target_mid = 10 * gaussian(
                target_error,
                rw.get("target_mid_sigma", 0.8)
            )+8*gaussian(
                target_error,
                rw.get("target_mid_sigma", 0.4) )

            reward_target_precision = 12 * gaussian(
                target_error,
                rw.get("target_precision_sigma", 0.10)
            )

            reward_target_improve = 8* np.clip(
                (prev_target_error - target_error) / 0.03,
                -1.0,
                1.0
            )

            reward_target_distance_penalty = -rw.get("target_distance", 1.0) * np.clip(
                target_error / 1.0,
                0.0,
                2.0
            )

            # =========================================================
            # 3. waypoint path 奖励
            # =========================================================
            # 如果路径进度后退，就降低 path_follow，避免贴着后方路径也拿奖励
            backward_gate = np.clip(
                1.0 - backward_delta / max(self.path_progress_clip, 1e-6),
                0.0,
                1.0
            )

            reward_path_follow = backward_gate * rw.get("path_follow", 0.05) * gaussian(
                path_error,
                self.path_sigma
            )#盘子离当前路径点越近，奖励越大。

            reward_path_progress = rw.get("path_progress", 10.0) * np.clip(
                progress_delta,
                0.0,
                self.path_progress_clip
            ) / max(self.path_progress_clip, 1e-6)

            # reward_path_backward = -rw.get("path_backward", 1.0) * np.clip(
            #     backward_delta / max(self.path_progress_clip, 1e-6),
            #     0.0,
            #     1.0
            # )

            # reward_no_progress = -rw.get("no_progress", 0.5) * float(
            #     progress_delta <= 1e-5 and target_error > 0.30
            # )
            prev_path_error = to_float(self.info.get("path_error", path_error))

            reward_waypoint_improve = 8* np.clip(
                (prev_path_error - path_error) / 0.03,
                -1.0,
                1.0
            )

            reward_waypoint_reach = 8 * gaussian(
                path_error,
                rw.get("waypoint_sigma", 0.15)
            )
            # =========================================================
            # 4. 盘子朝目标方向运动奖励
            # =========================================================
            plate_pos_now = self._get_plate_pos()
            prev_plate_pos = getattr(
                self,
                "prev_plate_pos_for_reward",
                plate_pos_now.copy()
            )

            dt = max(1.0 / float(self.fps), 1e-6)
            plate_vel_xy = (plate_pos_now[:2] - prev_plate_pos[:2]) / dt

            target_xy = np.asarray(info["plate_target_pos"][:2])
            to_target_xy = target_xy - plate_pos_now[:2]
            to_target_norm = float(np.linalg.norm(to_target_xy))

            if to_target_norm > 1e-6:
                to_target_dir = to_target_xy / to_target_norm
                vel_to_target = float(np.dot(plate_vel_xy, to_target_dir))
            else:
                vel_to_target = 0.0

            reward_plate_vel_to_target = 2 * np.clip(
                vel_to_target / 0.2,
                -1.0,
                1.0
            )

            self.prev_plate_pos_for_reward = plate_pos_now.copy()

            # =========================================================
            # 5. 盘子水平约束
            # =========================================================
            site_id = self.Dcmm.model.site("frame_bottom_center").id
            site_rmat = self.Dcmm.data.site_xmat[site_id].reshape(3, 3)

            current_z_axis = site_rmat[:, 2]
            world_z_axis = np.array([0.0, 0.0, 1.0])

            alignment = float(np.dot(current_z_axis, world_z_axis))
            alignment = float(np.clip(alignment, -1.0, 1.0))

            plate_tilt_error = max(0.0, 1.0 - alignment)

            reward_plate_level = -10 * np.clip(
                (plate_tilt_error / 0.10) ** 2,
                0.0,
                8.0
            )

            # =========================================================
            # 6. 双底盘中点靠近目标 xy
            # =========================================================
            base1_xy = self.Dcmm.data.body("base_link").xpos[:2].copy()
            base2_xy = self.Dcmm.data.body("base_link_copy").xpos[:2].copy()
            base_mid_xy = 0.5 * (base1_xy + base2_xy)

            mid_target_error = float(np.linalg.norm(base_mid_xy - target_xy))
            reward_base_mid = (
                3.0 * np.clip(1.0 - mid_target_error / 2.0, 0.0, 6.0)
                + 3.0 * gaussian(mid_target_error, 1.5)
                + 3.0 * gaussian(mid_target_error, 0.4)
            )

            # =========================================================
            # 6.x 两个底盘运动量差惩罚
            # 目标：
            #   防止一个底盘正常移动，另一个底盘几乎不动
            # =========================================================
            dt = max(1.0 / float(self.fps), 1e-6)

            prev_base1_xy = getattr(
                self,
                "prev_base1_xy_for_motion_reward",
                base1_xy.copy()
            )
            prev_base2_xy = getattr(
                self,
                "prev_base2_xy_for_motion_reward",
                base2_xy.copy()
            )

            base1_delta_xy = base1_xy - prev_base1_xy
            base2_delta_xy = base2_xy - prev_base2_xy

            base1_motion = float(np.linalg.norm(base1_delta_xy))
            base2_motion = float(np.linalg.norm(base2_delta_xy))

            base1_speed = base1_motion / dt
            base2_speed = base2_motion / dt

            base_motion_diff = abs(base1_motion - base2_motion)
            base_speed_diff = abs(base1_speed - base2_speed)

            # 更新历史位置，给下一步用
            self.prev_base1_xy_for_motion_reward = base1_xy.copy()
            self.prev_base2_xy_for_motion_reward = base2_xy.copy()

            # 1. 惩罚两个底盘每一步位移量不同
            reward_base_motion_diff = - 4 * np.clip(
                base_motion_diff / 0.03,
                0.0,
                4.0
            )
            min_base_motion = min(base1_motion, base2_motion)
            need_move_gate = float(target_error > 0.1)

            reward_both_base_move = 2 * need_move_gate * np.clip(
                min_base_motion / 0.02,
                0.0,
                1.5
            )

            slow_motion_threshold = 0.005

            reward_base_one_stuck = -2 * need_move_gate * float(
                base1_motion < slow_motion_threshold or base2_motion < slow_motion_threshold
            )
            base1_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link").xquat[0],
                self.Dcmm.data.body("base_link").xquat[3]
            ))

            base2_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link_copy").xquat[0],
                self.Dcmm.data.body("base_link_copy").xquat[3]
            ))

            base_vec = base2_xy - base1_xy
            base_line_yaw = float(math.atan2(base_vec[1], base_vec[0]))

            base1_face_line_error = abs(math.atan2(
                math.sin(base1_yaw - base_line_yaw),
                math.cos(base1_yaw - base_line_yaw)
            ))

            base2_face_line_error = abs(math.atan2(
                math.sin(base2_yaw - (base_line_yaw + math.pi)),
                math.cos(base2_yaw - (base_line_yaw + math.pi))
            ))

            reward_base_yaw_face = -8.0 * float(np.clip(
                ((base1_face_line_error + base2_face_line_error) / 0.30) ** 2,
                0.0,
                5.0
            ))

            # =========================================================
            # 7. 双底盘距离约束：范围奖励
            # =========================================================
            base_link_dist = float(np.linalg.norm(base1_xy - base2_xy))

            dist_min = rw.get("base_dist_min", 1.05)
            dist_max = rw.get("base_dist_max", 1.1)

            if base_link_dist < dist_min:
                base_dist_error = dist_min - base_link_dist
            elif base_link_dist > dist_max:
                base_dist_error = base_link_dist - dist_max
            else:
                base_dist_error = 0.0

            reward_base_dist = -8 * np.clip(
                (base_dist_error / 0.05) ** 2,
                0.0,
                8.0
            )

            # =========================================================
            # 8. 两个底盘速度同步
            # =========================================================
            # base1_vel = self.Dcmm.data.body("base_link").cvel[3:5]
            # base2_vel = self.Dcmm.data.body("base_link_copy").cvel[3:5]

            # vel_sync_error = float(np.linalg.norm(base1_vel - base2_vel))
            # reward_vel_sync = -rw.get("vel_sync", 0.5) * vel_sync_error
            base_action = np.concatenate([
                np.asarray(ctrl["base"][:2], dtype=np.float64).reshape(-1),
                np.asarray(ctrl["base_copy"][:2], dtype=np.float64).reshape(-1),
            ])

            base1_world_vel_cmd = np.array([
                math.cos(base1_yaw) * base_action[0] - math.sin(base1_yaw) * base_action[1],
                math.sin(base1_yaw) * base_action[0] + math.cos(base1_yaw) * base_action[1],
            ], dtype=np.float64)

            base2_world_vel_cmd = np.array([
                math.cos(base2_yaw) * base_action[2] - math.sin(base2_yaw) * base_action[3],
                math.sin(base2_yaw) * base_action[2] + math.cos(base2_yaw) * base_action[3],
            ], dtype=np.float64)

            # 世界坐标系下的 XY 速度误差
            base_world_vel_error = float(np.linalg.norm(
                base1_world_vel_cmd - base2_world_vel_cmd
            ))
            base_world_vx_error = abs(float(
                base1_world_vel_cmd[0] - base2_world_vel_cmd[0]
            ))

            base_world_vy_error = abs(float(
                base1_world_vel_cmd[1] - base2_world_vel_cmd[1]
            ))

            # =========================================================
            # 奖励项
            # =========================================================
            # 0.10 是速度误差归一化尺度。
            # 如果 base_action 已经是真实速度，单位大概是 m/s，那么 0.10 表示：
            #   两个底盘世界速度差 0.10 m/s 左右时，会产生明显惩罚。
            #
            # 如果 base_action 是 [-1, 1] 的归一化动作，
            # 那这里的 0.10 就表示动作尺度下的误差，需要根据你的动作缩放重新调。
            reward_vel_sync = (
                -8.0 * float(np.clip(
                    (base_world_vel_error / 0.10) ** 2,
                    0.0,
                    5.0
                ))
            )

            # =========================================================
            # 9. 两根杆 / 两个末端高度同步
            # =========================================================
            bar1_z = float(self.Dcmm.data.body("bar_left").xpos[2])
            bar2_z = float(self.Dcmm.data.body("bar_right").xpos[2])
            bar_height_error = abs(bar1_z - bar2_z)

            reward_bar_level = -4 * np.clip(
                bar_height_error / 0.05,
                0.0,
                3.0
            )

            ee1_z = float(self.Dcmm.data.body("arm_seg6").xpos[2])
            ee2_z = float(self.Dcmm.data.body("arm_seg6_copy").xpos[2])
            ee_height_error = abs(ee1_z - ee2_z)

            reward_ee_height = -rw.get("ee_height", 0.2) * np.clip(
                ee_height_error / 0.10,
                0.0,
                3.0
            )
            base1_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link").xquat[0],
                self.Dcmm.data.body("base_link").xquat[3]
            ))

            base2_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link_copy").xquat[0],
                self.Dcmm.data.body("base_link_copy").xquat[3]
            ))

            prev_base1_yaw = float(getattr(self, "prev_base1_yaw_for_reward", base1_yaw))
            prev_base2_yaw = float(getattr(self, "prev_base2_yaw_for_reward", base2_yaw))

            base1_yaw_delta = abs(math.atan2(
                math.sin(base1_yaw - prev_base1_yaw),
                math.cos(base1_yaw - prev_base1_yaw)
            ))

            base2_yaw_delta = abs(math.atan2(
                math.sin(base2_yaw - prev_base2_yaw),
                math.cos(base2_yaw - prev_base2_yaw)
            ))

            base_yaw_delta = max(base1_yaw_delta, base2_yaw_delta)

            reward_base_yaw_delta = -8 * float(np.clip(
                (base_yaw_delta / 0.01) ** 2,
                0.0,
                5.0
            ))

            self.prev_base1_yaw_for_reward = base1_yaw
            self.prev_base2_yaw_for_reward = base2_yaw
            # =========================================================
            # 10. 动作惩罚
            # =========================================================
            reward_ctrl = -1 * self.norm_ctrl(
                ctrl,
                ["base", "base_copy", "arm", "arm_copy", "hand", "hand_copy"]
            )
            # =========================================================
            # 10.x 底座 action 贴边惩罚
            # 目标：
            #   防止两个底盘的策略输出长期接近动作上限。
            #
            # 说明：
            #   ctrl["base"] 和 ctrl["base_copy"] 是已经反归一化后的底座目标速度。
            #   你的 Tracking 里底座动作通常是 [-1.5, 1.5]。
            #
            # 惩罚逻辑：
            #   |action| <= 1.05 时不惩罚；
            #   |action| 从 1.05 到 1.5 之间逐渐增加惩罚；
            #   |action| 越接近 1.5，惩罚越大。
            # =========================================================
            base_action = np.concatenate([
                np.asarray(ctrl["base"][:2], dtype=np.float64).reshape(-1),
                np.asarray(ctrl["base_copy"][:2], dtype=np.float64).reshape(-1),
            ])

            base_action_limit = 1.8

            # 这里是关键：
            # 不要从很小动作就开始罚，否则会慢。
            # 只在接近边界时罚。
            base_action_soft_bound = 1.1

            base_action_abs = np.abs(base_action)

            base_action_excess = np.maximum(
                base_action_abs - base_action_soft_bound,
                0.0
            )

            base_action_margin = max(
                base_action_limit - base_action_soft_bound,
                1e-6
            )

            base_action_saturation_ratio = np.clip(
                base_action_excess / base_action_margin,
                0.0,
                1.0
            )

            reward_base_action_bound = -2.0 * float(
                np.sum(base_action_saturation_ratio ** 2)
            )


            # ---------------------------------------------------------
            # 11. 动作边界惩罚：机械臂
            # ---------------------------------------------------------
            arm_action = np.asarray(ctrl["arm"][:3], dtype=np.float64)
            arm_copy_action = np.asarray(ctrl["arm_copy"][:3], dtype=np.float64)

            arm_action_all = np.concatenate([
                arm_action,
                arm_copy_action
            ])

            arm_action_abs = np.abs(arm_action_all)

            # 你的 arm denorm 是 0.025，接近 0.025 才罚
            arm_action_excess = np.maximum(
                arm_action_abs - 0.035,
                0.0
            )

            reward_arm_action_bound = -2.0 * float(np.sum(
                np.clip(arm_action_excess / 0.003, 0.0, 1.0) ** 2
            ))

            # =========================================================
            # 11. IK 失败惩罚
            # =========================================================
            reward_ik = 0.0
            if not self.arm_limit:
                reward_ik = -rw.get("ik_fail", 2.0)



            # =========================================================
            # 13. 碰撞惩罚
            # =========================================================
            reward_collision = 0.0

            if self.contacts.get("any_base_collision", False):
                reward_collision -= rw.get("collision", 10.0)

            if self.contacts.get("object_failed", False):
                reward_collision -= rw.get("collision", 10.0)
            # =========================================================
            # 底盘防翻车：惩罚两个底盘倾斜
            # =========================================================
            base1_rmat = self.Dcmm.data.body("base_link").xmat.reshape(3, 3)
            base2_rmat = self.Dcmm.data.body("base_link_copy").xmat.reshape(3, 3)

            base1_z_axis = base1_rmat[:, 2]
            base2_z_axis = base2_rmat[:, 2]

            world_z = np.array([0.0, 0.0, 1.0])

            base1_upright = float(np.clip(np.dot(base1_z_axis, world_z), -1.0, 1.0))
            base2_upright = float(np.clip(np.dot(base2_z_axis, world_z), -1.0, 1.0))

            base1_tilt_error = max(0.0, 1.0 - base1_upright)
            base2_tilt_error = max(0.0, 1.0 - base2_upright)

            reward_base_upright = -120 * (
                np.clip((base1_tilt_error / 0.08) ** 2, 0.0, 10.0)
                + np.clip((base2_tilt_error / 0.08) ** 2, 0.0, 10.0)
            )
            base1_q = self.Dcmm.data.body("base_link").xquat.copy()
            base2_q = self.Dcmm.data.body("base_link_copy").xquat.copy()
            #print("base yaw deg:", np.degrees(quat2theta(base1_q[0], base1_q[3])), np.degrees(quat2theta(base2_q[0], base2_q[3])))
            # =========================================================
            # 14. 成功奖励
            # =========================================================

            safe = (
                not self.contacts.get("object_failed", False)
                and not self.contacts.get("any_base_collision", False)
            )
            tracking_success = bool(
                target_error < self.traj_success_threshold and safe
            )

            reward_success = 2 if tracking_success else 0.0

            # ---------------------------------------------------------
            #机械臂动作相同奖励
            arm_action = np.asarray(ctrl["arm"][:3], dtype=np.float64)
            arm_copy_action = np.asarray(ctrl["arm_copy"][:3], dtype=np.float64)

            base_action = np.asarray(ctrl["base"][:2], dtype=np.float64)
            base_copy_action = np.asarray(ctrl["base_copy"][:2], dtype=np.float64)

            base1_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link").xquat[0],
                self.Dcmm.data.body("base_link").xquat[3]
            ))

            base2_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link_copy").xquat[0],
                self.Dcmm.data.body("base_link_copy").xquat[3]
            ))

            # =========================================================
            # 1. 把机械臂局部输出转到世界坐标系
            # =========================================================
            arm1_world_delta = np.array([
                math.cos(base1_yaw) * arm_action[0] - math.sin(base1_yaw) * arm_action[1],
                math.sin(base1_yaw) * arm_action[0] + math.cos(base1_yaw) * arm_action[1],
                arm_action[2],
            ], dtype=np.float64)

            arm2_world_delta = np.array([
                math.cos(base2_yaw) * arm_copy_action[0] - math.sin(base2_yaw) * arm_copy_action[1],
                math.sin(base2_yaw) * arm_copy_action[0] + math.cos(base2_yaw) * arm_copy_action[1],
                arm_copy_action[2],
            ], dtype=np.float64)

            # 两个机械臂末端输出在世界坐标系下应该尽量一致
            # 这才是“对向底座下的正确镜像同步”
            arm_output_world_xy_error = float(np.linalg.norm(
                arm1_world_delta[:2] - arm2_world_delta[:2]
            ))

            arm_output_world_z_error = abs(float(
                arm1_world_delta[2] - arm2_world_delta[2]
            ))

            # =========================================================
            # 2. 把底座局部速度输出也转到世界坐标系
            # =========================================================
            # base_action 是底座局部坐标系下的速度指令。
            # 机械臂 action 是每个 policy step 的末端位移量。
            # 所以这里把底座速度乘 dt，近似转成这一帧底座位移量。
            dt = max(1.0 / float(self.fps), 1e-6)

            base1_world_delta = np.array([
                math.cos(base1_yaw) * base_action[0] - math.sin(base1_yaw) * base_action[1],
                math.sin(base1_yaw) * base_action[0] + math.cos(base1_yaw) * base_action[1],
            ], dtype=np.float64) * dt

            base2_world_delta = np.array([
                math.cos(base2_yaw) * base_copy_action[0] - math.sin(base2_yaw) * base_copy_action[1],
                math.sin(base2_yaw) * base_copy_action[0] + math.cos(base2_yaw) * base_copy_action[1],
            ], dtype=np.float64) * dt

            # =========================================================
            # 3. 比较“底座移动 + 机械臂输出”后的总支撑点运动
            # =========================================================
            # 这一步很重要：
            #   有时候机械臂输出本身不同，但它是在补偿底座运动；
            #   所以更应该看底座和机械臂叠加之后，两个末端支撑点在世界坐标系下是否一致。
            support1_world_delta_xy = base1_world_delta + arm1_world_delta[:2]
            support2_world_delta_xy = base2_world_delta + arm2_world_delta[:2]

            support_world_xy_error = float(np.linalg.norm(
                support1_world_delta_xy - support2_world_delta_xy
            ))

            # =========================================================
            # 4. 合成奖励
            # =========================================================
            reward_arm_mirror_sync = (
                # 两个机械臂输出转到世界坐标系后，XY 位移应该接近
                -4.0 * float(np.clip(
                    (arm_output_world_xy_error / 0.020) ** 2,
                    0.0,
                    5.0
                ))

                # 两个机械臂 Z 方向应该同步
                -4.0 * float(np.clip(
                    (arm_output_world_z_error / 0.010) ** 2,
                    0.0,
                    5.0
                ))

                # 底座输出 + 机械臂输出之后，两个支撑点的总世界位移应该接近
                -4.0 * float(np.clip(
                    (support_world_xy_error / 0.030) ** 2,
                    0.0,
                    5.0
                ))
            )
            # =========================================================
            # 15. 总奖励
            # =========================================================
            rewards = (
                # 最终目标
                reward_target_far
                + reward_target_mid
                + reward_target_precision
                + reward_target_improve
                + reward_target_distance_penalty

                # 路径
                + reward_path_follow
                #+ reward_path_progress
                #+ reward_path_backward
                #+ reward_no_progress
                + reward_waypoint_improve
                + reward_waypoint_reach
                # 直接鼓励盘子朝目标方向动
                + reward_plate_vel_to_target
                + reward_base_yaw_face
                + reward_both_base_move
                + reward_base_one_stuck
                +reward_base_yaw_delta
                # 底盘
                + reward_base_mid
                + reward_base_dist
                + reward_vel_sync
                #+ reward_base_motion_diff

                # 姿态 / 高度 / 动作
                + reward_plate_level
                + reward_bar_level
                + reward_ee_height
                #+ reward_arm_posture
                #+ reward_arm_symmetry
                + reward_ctrl
                + reward_arm_mirror_sync 
                + reward_base_action_bound
                + reward_arm_action_bound
                #+reward_base_yaw_delta

                # 安全
                + reward_ik
                + reward_collision
                + reward_base_upright
                + reward_success
            )

            # =========================================================
            # 16. 调试信息
            # =========================================================
            info["reward_target_far"] = float(reward_target_far)
            info["reward_target_mid"] = float(reward_target_mid)
            info["reward_target_precision"] = float(reward_target_precision)
            info["reward_target_improve"] = float(reward_target_improve)
            info["reward_target_distance_penalty"] = float(reward_target_distance_penalty)
            #info["reward_base_motion_diff"] = float(reward_base_motion_diff)
            info["reward_path_follow"] = float(reward_path_follow)
            #info["reward_path_progress"] = float(reward_path_progress)
            #info["reward_path_backward"] = float(reward_path_backward)
            #info["reward_no_progress"] = float(reward_no_progress)
            info["waypoint_improve"] = float(reward_waypoint_improve)
            info["waypoint_reach"] = float(reward_waypoint_reach)
            #info["reward_both_base_move"] = float(reward_both_base_move)
            #info["reward_base_one_stuck"] = float(reward_base_one_stuck)

            info["reward_plate_vel_to_target"] = float(reward_plate_vel_to_target)
            info["reward_plate_level"] = float(reward_plate_level)

            info["reward_base_mid"] = float(reward_base_mid)
            info["reward_base_dist"] = float(reward_base_dist)
            info["reward_vel_sync"] = float(reward_vel_sync)

            info["reward_bar_level"] = float(reward_bar_level)
            info["reward_ee_height"] = float(reward_ee_height)

            #info["reward_arm_posture"] = float(reward_arm_posture)
            #info["reward_arm_symmetry"] = float(reward_arm_symmetry)
            info["reward_ctrl"] = float(reward_ctrl)
            info["reward_ik"] = float(reward_ik)
            info["reward_collision"] = float(reward_collision)
            info["reward_success"] = float(reward_success)

            info["debug_traj_error"] = float(traj_error)
            info["debug_target_error"] = float(target_error)

            info["debug_path_error"] = float(path_error)
            info["debug_path_progress"] = float(path_progress)
            info["debug_path_progress_delta"] = float(progress_delta)
            info["debug_path_progress_raw_delta"] = float(progress_delta_raw)
            info["debug_path_backward_delta"] = float(backward_delta)

            info["debug_plate_vel_to_target"] = float(vel_to_target)
            info["debug_mid_target_error"] = float(mid_target_error)

            info["debug_base_link_dist"] = float(base_link_dist)
            info["debug_base_pair_distance"] = float(base_link_dist)
            info["debug_base_dist_error"] = float(base_dist_error)

            info["debug_plate_alignment"] = float(alignment)
            info["debug_bar_height_error"] = float(bar_height_error)
            info["debug_ee_height_error"] = float(ee_height_error)

            info["debug_base1_x"] = float(base1_xy[0])
            info["debug_base1_y"] = float(base1_xy[1])
            info["debug_base2_x"] = float(base2_xy[0])
            info["debug_base2_y"] = float(base2_xy[1])

            info["reward_base_upright"] = float(reward_base_upright)
            info["debug_base1_upright"] = float(base1_upright)
            info["debug_base2_upright"] = float(base2_upright)

            self.prev_path_progress = float(path_progress)

            return rewards
        else:
            # =========================================================
            # Catching reward：不使用 waypoint，不使用预测落点
            # 目标：
            #   1. 盘子靠近当前物体下方目标点
            #   2. 减少“XY 靠近但没接住也高分”的情况
            #   3. 物体安全落到盘子底面
            #   4. 惩罚边缘接触、失败接触
            #   5. 约束两个对向底盘在世界坐标系下同步运动
            #   6. 约束两个机械臂支撑点同步，减少盘子倾斜
            # =========================================================

            target_error = to_float(info["target_error"])
            prev_target_error = to_float(
                self.info.get("target_error", target_error)
            )
            gripper1_pos = self.Dcmm.data.body("arm_seg6").xpos.copy()
            gripper2_pos = self.Dcmm.data.body("arm_seg6_copy").xpos.copy()
            #plate_pos_now =0.5 * (gripper1_pos + gripper2_pos)
            plate_pos_now = self._get_plate_pos()
            #target_pos = 0.5 * (gripper1_pos + gripper2_pos)
            target_pos = np.asarray(info["plate_target_pos"], dtype=np.float64)

            object_pos_now = self.Dcmm.data.body(self.Dcmm.object_name).xpos.copy()
            plate_center_now = self.Dcmm.data.site("frame_bottom_center").xpos.copy()

            target_vec = target_pos - plate_pos_now
            xy_error = float(np.linalg.norm(target_vec[:2]))
            z_error = float(abs(target_vec[2]))
            object_plate_vec = object_pos_now - plate_center_now
            object_plate_xy_error = float(np.linalg.norm(object_plate_vec[:2]))
            object_plate_z_error = float(abs(object_plate_vec[2]))

            target_object_vec = target_pos - object_pos_now
            target_object_xy_error = float(np.linalg.norm(target_object_vec[:2]))
            target_object_z_error = float(abs(target_object_vec[2]))
            z_gate_loose = gaussian(z_error, 0.20)
            z_gate_tight = gaussian(z_error, 0.12)

            # 远距离奖励：保留一点远距离梯度，但数值不大
            reward_target_far = 3.0 * gaussian(xy_error, 1.2)

            # 中距离奖励：让盘子靠近目标，但用 target_error 控制，不让它过大
            reward_target_mid = 8.0 * gaussian(target_error, 0.45)

            # XY 奖励：必须结合 z_gate，避免“只对准 XY 但没接住”拿太高分
            reward_target_xy = (
                20.0 * gaussian(xy_error, 0.25) * z_gate_loose
                + 40.0 * gaussian(xy_error, 0.12) * z_gate_tight
            )

            # 精确奖励：只有 XY 和 Z 都比较接近时才明显
            reward_target_precision = (
                35.0
                * gaussian(xy_error, 0.10)
                * gaussian(z_error, 0.12)
            )

            # 每一步距离变小就奖励
            reward_target_improve = 5.0 * np.clip(
                (prev_target_error - target_error) / 0.05,
                -1.0,
                1.0
            )

            # 距离太远惩罚
            reward_target_distance_penalty = -4.0 * np.clip(
                target_error / 1.0,
                0.0,
                2.0
            )

            # XY 仍然偏太远时惩罚
            reward_xy_far_penalty = -10.0 * float(
                np.clip((xy_error - 0.20) / 0.40, 0.0, 1.0) ** 2
            )

            # Z 方向还没接近时惩罚，防止只在平面上追到但高度不对
            reward_z_far_penalty = -6.0 * float(
                np.clip((z_error - 0.12) / 0.30, 0.0, 1.0) ** 2
            )

            # =========================================================
            # 2. 盘子朝当前目标方向运动奖励
            # =========================================================
            prev_plate_pos = getattr(
                self,
                "prev_plate_pos_for_reward",
                plate_pos_now.copy()
            )

            dt = max(1.0 / float(self.fps), 1e-6)
            plate_vel_xy = (plate_pos_now[:2] - prev_plate_pos[:2]) / dt

            to_target_xy = target_pos[:2] - plate_pos_now[:2]
            to_target_norm = float(np.linalg.norm(to_target_xy))

            if to_target_norm > 1e-6:
                to_target_dir = to_target_xy / to_target_norm
                vel_to_target = float(np.dot(plate_vel_xy, to_target_dir))
            else:
                vel_to_target = 0.0

            if xy_error > 0.25:
                reward_plate_vel_to_target = 2.0 * np.clip(
                    vel_to_target / 0.20,
                    -1.0,
                    1.0
                )
            else:
                reward_plate_vel_to_target = 0.0

            self.prev_plate_pos_for_reward = plate_pos_now.copy()

            # =========================================================
            # 3. 物体落盘 / 接住并保持奖励
            # =========================================================
            object_on_plate = bool(self.contacts.get("object_on_plate", False))
            object_on_edge = bool(self.contacts.get("object_on_edge", False))
            object_failed = bool(self.contacts.get("object_failed", False))

            any_base_collision = bool(self.contacts.get("any_base_collision", False))
            any_arm_collision = bool(self.contacts.get("any_arm_collision", False))

            safe = (
                not object_failed
                and not any_base_collision
                and not any_arm_collision
            )

            raw_success = object_on_plate and safe

            if raw_success:
                self.success_counter += 1
            else:
                self.success_counter = 0

            catch_success_now = self.success_counter >= self.success_hold_steps

            if catch_success_now and not self.has_success:
                self.has_success = True
                self.catch_time = float(info["env_time"])
            #     print(
            #     "\n[CATCH HOLD SUCCESS REACHED] "
            #     f"env_time={float(info['env_time']):.4f}, "
            #     f"steps={self.steps}, "
            #     f"success_counter={self.success_counter}, "
            #     f"success_hold_steps={self.success_hold_steps}, "
            #     f"raw_success={raw_success}, "
            #     f"object_on_plate={object_on_plate}, "
            #     f"object_failed={object_failed}, "
            #     f"any_base_collision={any_base_collision}, "
            #     f"any_arm_collision={any_arm_collision}, "
            #     f"safe={safe}, "
            #     f"has_success={self.has_success}, "
            #     f"catch_time={self.catch_time:.4f}"
            # )

            info["is_success"] = float(self.has_success)
            info["raw_success"] = float(raw_success)
            info["success_counter"] = float(self.success_counter)
            info["success_object_on_plate"] = float(object_on_plate)
            info["success_safe"] = float(safe)
            info["success_has_success"] = float(self.has_success)
            info["success_bonus_given"] = float(self.success_bonus_given)
            info["success_catch_time"] = float(self.catch_time)

            # 加上这两个字段，防止 PPO / logger 端只读 done_success 或 episode_success
            info["done_success"] = float(self.has_success)
            info["episode_success"] = float(self.has_success)

            # 只有安全落到 frame_bottom 才给接触奖励
            reward_object_on_plate = 40.0 if raw_success else 0.0

            # 边缘接触不是成功，要明显惩罚
            reward_object_on_edge = (
                -15.0 if (object_on_edge and safe and not object_on_plate) else 0.0
            )

            # 成功大奖励只给一次
            if catch_success_now and not self.success_bonus_given:
                reward_success = 20.0
                # print(reward_success)
                self.success_bonus_given = True
            else:
                reward_success = 0.0

            # 接住以后继续保持
            if self.has_success and raw_success:
                hold_time = max(0.0, float(info["env_time"]) - float(self.catch_time))
                reward_stability = 30.0 * hold_time
            else:
                reward_stability = 0.0

            # =========================================================
            # 4. 失败惩罚
            # =========================================================
            reward_collision = 0.0

            if object_failed:
                reward_collision -= 120.0

            if any_base_collision:
                reward_collision -= 80.0

            if any_arm_collision:
                reward_collision -= 60.0

            # =========================================================
            # 5. 盘子水平惩罚
            # =========================================================
            site_id = self.Dcmm.model.site("frame_bottom_center").id
            site_rmat = self.Dcmm.data.site_xmat[site_id].reshape(3, 3)

            current_z_axis = site_rmat[:, 2]
            world_z_axis = np.array([0.0, 0.0, 1.0])

            alignment = float(np.dot(current_z_axis, world_z_axis))
            alignment = float(np.clip(alignment, -1.0, 1.0))

            plate_tilt_error = max(0.0, 1.0 - alignment)

            reward_plate_level = -4.0 * np.clip(
                (plate_tilt_error / 0.10) ** 2,
                0.0,
                10.0
            )

            # =========================================================
            # 6. 两个底座距离和中点约束
            # =========================================================
            # =========================================================
            # 6. 两个底座队形约束
            # 目标：
            #   1. 两个底盘中点追当前目标点 target_pos
            #   2. 两个底盘之间距离保持稳定
            #   3. 防止一个车追过去，另一个车被甩开
            #   4. 防止两个车左右散开导致盘子被拉斜
            # =========================================================
            base1_xy = self.Dcmm.data.body("base_link").xpos[:2].copy()
            base2_xy = self.Dcmm.data.body("base_link_copy").xpos[:2].copy()

            base_mid_xy = 0.5 * (base1_xy + base2_xy)

            # 使用 reward 当前正在追的目标点，而不是另外重新取 obj_xy
            # 这样底盘队形和盘子目标保持一致
            target_xy = target_pos[:2].copy()

            # ---------------------------------------------------------
            # 6.1 两个底盘中点追目标
            # ---------------------------------------------------------
            base_mid_target_error = float(np.linalg.norm(base_mid_xy - target_xy))

            # 为了兼容你后面原来的日志变量名，保留这个名字
            base_mid_object_error = base_mid_target_error

            reward_base_mid_object = (
                8.0 * gaussian(base_mid_target_error, 1.2)
                + 12.0 * gaussian(base_mid_target_error, 0.50)
                - 4.0 * float(np.clip(base_mid_target_error, 0.0, 3.0))
            )

            # ---------------------------------------------------------
            # 6.2 两个底盘之间距离保持稳定
            # ---------------------------------------------------------
            base_link_dist = float(np.linalg.norm(base1_xy - base2_xy))

            # 你原来允许 1.0 ~ 1.1，这里直接把目标距离设为中间值 1.05
            desired_base_dist = 1.05
            base_dist_error = abs(base_link_dist - desired_base_dist)

            reward_base_dist = -12.0 * float(np.clip(
                (base_dist_error / 0.1) ** 2,
                0.0,
                10.0
            ))

            if raw_success or self.has_success:
                reward_post_catch_base_dist = -20.0 * float(np.clip(
                    (base_dist_error / 0.08) ** 2,
                    0.0,
                    5.0
                ))
            else:
                reward_post_catch_base_dist = 0.0

            # =========================================================
            # 7. 两个底座朝向约束
            # =========================================================
            base1_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link").xquat[0],
                self.Dcmm.data.body("base_link").xquat[3]
            ))

            base2_yaw = float(quat2theta(
                self.Dcmm.data.body("base_link_copy").xquat[0],
                self.Dcmm.data.body("base_link_copy").xquat[3]
            ))

            base_vec = base2_xy - base1_xy
            base_line_yaw = float(math.atan2(base_vec[1], base_vec[0]))

            base1_face_line_error = abs(math.atan2(
                math.sin(base1_yaw - base_line_yaw),
                math.cos(base1_yaw - base_line_yaw)
            ))

            base2_face_line_error = abs(math.atan2(
                math.sin(base2_yaw - (base_line_yaw + math.pi)),
                math.cos(base2_yaw - (base_line_yaw + math.pi))
            ))

            reward_base_yaw_face = -4 * float(np.clip(
                ((base1_face_line_error + base2_face_line_error) / 0.3) ** 2,
                0.0,
                5.0
            ))

            # =========================================================
            # 8. 两个底座世界坐标速度同步
            # 对向底盘：局部动作可以相反，但转到世界坐标后应该一致。
            # =========================================================
            base_action_all = np.concatenate([
                np.asarray(ctrl["base"][:2], dtype=np.float64).reshape(-1),
                np.asarray(ctrl["base_copy"][:2], dtype=np.float64).reshape(-1),
            ])

            base1_world_vel_cmd = np.array([
                math.cos(base1_yaw) * base_action_all[0] - math.sin(base1_yaw) * base_action_all[1],
                math.sin(base1_yaw) * base_action_all[0] + math.cos(base1_yaw) * base_action_all[1],
            ], dtype=np.float64)

            base2_world_vel_cmd = np.array([
                math.cos(base2_yaw) * base_action_all[2] - math.sin(base2_yaw) * base_action_all[3],
                math.sin(base2_yaw) * base_action_all[2] + math.cos(base2_yaw) * base_action_all[3],
            ], dtype=np.float64)

            base_world_vel_error = float(np.linalg.norm(
                base1_world_vel_cmd - base2_world_vel_cmd
            ))

            base_world_vx_error = abs(float(
                base1_world_vel_cmd[0] - base2_world_vel_cmd[0]
            ))

            base_world_vy_error = abs(float(
                base1_world_vel_cmd[1] - base2_world_vel_cmd[1]
            ))

            # 原来最大只有 -2.5，太弱；现在最大约 -10
            reward_vel_sync = (
                -2.0 * float(np.clip(
                    (base_world_vel_error / 0.25) ** 2,
                    0.0,
                    5.0
                ))
            )

            # =========================================================
            # 9. 两根杆高度同步
            # =========================================================
            bar1_z = float(self.Dcmm.data.body("bar_left").xpos[2])
            bar2_z = float(self.Dcmm.data.body("bar_right").xpos[2])
            bar_height_error = abs(bar1_z - bar2_z)

            reward_bar_level = -2.0 * np.clip(
                bar_height_error / 0.05,
                0.0,
                3.0
            )

            # =========================================================
            # 10. 底盘 yaw 相邻帧变化惩罚
            # =========================================================
            prev_base1_yaw = float(getattr(self, "prev_base1_yaw_for_reward", base1_yaw))
            prev_base2_yaw = float(getattr(self, "prev_base2_yaw_for_reward", base2_yaw))

            base1_yaw_delta = abs(math.atan2(
                math.sin(base1_yaw - prev_base1_yaw),
                math.cos(base1_yaw - prev_base1_yaw)
            ))

            base2_yaw_delta = abs(math.atan2(
                math.sin(base2_yaw - prev_base2_yaw),
                math.cos(base2_yaw - prev_base2_yaw)
            ))

            base_yaw_delta = max(base1_yaw_delta, base2_yaw_delta)

            reward_base_yaw_delta = -10 * float(np.clip(
                (base_yaw_delta / 0.02) ** 2,
                0.0,
                10.0
            ))

            self.prev_base1_yaw_for_reward = base1_yaw
            self.prev_base2_yaw_for_reward = base2_yaw

            # =========================================================
            # 11. 两个机械臂输出世界坐标同步
            # =========================================================
            arm_action = np.asarray(ctrl["arm"][:3], dtype=np.float64)
            arm_copy_action = np.asarray(ctrl["arm_copy"][:3], dtype=np.float64)

            base_action = np.asarray(ctrl["base"][:2], dtype=np.float64)
            base_copy_action = np.asarray(ctrl["base_copy"][:2], dtype=np.float64)

            arm1_world_delta = np.array([
                math.cos(base1_yaw) * arm_action[0] - math.sin(base1_yaw) * arm_action[1],
                math.sin(base1_yaw) * arm_action[0] + math.cos(base1_yaw) * arm_action[1],
                arm_action[2],
            ], dtype=np.float64)

            arm2_world_delta = np.array([
                math.cos(base2_yaw) * arm_copy_action[0] - math.sin(base2_yaw) * arm_copy_action[1],
                math.sin(base2_yaw) * arm_copy_action[0] + math.cos(base2_yaw) * arm_copy_action[1],
                arm_copy_action[2],
            ], dtype=np.float64)

            arm_output_world_xy_error = float(np.linalg.norm(
                arm1_world_delta[:2] - arm2_world_delta[:2]
            ))

            arm_output_world_z_error = abs(float(
                arm1_world_delta[2] - arm2_world_delta[2]
            ))

            dt = max(1.0 / float(self.fps), 1e-6)

            base1_world_delta = np.array([
                math.cos(base1_yaw) * base_action[0] - math.sin(base1_yaw) * base_action[1],
                math.sin(base1_yaw) * base_action[0] + math.cos(base1_yaw) * base_action[1],
            ], dtype=np.float64) * dt

            base2_world_delta = np.array([
                math.cos(base2_yaw) * base_copy_action[0] - math.sin(base2_yaw) * base_copy_action[1],
                math.sin(base2_yaw) * base_copy_action[0] + math.cos(base2_yaw) * base_copy_action[1],
            ], dtype=np.float64) * dt

            support1_world_delta_xy = base1_world_delta + arm1_world_delta[:2]
            support2_world_delta_xy = base2_world_delta + arm2_world_delta[:2]

            support_world_xy_error = float(np.linalg.norm(
                support1_world_delta_xy - support2_world_delta_xy
            ))

            # 原来这个项乘 0.2 后太弱；这里直接让该项本身更有效
            reward_arm_mirror_sync = (
                -1.0 * float(np.clip(
                    (arm_output_world_xy_error / 0.020) ** 2,
                    0.0,
                    3.0
                ))
                -1.5 * float(np.clip(
                    (arm_output_world_z_error / 0.020) ** 2,
                    0.0,
                    3.0
                ))
                -1.0 * float(np.clip(
                    (support_world_xy_error / 0.030) ** 2,
                    0.0,
                    3.0
                ))
            )

            arm_mirror_sync_error = (
                arm_output_world_xy_error
                + arm_output_world_z_error
                + support_world_xy_error
            )
            # if reward_object_on_plate > 0.0:
            #     print(reward_object_on_plate)
            # =========================================================
            # 12. 控制惩罚
            # =========================================================
            reward_ctrl = -0.3 * self.norm_ctrl(
                ctrl,
                ["base", "base_copy", "arm", "arm_copy", "hand", "hand_copy"]
            )
            if raw_success or self.has_success:
                reward_post_catch_ctrl = -1.2 * self.norm_ctrl(
                    ctrl,
                    ["base", "base_copy", "arm", "arm_copy"]
                )
            else:
                reward_post_catch_ctrl = 0.0
            # =========================================================
            # 13. IK 惩罚
            # =========================================================
            reward_ik = 0.0
            if not self.arm_limit:
                reward_ik = -2.0

            # =========================================================
            # 14. 底盘防翻车
            # =========================================================
            base1_rmat = self.Dcmm.data.body("base_link").xmat.reshape(3, 3)
            base2_rmat = self.Dcmm.data.body("base_link_copy").xmat.reshape(3, 3)

            base1_z_axis = base1_rmat[:, 2]
            base2_z_axis = base2_rmat[:, 2]

            world_z = np.array([0.0, 0.0, 1.0])

            base1_upright = float(np.clip(np.dot(base1_z_axis, world_z), -1.0, 1.0))
            base2_upright = float(np.clip(np.dot(base2_z_axis, world_z), -1.0, 1.0))

            base1_tilt_error = max(0.0, 1.0 - base1_upright)
            base2_tilt_error = max(0.0, 1.0 - base2_upright)

            reward_base_upright = -50.0 * (
                np.clip((base1_tilt_error / 0.01) ** 2, 0.0, 10.0)
                + np.clip((base2_tilt_error / 0.01) ** 2, 0.0, 10.0)
            )
            #print(reward_success)
            # =========================================================
            # 15. 总奖励
            # =========================================================
            rewards = (
                reward_target_far
                + reward_target_mid
                + reward_target_xy
                + reward_target_precision
                + reward_target_improve
                + reward_target_distance_penalty
                + reward_xy_far_penalty
                + reward_z_far_penalty
                + reward_plate_vel_to_target

                + reward_object_on_plate
                + reward_success
                + reward_stability
                + reward_object_on_edge

                + reward_plate_level
                + reward_base_dist
                + reward_base_mid_object
                + reward_base_yaw_delta
                + reward_vel_sync
                + reward_bar_level
                + reward_arm_mirror_sync
                + reward_base_yaw_face

                + reward_ctrl
                + reward_ik
                + reward_collision
                + 0.5 * reward_base_upright
                +reward_post_catch_base_dist
                +reward_post_catch_ctrl
            )

            # =========================================================
            # 16. 日志
            # =========================================================
            info["reward_target_far"] = float(reward_target_far)
            info["reward_target_mid"] = float(reward_target_mid)
            info["reward_target_xy"] = float(reward_target_xy)
            info["reward_target_precision"] = float(reward_target_precision)
            info["reward_target_improve"] = float(reward_target_improve)
            info["reward_target_distance_penalty"] = float(reward_target_distance_penalty)
            info["reward_xy_far_penalty"] = float(reward_xy_far_penalty)
            info["reward_z_far_penalty"] = float(reward_z_far_penalty)
            info["reward_plate_vel_to_target"] = float(reward_plate_vel_to_target)

            info["success_object_on_edge"] = float(object_on_edge)
            info["reward_object_on_edge"] = float(reward_object_on_edge)
            info["reward_object_on_plate"] = float(reward_object_on_plate)
            info["reward_success"] = float(reward_success)
            info["reward_stability"] = float(reward_stability)

            info["reward_base_mid_object"] = float(reward_base_mid_object)
            info["debug_base_mid_object_error"] = float(base_mid_object_error)

            info["debug_base_mid_target_error"] = float(base_mid_target_error)
            info["debug_base_dist_error"] = float(base_dist_error)

            info["reward_vel_sync"] = float(reward_vel_sync)
            info["debug_base_world_vel_error"] = float(base_world_vel_error)
            info["debug_base_world_vx_error"] = float(base_world_vx_error)
            info["debug_base_world_vy_error"] = float(base_world_vy_error)

            info["reward_bar_level"] = float(reward_bar_level)
            info["reward_base_yaw_face"] = float(reward_base_yaw_face)
            info["reward_base_yaw_delta"] = float(reward_base_yaw_delta)
            info["reward_collision"] = float(reward_collision)
            info["reward_plate_level"] = float(reward_plate_level)
            info["reward_base_dist"] = float(reward_base_dist)
            info["reward_arm_mirror_sync"] = float(reward_arm_mirror_sync)
            info["reward_ctrl"] = float(reward_ctrl)
            info["reward_ik"] = float(reward_ik)
            info["reward_base_upright"] = float(reward_base_upright)

            info["debug_target_error"] = float(target_error)
            info["debug_xy_error"] = float(xy_error)
            info["debug_z_error"] = float(z_error)
            info["debug_object_plate_xy_error"] = float(object_plate_xy_error)
            info["debug_object_plate_z_error"] = float(object_plate_z_error)
            info["debug_target_object_xy_error"] = float(target_object_xy_error)
            info["debug_target_object_z_error"] = float(target_object_z_error)

            info["debug_base_pair_distance"] = float(base_link_dist)

            info["debug_arm_output_world_xy_error"] = float(arm_output_world_xy_error)
            info["debug_arm_output_world_z_error"] = float(arm_output_world_z_error)
            info["debug_support_world_xy_error"] = float(support_world_xy_error)
            info["debug_arm_mirror_sync_error"] = float(arm_mirror_sync_error)

            info["reward_total"] = float(rewards)

            return rewards
    

    def _step_mujoco_simulation(self, action_dict):
            # print(f"--- 断点 1 (初始状态) ---")
            # print(f"target_arm_qpos: {self.Dcmm.target_arm_qpos}")
            self.Dcmm.target_base_vel[0:2] = action_dict['base'][0:2]
            self.Dcmm.target_base_vel_copy[0:2] = action_dict['base_copy'][0:2]
            # if True:
            #     base_cmd = np.asarray(action_dict["base"][0:2], dtype=np.float64)
            #     base_copy_cmd = np.asarray(action_dict["base_copy"][0:2], dtype=np.float64)

            #     base_angle = np.arctan2(base_cmd[1], base_cmd[0])
            #     base_copy_angle = np.arctan2(base_copy_cmd[1], base_copy_cmd[0])

            #     base_copy_angle_prev = getattr(self, "base_copy_angle_prev_debug", base_copy_angle)
            #     base_copy_angle_jump = np.arctan2(
            #         np.sin(base_copy_angle - base_copy_angle_prev),
            #         np.cos(base_copy_angle - base_copy_angle_prev)
            #     )
            #     self.base_copy_angle_prev_debug = base_copy_angle
            #     mirror_xy_neg = base_cmd + base_copy_cmd

            #     mirror_x_neg_y_same = np.array([
            #         base_cmd[0] + base_copy_cmd[0],
            #         base_cmd[1] - base_copy_cmd[1],
            #     ], dtype=np.float64)
            #     print(
            #         f"[BASE DEBUG] "
            #         f"t={self.Dcmm.data.time - self.start_time:.3f} | "
            #         f"base={base_cmd} | "
            #         f"base_copy={base_copy_cmd} | "
            #         f"mirror_negxy={mirror_xy_neg} | "
            #         f"mirror_xneg_ysame={mirror_x_neg_y_same} | "
            #         f"angle={base_angle:.2f}/{base_copy_angle:.2f} | "
            #         f"copy_jump={base_copy_angle_jump:.2f}"
            #     )

            action_arm = np.concatenate((action_dict["arm"], np.zeros(3)))
            action_arm_copy = np.concatenate((action_dict["arm_copy"], np.zeros(3)))

            # print("action_dict['arm']:", action_dict["arm"])
            # print("action_dict['arm_copy']:", action_dict["arm_copy"])
            # print("action_arm:", action_arm)
            # print("action_arm_copy:", action_arm_copy)
            if self.keep_ee_level:
                # 推荐模式：
                # 不固定整个四元数，而是让 IK 每步构造“z 轴朝上、yaw 尽量保持当前”的水平目标姿态。
                result_QP, _ = self.Dcmm.move_ee_pose(
                    action_arm,
                    keep_level=True
                )

                result_QP_copy, _ = self.Dcmm.move_ee_pose_copy(
                    action_arm_copy,
                    keep_level=True
                )

            else:
                # 原始模式：
                # 后三维为 0，因此等价于保持当前末端姿态。
                result_QP, _ = self.Dcmm.move_ee_pose(action_arm)
                result_QP_copy, _ = self.Dcmm.move_ee_pose_copy(action_arm_copy)

            if np.isnan(result_QP[0]).any():
                print("!!! 警告: 主臂 IK 求解器返回了 NaN !!!")
            if np.isnan(result_QP_copy[0]).any():
                print("!!! 警告: 从臂 IK 求解器返回了 NaN !!!")

            self.arm_limit_left = bool(result_QP[1])
            self.arm_limit_right = bool(result_QP_copy[1])
            self.arm_limit = self.arm_limit_left and self.arm_limit_right

            if self.arm_limit_left:
                self.Dcmm.target_arm_qpos[:] = result_QP[0]

            if self.arm_limit_right:
                self.Dcmm.target_arm_qpos_copy[:] = result_QP_copy[0]

            self.Dcmm.action_hand2qpos(action_dict["hand"])
            self.Dcmm.action_hand2qpos_copy(action_dict["hand_copy"])

            # Add Target Action to the Buffer
            self.update_target_ctrl()

            # Reset the Criteria for Successfully Touch
            self.step_touch = False

            for _ in range(self.steps_per_policy):
                # Update the control command according to the latest policy output
                #ctrl = self._get_ctrl()
                self.Dcmm.data.ctrl[:-1] = self._get_ctrl()

                if self.render_per_step:
                    img = self.render()

                # ================== 【保持不变：轨迹逻辑开始】 ==================

                # =========================================================
                # 方案一新增：
                # 记录进入物体运动逻辑前，物体是否已经抛出。
                #
                # 这样后面可以检测：
                #   was_object_throw == False
                #   self.object_throw == True
                #
                # 这个瞬间就是 WAIT -> THROWN 的第一次切换。
                # =========================================================
                was_object_throw = bool(self.object_throw)

                # 1. 计算物体已经运动的总时间
                current_move_time = (
                    self.Dcmm.data.time
                    - self.start_time
                    - self.object_static_time
                )

                # 阶段 A：物体静止期
                if self.Dcmm.data.time - self.start_time < self.object_static_time:
                    self.Dcmm.set_throw_pos_vel(
                        pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                        velocity=np.zeros(6)
                    )
                    self.Dcmm.data.ctrl[-1] = (
                        self.random_mass * -self.Dcmm.model.opt.gravity[2]
                    )

                # 阶段 B：物体运动期
                else:
                    # 无论直线还是曲线，都始终施加力抵消重力
                    #self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
                    if self.task == "Tracking":
                        self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
                    else:
                        self.Dcmm.data.ctrl[-1] = 0

                    # --- 情况 1: 直线运动 ---
                    if self.trajectory_type == 'throw':
                        if not self.object_throw:
                            # 只有第一下给初速度
                            self.Dcmm.set_throw_pos_vel(
                                pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                velocity=self.object_vel6d[:]
                            )
                            self.object_throw = True

                        # 之后靠物理引擎惯性飞行，不需要在这里写代码

                    # --- 情况 2: 曲线运动 ---
                    elif self.trajectory_type == 'curve':
                        self.object_throw = True

                        # 基础线性位置 = 起点 + 速度 * 时间
                        target_pos = (
                            self.object_pos3d
                            + self.object_vel6d[:3] * current_move_time
                        )
                        target_vel = self.object_vel6d[:3].copy()

                        # 计算正弦偏移量和速度偏移量
                        sine_offset = self.curve_amp * math.sin(
                            self.curve_freq * current_move_time + self.curve_phase
                        )
                        sine_vel_offset = self.curve_amp * self.curve_freq * math.cos(
                            self.curve_freq * current_move_time + self.curve_phase
                        )

                        # 根据初始化时选定的轴叠加偏移
                        if self.curve_axis == 'y':
                            target_pos[1] += sine_offset
                            target_vel[1] += sine_vel_offset
                        elif self.curve_axis == 'z':
                            target_pos[2] += sine_offset
                            target_vel[2] += sine_vel_offset

                        # 每一帧都强行修正物体的位置和速度，实现曲线效果
                        self.Dcmm.set_throw_pos_vel(
                            pose=np.concatenate((target_pos, self.object_q)),
                            velocity=np.concatenate((target_vel, [0, 0, 0]))
                        )

                    # --- 情况 3: 圆周运动 ---
                    elif self.trajectory_type == 'circle':
                        self.object_throw = True

                        # 获取实时底座位置
                        current_base_pos = self.Dcmm.data.body("base_link").xpos[0:2]

                        # 当前角度
                        current_angle = (
                            self.circle_start_angle
                            + self.circle_omega * current_move_time
                        )

                        # 计算目标位置
                        target_x = (
                            current_base_pos[0]
                            + self.circle_radius * math.cos(current_angle)
                        )
                        target_y = (
                            current_base_pos[1]
                            + self.circle_radius * math.sin(current_angle)
                        )
                        target_pos = np.array(
                            [target_x, target_y, self.object_pos3d[2]]
                        )

                        # 计算目标速度
                        base_vel = self.Dcmm.data.qvel[0:2]

                        target_vx = (
                            -self.circle_radius
                            * self.circle_omega
                            * math.sin(current_angle)
                            + base_vel[0]
                        )
                        target_vy = (
                            self.circle_radius
                            * self.circle_omega
                            * math.cos(current_angle)
                            + base_vel[1]
                        )
                        target_vel = np.array([target_vx, target_vy, 0.0])

                        # 强行修正物体状态
                        self.Dcmm.set_throw_pos_vel(
                            pose=np.concatenate((target_pos, self.object_q)),
                            velocity=np.concatenate((target_vel, [0, 0, 0]))
                        )


                mujoco.mj_step(self.Dcmm.model, self.Dcmm.data)
                mujoco.mj_rnePostConstraint(self.Dcmm.model, self.Dcmm.data)
                #self._debug_base_yaw_drift(ctrl=action_dict, tag="after_mj_step")
                # Update the contact information
                self.contacts = self._get_contacts()

                object_on_plate = self.contacts["object_on_plate"]
                any_base_collision = self.contacts["any_base_collision"]

                if any_base_collision:
                    self.terminated = True
                    print(colored("!!! Base Collided !!!", "red"))

                if self.step_touch == False:
                    if self.task == "Tracking":
                        if object_on_plate:
                            self.step_touch = True

                if self.task == "Tracking":
                    pass

                if self.terminated:
                    break   
 

    def step(self, action):#这里面的action是从PPO传进来的，是[-1,1]*denorm
        ''' 
        actions_dict = {
                'arm': arm_tensor,4维
                'base': base_tensor,2维
                'hand': hand_tensor,12维
            }
            接动作,运行 mujoco,生 obs,生 reward,返回给 PPO
        '''
        # print("#######################action################################")
        # print(f"Base: {action['base']}, Base_Copy: {action['base_copy']}")
        # print("#######################action################################")
        self.steps += 1
        self.steps_since_replan += 1
        self._step_mujoco_simulation(action)
        
        obs = self._get_obs()
        # 在 DcmmVecEnv.py 的 step 函数里
        def contains_nan(data):
            if isinstance(data, dict):
                # 如果是字典，递归检查每一个 value
                return any(contains_nan(v) for v in data.values())
            elif isinstance(data, (np.ndarray, list)):
                # 如果是数组或列表，直接用 numpy 检查
                arr = np.asanyarray(data)
                if np.issubdtype(arr.dtype, np.number): # 只检查数值型
                    return np.any(np.isnan(arr))
            return False
        info = self._get_info()

    #    # ================== 轨迹误差调试打印：加在这里 ==================
        if self.task == "Tracking" and self.steps < 80 and self.steps % 1 == 0:
            plate_pos = np.asarray(info["plate_pos"])
            ref_pos = np.asarray(info["plate_ref_pos"])
            target_pos = np.asarray(info["plate_target_pos"])

            traj_error = float(info["traj_error"])
            target_error = float(info["target_error"])

            alpha = float(self._traj_alpha())
            t_hit = float(getattr(self, "last_pred_t_hit", -1.0))

            ref_ok = traj_error < 0.08
            target_ok = target_error < self.traj_success_threshold#两个threshold用来判断是不是跟上ref和target了
            late = alpha > 0.95 and target_error > self.traj_success_threshold
        if self.task == 'Catching':
            pass
                #print(colored("!!! Stage Error: Object escaped during grasping4 !!!", "yellow"))
        # Design the reward function
        reward = self.compute_reward(obs, info, action)
        self.info["base_distance"] = info["base_distance"]
        self.info["base_distance_copy"] = info["base_distance_copy"]
        self.info["ee_distance"] = info["ee_distance"]
        # self.info["gripper_dist"] = info["gripper_dist"]
        # self.info["qpos_sum"] = info["qpos_sum"]
        self.info["plate_distance"] = info["plate_distance"]
        self.info["plate_z_distance"] = info["plate_z_distance"]

        self.info["plate_pos"] = info["plate_pos"]
        self.info["plate_ref_pos"] = info["plate_ref_pos"]
        self.info["plate_target_pos"] = info["plate_target_pos"]
        self.info["traj_error"] = info["traj_error"]
        self.info["target_error"] = info["target_error"]
        self.info["traj_progress"] = info["traj_progress"]
        self.info["traj_phase"] = info["traj_phase"]
        self.info["path_error"] = info.get("path_error", info["traj_error"])
        self.info["path_progress"] = info.get("path_progress", 0.0)
        self.info["path_closest_point"] = info.get(
            "path_closest_point",
            info["plate_ref_pos"]
        )
        # self.info["plate_diatance"] = info["plate_diatance"]Rendering
        imgs = self.render() if self.render_mode is not None else None
        # Update the imgs
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand']),
                               len(self.action_buffer['base_copy']),
                               len(self.action_buffer['arm_copy']),
                               len(self.action_buffer['hand_copy'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))
        #############################下面是碰到就停的版本
        # if self.task == "Catching":
        #     # 慢速 Catching 阶段：
        #     # 只要接住一次，就让 episode 成功结束。
        #     # 这样 PPO 能明确知道：物体落到 frame_bottom 是好结果。

        #     if bool(info.get("is_success", 0.0)):
        #         truncated = True
        #         terminated = False

        #     elif self.contacts.get("object_failed", False):
        #         truncated = False
        #         terminated = True

        #     elif self.contacts.get("any_base_collision", False):
        #         truncated = False
        #         terminated = True

        #     elif self.contacts.get("any_arm_collision", False):
        #         truncated = False
        #         terminated = True

        #     elif info["env_time"] > self.env_time:
        #         truncated = False
        #         terminated = True

        #     else:
        #         truncated = False
        #         terminated = False
        ###################################上面是碰到就停的版本
        if self.task == "Catching":
            # =====================================================
            # Catching 结束逻辑：
            # 成功判断逻辑仍然使用原来的 info["is_success"]
            #
            # 现在修改为：
            #   1. 没成功前：失败接触 / 碰撞 / 超时 都算失败
            #   2. 一旦成功过：不立刻结束
            #   3. 成功后一直运行到 env_time
            #   4. 到 env_time 时，如果成功过，则 truncated=True，统计成功
            # =====================================================

            has_success = bool(info.get("is_success", 0.0))

            # -----------------------------------------------------
            # 已经成功过：
            # 不立刻结束，也不因为后续 object_failed / collision 提前失败
            # 一直跑到环境时间结束
            # -----------------------------------------------------
            if has_success:
                if info["env_time"] > self.env_time:
                    # 成功 episode 跑完整个 env_time 后结束
                    truncated = True
                    terminated = False
                else:
                    # 成功了，但还没到 env_time，继续运行
                    truncated = False
                    terminated = False

                    # 防止之前某些接触把 self.terminated 置成 True
                    self.terminated = False

            # -----------------------------------------------------
            # 还没有成功：
            # 保持原来的失败逻辑
            # -----------------------------------------------------
            elif self.contacts.get("object_failed", False):
                truncated = False
                terminated = True

            elif self.contacts.get("any_base_collision", False):
                truncated = False
                terminated = True

            elif self.contacts.get("any_arm_collision", False):
                truncated = False
                terminated = True

            elif info["env_time"] > self.env_time:
                # 没成功但时间到了，算失败
                truncated = False
                terminated = True

            else:
                # 没成功、没失败、没超时，继续运行
                truncated = False
                terminated = False
        elif self.task == "Tracking":
            tracking_success = self._check_tracking_success(info)
            truncated = bool(tracking_success)
            terminated = self.terminated

            if info["env_time"] > self.env_time:
                if not bool(info.get("is_success", 0.0)):
                    truncated = False
                    terminated = True
        # elif self.task == "Tracking":
        #     if self.step_touch:
        #         # print("Tracking Success!!!!!!")
        #         truncated = True
        #     else: truncated = False

        elif self.task == "Tracking":
            tracking_success = self._check_tracking_success(info)

            # 你现在 PPO 里是用 truncates 统计 success，
            # 所以成功时必须让 truncated=True
            truncated = bool(tracking_success)
        
        #terminated = self.terminated
        if info["env_time"] > self.env_time:
            if self.task == "Tracking":
                # Tracking：没成功到时间，算失败
                if not bool(info.get("is_success", 0.0)):
                    truncated = False
                    terminated = True

            elif self.task == "Catching":
                # Catching：如果曾经接住并保持过，到时间算成功
                if bool(info.get("is_success", 0.0)):
                    truncated = True
                    terminated = False
                else:
                    truncated = False
                    terminated = True
        # if done and self.task == "Catching":
        #     print(
        #         "[CATCH END] "
        #         f"time={info['env_time']:.3f} "
        #         f"terminated={terminated} "
        #         f"truncated={truncated} "
        #         f"is_success={info.get('is_success', None)} "
        #         f"success_counter={info.get('success_counter', None)} "
        #         f"object_failed={self.contacts.get('object_failed', False)} "
        #         f"object_on_plate={self.contacts.get('object_on_plate', False)} "
        #         f"base_collision={self.contacts.get('any_base_collision', False)} "
        #         f"arm_collision={self.contacts.get('any_arm_collision', False)} "
        #         f"reward_success={info.get('reward_success', None)} "
        #         f"reward_stability={info.get('reward_stability', None)} "
        #         f"reward_collision={info.get('reward_collision', None)} "
        #         f"object_z={self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]:.3f}"
        #     )
        done = terminated or truncated

        # ==========================================================
        # 明确告诉 PPO：这个 episode 结束时是不是成功结束
        # ==========================================================
        # 注意：
        #   is_success 表示这个 episode 曾经达到成功条件；
        #   done_success 表示这一次 done 是不是成功 done。
        #
        # 这样 PPO 训练统计时不用猜：
        #   truncated=True 到底是成功，还是普通时间截断。
        # ==========================================================
        episode_has_success = bool(info.get("is_success", 0.0))

        info["terminated"] = float(terminated)
        info["truncated"] = float(truncated)
        info["done"] = float(done)

        # 当前你的逻辑里：
        #   成功结束一般是 truncated=True, terminated=False
        #   失败结束一般是 terminated=True, truncated=False
        info["done_success"] = float(
            bool(done)
            and bool(truncated)
            and (not bool(terminated))
            and episode_has_success
        )

        # 额外保留一个 episode_success 字段，方便 PPO 或 TensorBoard 直接读取
        info["episode_success"] = float(
            bool(done)
            and episode_has_success
        )
        # if done and self.task == "Catching":
        #     print(
        #         "\n[CATCH EPISODE END DEBUG] "
        #         f"env_time={float(info['env_time']):.4f}, "
        #         f"env_time_limit={float(self.env_time):.4f}, "
        #         f"steps={self.steps}, "
        #         f"terminated={terminated}, "
        #         f"truncated={truncated}, "
        #         f"done={done}, "
        #         f"is_success={info.get('is_success', None)}, "
        #         f"done_success={info.get('done_success', None)}, "
        #         f"episode_success={info.get('episode_success', None)}, "
        #         f"success_counter={info.get('success_counter', None)}, "
        #         f"success_hold_steps={getattr(self, 'success_hold_steps', None)}, "
        #         f"success_has_success={info.get('success_has_success', None)}, "
        #         f"reward_success={info.get('reward_success', None)}, "
        #         f"reward_object_on_plate={info.get('reward_object_on_plate', None)}, "
        #         f"reward_stability={info.get('reward_stability', None)}, "
        #         f"object_on_plate={self.contacts.get('object_on_plate', False)}, "
        #         f"object_failed={self.contacts.get('object_failed', False)}, "
        #         f"any_base_collision={self.contacts.get('any_base_collision', False)}, "
        #         f"any_arm_collision={self.contacts.get('any_arm_collision', False)}"
        #     )
        if done:
            # TEST ONLY
            # self.reset()
            pass
        # ==========================================================
        # DEBUG 1: 环境端 done 时的成功状态
        # 用来确认环境自己到底有没有判成功
        # ==========================================================
        return obs, reward, terminated, truncated, info
        #return obs, reward, terminated, truncated, info

    def preprocess_depth_with_mask(self, rgb_img, depth_img, 
                                   depth_threshold=3.0, 
                                   num_white_points_range=(5, 15),
                                   point_size_range=(1, 5)):
        # Define RGB Filter
        lower_rgb = np.array([5, 0, 0])
        upper_rgb = np.array([255, 15, 15])
        rgb_mask = cv.inRange(rgb_img, lower_rgb, upper_rgb)
        depth_mask = cv.inRange(depth_img, 0, depth_threshold)
        combined_mask = np.logical_and(rgb_mask, depth_mask)
        # Apply combined mask to depth image
        masked_depth_img = np.where(combined_mask, depth_img, 0)
        # Calculate mean depth within combined mask
        masked_depth_mean = np.nanmean(np.where(combined_mask, depth_img, np.nan))
        # Generate random number of white points
        num_white_points = np.random.randint(num_white_points_range[0], num_white_points_range[1])
        # Generate random coordinates for white points
        random_x = np.random.randint(0, depth_img.shape[1], size=num_white_points)
        random_y = np.random.randint(0, depth_img.shape[0], size=num_white_points)
        # Generate random sizes for white points in the specified range
        random_sizes = np.random.randint(point_size_range[0], point_size_range[1], size=num_white_points)
        # Create masks for all white points at once
        y, x = np.ogrid[:masked_depth_img.shape[0], :masked_depth_img.shape[1]]
        point_masks = ((x[..., None] - random_x) ** 2 + (y[..., None] - random_y) ** 2) <= random_sizes ** 2
        # Update masked depth image with the white points
        masked_depth_img[np.any(point_masks, axis=2)] = np.random.uniform(1.5, 3.0)

        return masked_depth_img, masked_depth_mean

    def render(self):
        imgs = np.zeros((0, self.img_size[0], self.img_size[1]))
        imgs_depth = np.zeros((0, self.img_size[0], self.img_size[1]))
        # imgs_rgb = np.zeros((self.img_size[0], self.img_size[1], 3))
        for camera_name in self.camera_name:
            if self.render_mode == "human":
                self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                return imgs
            elif self.render_mode != "depth_rgb_array":
                img = self.mujoco_renderer.render(
                    self.render_mode, camera_name = camera_name
                )
                if self.imshow_cam and self.render_mode == "rgb_array":
                    cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                # Converts the depth array valued from 0-1 to real meters
                elif self.render_mode == "depth_array":
                    img = self.Dcmm.depth_2_meters(img)
                    if self.imshow_cam:
                        depth_norm = np.zeros(img.shape, dtype=np.uint8)
                        cv.convertScaleAbs(img, depth_norm, alpha=(255.0/img.max()))
                        cv.imshow(camera_name+"_depth", depth_norm)
                        cv.waitKey(1)
                    img = np.expand_dims(img, axis=0)
            else:
                img_rgb = self.mujoco_renderer.render(
                    "rgb_array", camera_name = camera_name
                )
                img_depth = self.mujoco_renderer.render(
                    "depth_array", camera_name = camera_name
                )   
                # Converts the depth array valued from 0-1 to real meters
                img_depth = self.Dcmm.depth_2_meters(img_depth)
                img_depth, _ = self.preprocess_depth_with_mask(img_rgb, img_depth)
                if self.imshow_cam:
                    cv.imshow(camera_name+"_rgb", cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
                    cv.imshow(camera_name+"_depth", img_depth)
                    cv.waitKey(1)
                img_depth = cv.resize(img_depth, (self.img_size[1], self.img_size[0]))
                img_depth = np.expand_dims(img_depth, axis=0)
                imgs_depth = np.concatenate((imgs_depth, img_depth), axis=0)
            # Sync the viewer (if exists) with the data
            if self.Dcmm.viewer != None: 
                self.Dcmm.viewer.sync()
        if self.render_mode == "depth_rgb_array":
            # Only keep the depth image
            imgs = imgs_depth
        return imgs

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
        if self.Dcmm.viewer != None: self.Dcmm.viewer.close()

    def run_test(self):
        global cmd_lin_x, cmd_lin_y, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
        self.reset()
        action = np.zeros(8)
        while True:
            # Note: action's dim = 18, which includes 2 for the base, 4 for the arm, and 12 for the hand
            # print("##### stage: ", self.stage)
            # Keyboard control
            action[0:2] = np.array([cmd_lin_x, cmd_lin_y])#将底座控制命令写入动作向量前两位
            if trigger_delta:#是否触发机械臂 delta 动作
                print("delta_xyz: ", delta_xyz)
                action[2:6] = np.array([delta_xyz, delta_xyz, delta_xyz, delta_xyz])
                trigger_delta = False
            else:
                action[2:6] = np.zeros(4)
            if trigger_delta_hand:
                print("delta_xyz_hand: ", delta_xyz_hand)
                action[6:8] = np.ones(2)*delta_xyz_hand
                trigger_delta_hand = False
            else:
                action[6:8] = np.zeros(2)
            base_tensor = action[:2]
            arm_tensor = action[2:6]
            hand_tensor = action[6:8]
            actions_dict = {
                'arm': arm_tensor,
                'base': base_tensor,
                'hand': hand_tensor
            }
            # print("self.Dcmm.data.body('link6'):", self.Dcmm.data.body('link6'))
            observation, reward, terminated, truncated, info = self.step(actions_dict)

if __name__ == "__main__":
    os.chdir('../../')#把当前工作目录切换到上两级目录。


    parser = argparse.ArgumentParser(description="Args for DcmmVecEnv")
    parser.add_argument('--viewer', action='store_true', help="open the mujoco.viewer or not")
    parser.add_argument('--imshow_cam', action='store_true', help="imshow the camera image or not")
    args = parser.parse_args()
    print("args: ", args)
    env = DcmmVecEnv(task='Catching', object_name='object', render_per_step=False, 
                    print_reward=False, print_info=False, 
                    print_contacts=False, print_ctrl=False, 
                    print_obs=True, camera_name = ["top"],
                    render_mode="rgb_array", imshow_cam=args.imshow_cam, 
                    viewer = args.viewer, object_eval=False,
                    env_time = 2.5, steps_per_policy=20)
    env.run_test()