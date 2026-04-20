"""
Author: Yuanhang Zhang
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
        task="tracking",
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
        self.gripper_width_max = 0.3  # 夹爪最大开度
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
        self.random_mass = 0.25#物体质量的扰动范围
        self.object_static_time = 0.75#物体在初始状态下静止不动的时间。仿真开始的前 0.75 秒内，物体保持静止
        self.object_throw = False#是否让物体在环境中被抛掷或动态移动
        self.object_train = True#指示当前环境处于训练阶段还是评估阶段。
        if object_eval: self.set_object_eval()
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
        self.observation_space = spaces.Dict(#定义机器人环境的观测空间，一共30维
            {
                "base": spaces.Dict({
                    "v_lin_3d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),#spaces.Box(low, high, shape, dtype)

                }),
                "base_copy": spaces.Dict({
                    "v_lin_3d": spaces.Box(-4, 4, shape=(2,), dtype=np.float32),#spaces.Box(low, high, shape, dtype)

                }),
                "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(9, 15)]),#（9,15)代表的是6个关节，这是给6个关节找上下限
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(9, 15)]),
                                            dtype=np.float32),
                }),
                "arm_copy": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(26, 32)]),#（9,15)代表的是6个关节，这是给6个关节找上下限
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(26, 32)]),
                                            dtype=np.float32),
                }),
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
                    "pos3d_copy": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d_copy": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
                    ## TODO: to be determined
                    # "shape": spaces.Box(-5, 5, shape=(2,), dtype=np.float32),
                }),
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
        self.initial_obj_pos3d = self._get_relative_object_pos3d()
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
            "object_on_plate": False     # 物体是否成功落在盘子上（用于奖励）
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
        # 盘子 ID
        plate_id = self.plate_id 
        # 被接的物体 ID
        object_id = self.object_id
        # 地面 ID
        floor_id = 0

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

        # C. 外部物体判定（最关键）：只能碰到盘子
        if object_partners.size > 0:
            for p_id in object_partners:
                # 如果碰到了不是盘子的东西（包括地面、底座、手臂等）
                if p_id != plate_id:
                    results["object_failed"] = True
                    break
            
            # 顺便判定物体是否当前正落在盘子上（可用于计算 Reward）
            if plate_id in object_partners:
                results["object_on_plate"] = True

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
        base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link").xquat[3])#确定底座朝向，相对于世界坐标系
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
    
    def _get_relative_object_pos3d(self):
        # Caclulate the object_pos3d w.r.t. the base_link
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])#底座在水平平面上的旋转角度。
        x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
                                self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
                                base_yaw)#物体在机械臂基座（arm_base）坐标系下的相对位置，但只针对 平面 XY 方
        return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#物体在 Z 方向（垂直方向）相对于机械臂基座的高度差。
        #小球相对于arm_base的位置
    def _get_relative_object_pos3d_copy(self):
        # Caclulate the object_pos3d w.r.t. the base_link
        base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link_copy").xquat[3])#底座在水平平面上的旋转角度。
        x,y = relative_position(self.Dcmm.data.body("arm_base_copy").xpos[0:2], 
                                self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
                                base_yaw)#物体在机械臂基座（arm_base）坐标系下的相对位置，但只针对 平面 XY 方
        return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base_copy").xpos[2]])#物体在 Z 方向（垂直方向）相对于机械臂基座的高度差。
        #小球相对于arm_base的位置

    def _get_relative_object_v_lin_3d(self):
        # Caclulate the object_v_lin3d w.r.t. the base_link
        base_vel = self.Dcmm.data.body("arm_base").cvel[3:6]#arm_base底座刚体在当前仿真状态下的数据，cvel代表速度[3:6]是线速度，[0:3]是角速度
        global_object_v_lin = self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3]#.joint(name) 返回 仿真中名为 name 的关节对象的数据，对于关节来说[0:3]是线速度
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])#返回该刚体在世界坐标系下的 四元数旋转，将 四元数中的 w 和 z 分量 转换为 Yaw 角
        #移动机器人底座在水平平面（XY 平面）上的旋转角，也就是 机器人“朝向”的角度。
        object_v_lin_x = math.cos(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        object_v_lin_y = -math.sin(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_object_v_lin[1]-base_vel[1])#把物体的线速度从全局坐标系转换到机器人底座的局部坐标系
        return np.array([object_v_lin_x, object_v_lin_y, global_object_v_lin[2]-base_vel[2]])#物体相对于机器人底座在竖直方向（z轴）的线速度
        #小球相对于base_link的速度，其实就是相对于aem_base的速度
    def _get_relative_object_v_lin_3d_copy(self):
        # Caclulate the object_v_lin3d w.r.t. the base_link
        base_vel = self.Dcmm.data.body("arm_base_copy").cvel[3:6]#arm_base底座刚体在当前仿真状态下的数据，cvel代表速度[3:6]是线速度，[0:3]是角速度
        global_object_v_lin = self.Dcmm.data.joint(self.Dcmm.object_name).qvel[0:3]#.joint(name) 返回 仿真中名为 name 的关节对象的数据，对于关节来说[0:3]是线速度
        base_yaw = quat2theta(self.Dcmm.data.body("base_link_copy").xquat[0], self.Dcmm.data.body("base_link_copy").xquat[3])#返回该刚体在世界坐标系下的 四元数旋转，将 四元数中的 w 和 z 分量 转换为 Yaw 角
        #移动机器人底座在水平平面（XY 平面）上的旋转角，也就是 机器人“朝向”的角度。
        object_v_lin_x = math.cos(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.sin(base_yaw) * (global_object_v_lin[1]-base_vel[1])
        object_v_lin_y = -math.sin(base_yaw) * (global_object_v_lin[0]-base_vel[0]) + math.cos(base_yaw) * (global_object_v_lin[1]-base_vel[1])#把物体的线速度从全局坐标系转换到机器人底座的局部坐标系
        return np.array([object_v_lin_x, object_v_lin_y, global_object_v_lin[2]-base_vel[2]])#物体相对于机器人底座在竖直方向（z轴）的线速度
    
    def _get_obs(self):
        ee_pos3d = self._get_relative_ee_pos3d()#机械臂末端基于arm_base的坐标
        ee_pos3d_copy = self._get_relative_ee_pos3d_copy()
        obj_pos3d = self._get_relative_object_pos3d()#物体基于arm_base的坐标
        obj_pos3d_copy = self._get_relative_object_pos3d_copy()
        # 初始化记录 qpos
        if self.init_pos:#初始坐标
            self.prev_ee_pos3d[:] = ee_pos3d[:]
            self.prev_ee_pos3d_copy[:] = ee_pos3d_copy[:]
            self.prev_obj_pos3d[:] = obj_pos3d[:]
            self.prev_obj_pos3d_copy[:] = obj_pos3d_copy[:]
            self.init_pos = False
        # Add Obs Noise (Additive self.k_obs_base/arm/hand/object)
        self.obj_pos_history.append(obj_pos3d.copy())
        # 在 obs 字典定义的上方获取
        #current_hand_qpos = np.array(self.Dcmm.data.qpos[self.hand_qpos_addr : self.hand_qpos_addr + 2])
        obs = {
            "base": {
                "v_lin_3d": self._get_base_vel() + np.random.normal(0, self.k_obs_base, 2),#这个是相对于机器人自身的速度
                #"v_lin_2d": self.Dcmm.data.qvel[0:2],
            },
            "base_copy": {
                "v_lin_3d": self._get_base_vel_copy() + np.random.normal(0, self.k_obs_base, 2),#这个是相对于机器人自身的速度
                #"v_lin_2d": self.Dcmm.data.qvel[0:2],
            },
            "arm": {
                "ee_pos3d": ee_pos3d + np.random.normal(0, self.k_obs_arm, 3),#机械臂末端基于arm_base的坐标
                "ee_quat": self._get_relative_ee_quat() + np.random.normal(0, self.k_obs_arm, 4),#机械臂末端基于底座的相对四元数
                'ee_v_lin_3d': (ee_pos3d - self.prev_ee_pos3d)*self.fps + np.random.normal(0, self.k_obs_arm, 3),#delta机械臂末端相当于arm_base的速度
                "joint_pos": np.array(self.Dcmm.data.qpos[15:21]) + np.random.normal(0, self.k_obs_arm, 6),
            },
            "arm_copy": {
                "ee_pos3d": ee_pos3d_copy + np.random.normal(0, self.k_obs_arm, 3),#机械臂末端基于arm_base的坐标
                "ee_quat": self._get_relative_ee_quat_copy() + np.random.normal(0, self.k_obs_arm, 4),#机械臂末端基于底座的相对四元数
                'ee_v_lin_3d': (ee_pos3d_copy - self.prev_ee_pos3d_copy)*self.fps + np.random.normal(0, self.k_obs_arm, 3),#delta机械臂末端相当于arm_base的速度
                "joint_pos": np.array(self.Dcmm.data.qpos[38:44]) + np.random.normal(0, self.k_obs_arm, 6),
            },#,21,22,,,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
            # 修改这里，直接使用上面获取的 current_hand_qpos
            #"hand": current_hand_qpos + np.random.normal(0, self.k_obs_hand, 2),
            #"hand": self._get_hand_obs() + np.random.normal(0, self.k_obs_hand, 2),
            "object": {
                "pos3d": obj_pos3d + np.random.normal(0, self.k_obs_object, 3),#物体相对于arm_base的坐标
                "pos3d_copy": obj_pos3d + np.random.normal(0, self.k_obs_object, 3),
                "v_lin_3d": self._get_relative_object_v_lin_3d() + np.random.normal(0, self.k_obs_object, 3),
                "v_lin_3d_copy": self._get_relative_object_v_lin_3d_copy() + np.random.normal(0, self.k_obs_object, 3),
                #"v_lin_3d": (obj_pos3d - self.prev_obj_pos3d)*self.fps + np.random.normal(0, self.k_obs_object, 3),速度计算
                #"pos_history": np.array(self.obj_pos_history).flatten() + np.random.normal(0, self.k_obs_object, 3 * self.obj_history_len),
            },
        }
        self.prev_ee_pos3d = ee_pos3d
        self.prev_ee_pos3d_copy = ee_pos3d_copy
        self.prev_obj_pos3d = obj_pos3d
        self.prev_obj_pos3d_copy = obj_pos3d_copy
        if self.print_obs:
            print("##### print obs: \n", obs)
        return obs#obs是一个字典

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
        mv_steer = np.clip(mv_steer, -7.5, 7.5)
        mv_steer_copy = np.clip(mv_steer_copy, -7.5, 7.5)
        mv_drive = np.clip(mv_drive, -40.19, 40.19)
        mv_drive_copy = np.clip(mv_drive_copy, -40.19, 40.19)  
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
        if object_body is not None:#在这里找一个inertial的子节点
            inertial = object_body.find("inertial")#这个返回的是inertial的子节点
            if inertial is not None:#随机一个质量
                # Generate a random mass within the specified range
                #self.random_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[0])#uniform服从均匀分布
                self.random_mass = 1
                # Update the mass attribute，随机化质量
                inertial.set("mass", str(self.random_mass))#set("属性名", "属性值")，修改节点属性MuJoCo 在加载 XML 的时候会把 "0.25" 解析成 浮点数 0.25 存到它的内部模型里。所以在仿真运行时，质量确实是数字。

            joint = object_body.find("joint")
            if joint is not None:#随机阻尼
                # Generate a random damping within the specified range
                random_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])#随机阻尼系数，damping 就是控制 关节转动或移动时受到的阻力大小
                # Update the damping attribute
                joint.set("damping", str(random_damping))
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
                    # # 1. 安全起见，先移除可能存在的四元数定义，防止加载冲突
                    # object_body.attrib.pop("quat", None)
                    # object_body.attrib.pop("axisangle", None)

                    # # 2. 计算角度：
                    # # 如果 Y 轴指向前方，绕 Y 轴旋转 90 度会使 Z 轴向右倒下
                    # # Euler 设定通常为 [roll, pitch, yaw] 对应 [x, y, z]
                    # roll = 0
                    # pitch = 90  # 绕 Y 轴旋转 90 度
                    # yaw = 0 # 让躺下的物体在地面上随机指向不同方向
                    
                    # euler_str = f"{roll} {pitch} {yaw}"
                    # object_body.set("euler", euler_str)
                    # print("### Object Geom Info ###")
                    # for key, value in geom.attrib.items():
                    #     print(f"{key}: {value}")
                else:
                    # object_shape = DcmmCfg.object_shape[object_id]
                    # geom.set("type", object_shape)  # Replace "box" with the desired type
                    # object_size = np.array([np.random.uniform(low=low, high=high) for low, high in DcmmCfg.object_size[object_shape]])
                    # geom.set("size", np.array_str(object_size)[1:-1])  # Replace with the desired size
                    #     # 1. 安全起见，先移除可能存在的四元数定义，防止加载冲突
                    # # 替换原来的 geom.set("size", ...)
                    # size_str = " ".join([f"{x:.4f}" for x in object_size])
                    # geom.set("size", size_str)
                    # object_body.attrib.pop("quat", None)
                    # object_body.attrib.pop("axisangle", None)

                    #     # 2. 计算角度：
                    #     # 如果 Y 轴指向前方，绕 Y 轴旋转 90 度会使 Z 轴向右倒下
                    #     # Euler 设定通常为 [roll, pitch, yaw] 对应 [x, y, z]
                    # roll = 0
                    # pitch = 90  # 绕 Y 轴旋转 90 度
                    # yaw = 0 # 让躺下的物体在地面上随机指向不同方向
                        
                    # euler_str = f"{roll} {pitch} {yaw}"
                    # object_body.set("euler", euler_str)
                    object_mesh = DcmmCfg.object_mesh[object_id]
                    geom.set("mesh", object_mesh)
                    # # 如果你想让它极其沉重（例如 50kg），直接硬编码或从配置读取
                    # target_mass = "5000.0" 
                    # geom.set("mass", target_mass)
                    # # 1. 安全起见，先移除可能存在的四元数定义，防止加载冲突
                    # object_body.attrib.pop("quat", None)
                    # object_body.attrib.pop("axisangle", None)

                    # # 2. 计算角度：
                    # # 如果 Y 轴指向前方，绕 Y 轴旋转 90 度会使 Z 轴向右倒下
                    # # Euler 设定通常为 [roll, pitch, yaw] 对应 [x, y, z]
                    # roll = 0
                    # pitch = 90  # 绕 Y 轴旋转 90 度
                    # yaw = 0 # 让躺下的物体在地面上随机指向不同方向
        xml_str = ET.tostring(root, encoding='unicode')#ET.tostring() 的作用是 把 XML 树（ElementTree 节点对象）转换为字符串。
        
        return xml_str#xml树的str
        #训练模式时随机生成object，评估的时候用固定的object
    def random_object_pose(self):
        # Random Position用来随机生成：球的位置、速度、投掷方向、姿态和等待时间
        # x = 0.6*np.random.rand() - 0.3 # (-0.3, 0.3)底座面朝方向的右边
        x = 0.6+0.4*np.random.rand()
        y = 0.5 + 0.4 * np.random.rand() # (2.2, 2.5)底座面朝方向的前面
        # Low or High Starting Position
        low_factor = False if np.random.rand() < 0.5 else True
        # low_factor = True
        if low_factor: height = np.random.uniform(0.65,0.75 )#0.7 + 0.3 * np.random.rand()# (0.7, 1.0)low_facor为true时，从低高度里面选高度，各有50%的可能
        else: height = np.random.uniform(0.65, 0.8)#0.8 + 0.4 * np.random.rand() # (1.0, 1.6)
        # Random Velocity
        # r_vel = 1 + np.random.rand() # (1, 2)
        # alpha_vel = math.pi * (np.random.rand()*1/6 + 5/12) # alpha_vel = (5/12 * pi, 7/12 * pi)
        # # alpha_vel = math.pi * (np.random.rand()*1/3 + 1/3) # alpha_vel = (1/3 * pi, 2/3 * pi)
        # v_lin_x = r_vel * math.cos(alpha_vel) # (-0.0, -0.5)
        # v_lin_y = - r_vel * math.sin(alpha_vel) # (-0.5, -1.0)
        # v_lin_z = 0.5 * np.random.rand() + 2.0 # (2.0, 2.5)
        # if y > 2.25: v_lin_y -= 0.4
        # if height < 1.0: v_lin_z += 1
        # x 方向速度 ∈ [0.6, 1.0]
        #v_lin_x = np.random.uniform(0.6, 1.0)
        v_lin_x = 0
        # y 方向速度 ∈ [-0.25, -0.1]
        #v_lin_y = np.random.uniform(-0.25, 0.25)
        v_lin_y = 0
        # z 方向速度 ∈ [-0.1, 0.1]
        #v_lin_z = np.random.uniform(-0.05, 0.05)
        v_lin_z = 0
        self.object_pos3d = np.array([x, y, height])#向下，向右，向上
        self.object_vel6d = np.array([v_lin_x, v_lin_y, v_lin_z, 0.0, 0.0, 0.0])
# ================== 【新增修改开始】 ==================
        # 1. 随机决定轨迹类型：50% 直线 (linear)，50% 曲线 (curve)
        self.trajectory_type = np.random.choice(['linear', 'curve','circle'], p=[1.0, 0,0])

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

    def reset(self):#整个 环境在每个 episode 开头调用的 reset() 函数
        # Reset the basic simulation
        self._reset_simulation()
        self.init_ctrl = True
        self.init_pos = True
        self.vel_init = False
        self.closed= False
        self.object_throw = False#物体是否已经被“抛出”（throw）。
        self.steps = 0#当前 episode 已经走了多少个环境 step
        # Reset the time
        self.start_time = self.Dcmm.data.time#由于_reset_simulation()已经全部reset了，所以这个数值就是0
        self.catch_time = self.Dcmm.data.time - self.start_time#这也不是为了真的算抓住的时间，而是在清零     

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
        # === 【修改点 1：计算指尖初始距离】 ===
        obj_pos_now = self.Dcmm.data.body(self.Dcmm.object_name).xpos
        # 计算指尖到物体的欧式距离
        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("arm_seg6").xpos - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),#ee离arm_base的距离
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),#arm_base距离物体的距离
            "base_distance_copy": np.linalg.norm(self.Dcmm.data.body("arm_base_copy").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),
            "evn_time": self.Dcmm.data.time - self.start_time,
            "qpos_sum": self.Dcmm.data.joint("gripper2_axis").qpos[0]+self.Dcmm.data.joint("gripper1_axis").qpos[0],
            "plate_distance":np.linalg.norm(self.Dcmm.data.site("frame_bottom_center").xpos - self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),
            "plate_z_distance" : np.abs(self.Dcmm.data.site("frame_bottom_center").xpos[2] - 
                      self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]),
            "ee_position":self.Dcmm.data.body("arm_seg6").xpos,
            "ee_position_copy":self.Dcmm.data.body("arm_seg6_copy").xpos,
        }
        # Get the observation and info
        
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]#self.initial_ee_pos3d = self._get_relative_ee_pos3d()ee相对于arm_base的相对位置
        self.prev_ee_pos3d_copy[:] = self.initial_ee_pos3d_copy[:]
        self.prev_obj_pos3d = self._get_relative_object_pos3d()#物体基于arm_base的坐标
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


    def compute_reward(self, obs, info, ctrl):
        '''
        Rewards:
        - Object Position Reward
        - Object Orientation Reward
        - Object Touch Success Reward
        - Object Catch Stability Reward
        - Collision Penalty
        - Constraint Penalty
        '''
        #盘子中心到物体的距离奖励
        rewards = 0.0

        reward_plate_distance =  (self.info["plate_distance"] - info["plate_distance"]) * DcmmCfg.reward_weights["r_plate_pos"]
        # elif 0.1 <= info["plate_distance"] <= 0.15:
        #     reward_plate_distance =  (self.info["plate_distance"] - info["plate_distance"]) * DcmmCfg.reward_weights["r_base_pos"]*1.7
        # r_weight = DcmmCfg.reward_weights["r_plate_pos"]
        # # --- 1. 计算三个区间的位移贡献 ---
        # # 区间 1: [0.15, inf) -> 斜率 1.0
        # # 计算方法：起始点和结束点都在 0.15 以上的部分有多少
        # pre = float(np.ravel(self.info["plate_distance"])[0])
        # now = float(np.ravel(info["plate_distance"])[0])
        # d_start_1 = max(pre, 0.15)
        # d_end_1 = max(now, 0.15)
        # dist_1 = d_start_1 - d_end_1
        # # 区间 2: [0.1, 0.15] -> 斜率 2.0
        # # 计算方法：将位移限制在 0.1 到 0.15 之间
        # d_start_2 = max(min(pre, 0.15), 0.1)
        # d_end_2 = max(min(now, 0.15), 0.1)
        # dist_2 = d_start_2 - d_end_2
        # # 区间 3: [0, 0.1] -> 斜率 0.5
        # # 计算方法：位移在 0.1 以下的部分
        # d_start_3 = min(pre, 0.1)
        # d_end_3 = min(now, 0.1)
        # dist_3 = d_start_3 - d_end_3
        # # --- 2. 汇总最终奖励 ---
        # # 每一段位移单独乘以它所在区间的斜率系数
        # reward_plate_distance = r_weight * (dist_1 * 1.0 + dist_2 * 2.0 + dist_3 * 0.5)

        reward_ee_precision = math.exp(-50*info["plate_distance"]**2) * DcmmCfg.reward_weights["r_precision"]
        #两个底盘的速度奖励
        # speed1 = self.Dcmm.data.body("base_link").cvel[3:5]
        # speed1_x = speed1[0]
        # speed1_y = speed1[1]
        # speed2 = self.Dcmm.data.body("base_link_copy").cvel[3:5]
        # speed2_x = speed2[0]
        # speed2_y = speed2[1]
        # reward_speed = - abs(speed1_x - speed2_x)*2 - abs(speed1_y - speed2_y)*2
        #reward_speed = -np.linalg.norm(speed1 - speed2)*2
        #两个底盘中心点xy坐标到物体xy坐标的距离
        # p1 = self.Dcmm.data.body('base_link').xpos[:2] 
        # p2 = self.Dcmm.data.body('base_link_copy').xpos[:2]
        # mid_point = (p1 + p2) / 2
        # obj_xy = self.Dcmm.data.body('object').xpos[:2]
        # dist_xy = np.linalg.norm(mid_point - obj_xy)
        # reward_mid = np.exp(-0.5 * dist_xy)
        #两个棒距离约束
        bar1 = self.Dcmm.data.body('bar_left').xpos[:2]
        bar2 = self.Dcmm.data.body('bar_right').xpos[:2]
        base_link_dist = np.linalg.norm(bar1-bar2)
        dist_min, dist_max = 0.28, 0.33
        dist_error = base_link_dist - np.clip(base_link_dist, dist_min, dist_max)
        reward_base_link_dist = -2 * np.abs(dist_error)
        #
        pre_height_dist = abs(self.info["ee_position"][2] - self.info["ee_position_copy"][2])
        now_height_dist = abs(info["ee_position"][2] - info["ee_position_copy"][2])
        reward_height_improvement = (pre_height_dist - now_height_dist) * 10.0
        reward_height_improvement = np.clip(reward_height_improvement, -5.0, 5.0)
        
        reward_base_pos_dist = (self.info["base_distance"] - info["base_distance"]) * DcmmCfg.reward_weights["r_base_pos"]
        reward_base_pos_copy = (self.info["base_distance_copy"] - info["base_distance_copy"]) * DcmmCfg.reward_weights["r_base_pos"]
        reward_base_pose = reward_base_pos_dist + reward_base_pos_copy
        # ee1_height = self.Dcmm.data.body("arm_seg6").xpos[2]
        # ee2_height = self.Dcmm.data.body("arm_seg6_copy").xpos[2]
        # height_dist = abs(ee1_height - ee2_height)
        # reward_height = math.exp(-100 * height_dist**2) * 4
        
        reward_z_diatance = 0
        if info["plate_distance"]<0.3:
            reward_z_diatance = (self.info["plate_z_distance"]-info["plate_z_distance"])* DcmmCfg.reward_weights["r_base_pos"]

        elif self.task == 'Tracking':
            ## Ctrl Penalty
            # Compute the norm of base and arm movement through the current actions in the grasping stage
            reward_ctrl = - self.norm_ctrl(ctrl, {"base","base_copy","arm","arm_copy","hand","hand_copy"})*0.7

            rewards = reward_ctrl +reward_plate_distance  + reward_base_link_dist +  reward_z_diatance + reward_ee_precision +reward_height_improvement +reward_base_pose
            # print(f"1. 控制惩罚 (reward_ctrl):             {reward_ctrl:.6f}")
            # print(f"2. 盘子距离奖励 (reward_plate_dist):    {reward_plate_distance:.6f}")
            # print(f"3. 底盘速度对齐 (reward_speed):         {reward_speed:.6f}")
            # print(f"4. 两棒距离约束 (reward_base_link_dist):     {reward_base_link_dist:.6f}")
            # print(f"5. Z轴距离奖励 (reward_z_diatance):     {reward_z_diatance:.6f}")
            # print(f"6. 末端精度奖励 指数奖励(reward_ee_precision):  {reward_ee_precision:.6f}")
            # print(f"7. 末端高度一致奖励 (reward_height_improve): {reward_height_improvement:.6f}")
            # print("-" * 30)
            # print(f"*** 总奖励 (TOTAL REWARDS):           {rewards:.6f} ***")
            # print("="*30 + "\n")
        return rewards

    def _step_mujoco_simulation(self, action_dict):
            # print(f"--- 断点 1 (初始状态) ---")
            # print(f"target_arm_qpos: {self.Dcmm.target_arm_qpos}")
            self.Dcmm.target_base_vel[0:2] = action_dict['base'][0:2] #<-- 这行旧代码被删除了
            self.Dcmm.target_base_vel_copy[0:2] = action_dict['base_copy'][0:2]
            action_arm = np.concatenate((action_dict["arm"], np.zeros(3)))#ik求解器需要七个数，后面三个元素补0表示不需要改变后面三个元素
            action_arm_copy = np.concatenate((action_dict["arm_copy"], np.zeros(3)))
            result_QP, _ = self.Dcmm.move_ee_pose(action_arm)#return result_QP:6个关节角度
            # print(f"--- 断点 2 (IK 计算后) ---")
            # print(f"Action Arm (输入): {action_arm}") # 确认一下是不是全 0
            # print(f"IK Success: {result_QP[1]}")      # 看看第一帧 IK 到底成没成功
            # print(f"IK Result QP[0]: {result_QP[0]}") # 看看解出来的角度是不是正常的
            result_QP_copy, _ = self.Dcmm.move_ee_pose_copy(action_arm_copy)
            if np.isnan(result_QP[0]).any():
                print("!!! 警告: 主臂 IK 求解器返回了 NaN !!!")
            if np.isnan(result_QP_copy[0]).any():
                print("!!! 警告: 从臂 IK 求解器返回了 NaN !!!")
            if result_QP[1]:#表示 IK 是否求解成功，或者该解是否满足关节限制。，一个布尔标志
                self.arm_limit = True
                self.Dcmm.target_arm_qpos[:] = result_QP[0]
            else:
                #print("IK Failed!!!")
                self.arm_limit = False
            if result_QP_copy[1]:#表示 IK 是否求解成功，或者该解是否满足关节限制。，一个布尔标志
                self.arm_limit = True
                self.Dcmm.target_arm_qpos_copy[:] = result_QP_copy[0]
            else:
                #print("IK Failed!!!")
                self.arm_limit = False
            # if self.info["gripper_dist"] < 0.20:#当夹爪距离物体小于20cm时才执行手部动作
            #     # 使用循环直接修改
            #     for key in action_dict["hand"]:
            #         action_dict["hand"][key] = -2
            self.Dcmm.action_hand2qpos(action_dict["hand"])#12维
            self.Dcmm.action_hand2qpos_copy(action_dict["hand_copy"])
            # Add Target Action to the Buffer
            self.update_target_ctrl()
            # Reset the Criteria for Successfully Touch
            self.step_touch = False#手是否与物体接触
            for _ in range(self.steps_per_policy):#重复执行循环体 self.steps_per_policy 次。每个动作持续多少仿真步
                # Update the control command according to the latest policy output
                self.Dcmm.data.ctrl[:-1] = self._get_ctrl()#最后一维就是object，最后写一个要单独来控制重力，不是全局的重力
                # --- 【插入位置】：在这里检查真凶 ---
                # target_idx = 16
                # # 确保索引不越界
                # if target_idx < len(self.Dcmm.data.ctrl):
                #     val = self.Dcmm.data.ctrl[target_idx]
                #     if np.isnan(val) or np.isinf(val) or abs(val) > 1e6:
                #         print(f"!!! 捕获真凶: Actuator {target_idx} 接收到的值是: {val}")
                #         # 如果你想看看到底是哪一步崩的，可以把整行 ctrl 打印出来
                #         print(f"Full CTRL: {self.Dcmm.data.ctrl}")
                # -----------------------------------
                if self.render_per_step:
                    # Rendering
                    img = self.render()#render函数返回的是虚拟世界中摄像头所看到的东西。则在每个仿真子步调用 self.render() 生成一帧图像（并可能显示或保存）。
                # ================== 【保持不变：轨迹逻辑开始】 ==================
                # 1. 计算物体已经运动的总时间 (当前时间 - 起始时间 - 静止等待时间)
                current_move_time = self.Dcmm.data.time - self.start_time - self.object_static_time

                # 阶段 A：物体静止期 (保持原样)
                if self.Dcmm.data.time - self.start_time < self.object_static_time:
                    self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                                velocity=np.zeros(6))
                    self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]

                # 阶段 B：物体运动期 (这里进行了大幅修改和增加)
                else:
                    # 无论直线还是曲线，都始终施加力抵消重力
                    self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
                    
                    # --- 情况 1: 直线运动 (原逻辑封装) ---
                    if self.trajectory_type == 'linear':
                        if not self.object_throw: # 只有第一下给初速度
                            self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
                                                        velocity=self.object_vel6d[:])
                            self.object_throw = True
                        # 之后靠物理引擎惯性飞行，不需要在这里写代码
                    
                    # --- 情况 2: 曲线运动 (完全新增) ---
                    elif self.trajectory_type == 'curve':
                        self.object_throw = True # 标记为已抛出
                        
                        # 基础线性位置 = 起点 + 速度 * 时间
                        target_pos = self.object_pos3d + self.object_vel6d[:3] * current_move_time#前两个参数是物体被抛出时的初始速度和位置，current_move_time是物体的累积运动时间
                        target_vel = self.object_vel6d[:3].copy()#目标速度向量。先复制初始的线性速度，为后面叠加波动速度做准备

                        # 计算正弦偏移量 (Position) 和 速度偏移量 (Velocity，即位置的导数)
                        sine_offset = self.curve_amp * math.sin(self.curve_freq * current_move_time + self.curve_phase)#计算正弦偏移
                        sine_vel_offset = self.curve_amp * self.curve_freq * math.cos(self.curve_freq * current_move_time + self.curve_phase)#计算速度，公式是对上面的位移进行求导

                        # 根据初始化时选定的轴 (Y 或 Z) 叠加偏移
                        if self.curve_axis == 'y':
                            target_pos[1] += sine_offset
                            target_vel[1] += sine_vel_offset
                        elif self.curve_axis == 'z':
                            target_pos[2] += sine_offset
                            target_vel[2] += sine_vel_offset
                        
                        # 每一帧都强行修正物体的位置和速度，实现曲线效果
                        self.Dcmm.set_throw_pos_vel(
                            pose=np.concatenate((target_pos, self.object_q)), 
                            velocity=np.concatenate((target_vel, [0,0,0]))
                        )
                    # 在 DcmmVecEnv.py 的 _step_mujoco_simulation 函数中
                    # 在 DcmmVecEnv.py 的 _step_mujoco_simulation 函数中
                    elif self.trajectory_type == 'circle':
                        self.object_throw = True
                        
                        # ================== 【关键修改 1：获取实时底座位置】 ==================
                        # 不再使用初始化的固定中心，而是实时获取底座在世界坐标系下的 X, Y 坐标
                        # "base_link" 是你在 test.xml 中定义的底座名称
                        current_base_pos = self.Dcmm.data.body("base_link").xpos[0:2] 
                        # ===================================================================

                        # 1. 计算当前角度: θ = θ0 + ω * t
                        current_angle = self.circle_start_angle + self.circle_omega * current_move_time
                        
                        # 2. 计算目标位置 (基于实时底座位置作为圆心)
                        # x = base_x + r * cos(θ), y = base_y + r * sin(θ)
                        target_x = current_base_pos[0] + self.circle_radius * math.cos(current_angle)
                        target_y = current_base_pos[1] + self.circle_radius * math.sin(current_angle)
                        target_pos = np.array([target_x, target_y, self.object_pos3d[2]]) # 高度保持初始随机高度
                        
                        # 3. 计算目标速度 (切向速度)
                        # 为了让物理引擎计算更准确，建议加上底座本身的移动速度 (可选)
                        base_vel = self.Dcmm.data.qvel[0:2] # 获取底座当前的线性速度 [vx, vy]
                        
                        target_vx = -self.circle_radius * self.circle_omega * math.sin(current_angle) + base_vel[0]
                        target_vy = self.circle_radius * self.circle_omega * math.cos(current_angle) + base_vel[1]
                        target_vel = np.array([target_vx, target_vy, 0.0])
                        
                        # 4. 强行修正物体状态
                        self.Dcmm.set_throw_pos_vel(
                            pose=np.concatenate((target_pos, self.object_q)), 
                            velocity=np.concatenate((target_vel, [0, 0, 0]))
                        )
                # ================== 【保持不变：轨迹逻辑结束】 ==================

                mujoco.mj_step(self.Dcmm.model, self.Dcmm.data) # 前面把速度和位置写进self.Dcmm.set_throw_pos_vel这个函数以后再执行这一步就会推进仿真一步，把速度位置写进仿真环境
                mujoco.mj_rnePostConstraint(self.Dcmm.model, self.Dcmm.data)

                # Update the contact information
                self.contacts = self._get_contacts()
                object_on_plate = self.contacts["object_on_plate"]
                any_base_collision = self.contacts["any_base_collision"]
                # (以下关于碰撞判定、terminated 逻辑的代码完全保持不变)
                if any_base_collision:
                    self.terminated = True
                    print(colored("!!! Base Collided !!!", "red")) # 检查是不是底座撞了
                if self.step_touch == False:
                    if self.task == "Tracking":
                        if object_on_plate:
                            self.step_touch = True
                    # elif self.task == "Catching":
                    #     if np.any(mask_finger1) and np.any(mask_finger2):
                    #         self.step_touch = True

                # if not self.terminated:
                #     if self.task == "Catching":
                #         self.terminated = np.any(mask_coll)
                        #print(self.terminated)
                        #print(colored("!!! Stage Error: Object escaped during grasping1 !!!", "yellow"))
                if self.task == "Tracking":
                        self.terminated = object_on_plate
                        #print(colored("!!! Stage Error: Object escaped during grasping2 !!!", "yellow"))
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

        self._step_mujoco_simulation(action)   
        # Get the obs and info
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
# --- 核心修改：NaN 拦截逻辑 ---
        if contains_nan(obs):
            #print(colored("[末端防护] 检测到 NaN，执行紧急重置...", "red"))
            # 1. 立即重置环境
            obs_reset, _ = self.reset()
            if isinstance(obs_reset, tuple): obs_reset = obs_reset[0]
            
            # 2. 构造安全的 info 字典
            info_nan = {
                "error": "NaN",
                "is_success": False,
                "ctrl": np.zeros(9) # 确保有 ctrl 键供外部调用
            }
            
            # 3. 直接提前返回，不进入下面的 compute_reward
            # 返回值顺序：obs, reward, terminated, truncated, info
            return obs_reset, -1.0, True, False, info_nan

        info = self._get_info()
        # # === 【新增保护】 ===
        # # 检查观测值里有没有 NaN，如果有，强行重置环境，防止崩到底层
        # flat_obs = np.concatenate([v.flatten() for v in obs.values() if isinstance(v, np.ndarray)])
        # if np.isnan(flat_obs).any():
        #     #print(" 检测到 NaN,环境将重置以避免崩溃。")
        #     self.reset()
        #     # 返回全 0 的观测，让网络“混”过这一步，或者返回 done=True
        #     return self._get_obs(), 0, True, False, self._get_info()
        # # ===================
        if self.task == 'Catching':
            if info['gripper_dist'] < DcmmCfg.distance_thresh and self.stage == "tracking":
                self.stage = "grasping"
            elif info['gripper_dist'] >= DcmmCfg.distance_thresh * 1.2 and self.stage == "grasping":
                self.terminated = True
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
        if self.task == "Catching":
            if info["env_time"] > self.env_time:
                #print("Catching Success!!!!!!")
                truncated = True
            else: truncated = False
        elif self.task == "Tracking":
            if self.step_touch:
                # print("Tracking Success!!!!!!")
                truncated = True
            else: truncated = False
        terminated = self.terminated
        if info["env_time"] > self.env_time:
                #print("Catching Success!!!!!!")
            truncated = False
            terminated = True
        done = terminated or truncated
        if done:
            # TEST ONLY
            # self.reset()
            pass
        return obs, reward, terminated, truncated, info

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