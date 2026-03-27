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
        self.arm1_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg1')
        self.arm2_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg2')
        self.arm3_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg3')
        self.arm4_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg4')
        self.arm5_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg5')
        self.arm6_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM,  'arm_seg6')
        #print("#############################id")
        #print(self.arm1_id)
        #print(self.arm2_id)
# ==========================================================
# --- 夹爪 ID 全能检测模块 (Debug Hand IDs) ---
# ==========================================================
# --- 改为以下精确获取 ---
        # # 1. 用于 Observation (qposadr): 对应调试输出的 21
        self.hand_qpos_addr = self.Dcmm.model.joint('gripper1_axis').qposadr[0]

        # # 2. 用于 Step 控制 (actuator): 对应调试输出的 14
        self.hand_ctrl_id = self.Dcmm.model.actuator('hand_actuator_0').id

        # # 3. 用于碰撞检测 (geom): 对应调试输出的 57 和 59
        self.f1_geom_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper1')
        self.f2_geom_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper2')

        # # 顺便获取手臂起始 ID，用于 mask_coll 判定 (假设 link1 是手臂起始)
        self.arm_start_geom_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'arm_seg1')
        # ------------------------------
        self.floor_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.object_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)
        self.base_id = mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, 'ranger_base')

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
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),#spaces.Box(low, high, shape, dtype)

                }),
                "arm": spaces.Dict({
                    "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
                    "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                    "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(9, 15)]),#（9,15)代表的是6个关节，这是给6个关节找上下限
                                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(9, 15)]),
                                            dtype=np.float32),
                }),
                "hand": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices]),
                                   high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices]),
                                   dtype=np.float32),#DcmmCfg.hand_mask这个是几维的，上面这个hand就是几维的。
                "object": spaces.Dict({
                    "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
                    "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
                    # -------- 【这里是新增的行】 --------
                    # 3 (x,y,z) * 3 (历史长度) = 9维
                    "pos_history": spaces.Box(-10, 10, shape=(3 * self.obj_history_len,), dtype=np.float32),
                    # -
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
        base_low = np.array([-4, -4, -1.0]) 
        base_high = np.array([4, 4, 1.0])
        # Define the limit for the arm action
        arm_low = -0.025*np.ones(4)
        arm_high = 0.025*np.ones(4)
        # Define the limit for the hand action
        hand_low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices])#hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15
        hand_high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices])#jnt_range是固定的表述，表示的就是第几个joint
        #传入hand活动范围的最大最小值，最大最小值是存储在.xml文件中的
        # Get initial ee_pos3d
        self.init_pos = True
        self.initial_ee_pos3d = self._get_relative_ee_pos3d()
        '''return np.array([x, y, 
                         self.Dcmm.data.body("link6").xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#末端执行器(link6)在垂直方向(Z轴)上相对于机械臂底座(arm_base)的高度差。
        x,y是末端执行器的位置'''
        self.initial_obj_pos3d = self._get_relative_object_pos3d()
        '''return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#物体在 Z 方向（垂直方向）相对于机械臂基座的高度差。
        x,y是机械臂基地的位置'''
        self.prev_ee_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_obj_pos3d = np.array([0.0, 0.0, 0.0])
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]
        self.prev_obj_pos3d[:] = self.initial_obj_pos3d[:]

        # Actions (dim = 20)
        self.action_space = spaces.Dict(
            {
                "base": spaces.Box(base_low, base_high, shape=(3,), dtype=np.float32),#定义机器人底座的控制动作。
                "arm": spaces.Box(arm_low, arm_high, shape=(4,), dtype=np.float32),#定义机器人机械臂的控制动作
                "hand": spaces.Box(low = hand_low,
                                   high = hand_high,
                                   dtype = np.float32),#这里的hand的维数是由上面的high和low确定的，他们是几维的hand就是几维的
            }
        )
        self.action_buffer = {
            "base": DynamicDelayBuffer(maxlen=2),
            "arm": DynamicDelayBuffer(maxlen=2),
            "hand": DynamicDelayBuffer(maxlen=2),#hand原来是12维
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
        self.obs_t_dim = self.obs_dim - 2 - 6  # dim = 18, 12 for the hand, 6 for the arm joint positions
        #还要-6的原因：Tracking Task（跟踪任务）不需要关节角度，只需要末端（EE）
        self.act_t_dim = self.act_dim - 2 # dim = 6, 12 for the hand
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
        gripper_center_init = self._get_gripper_center()
        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("arm_seg6").xpos - 
                                          self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),#计算手与目标物体的距离
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] - 
                                            self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),#机械臂基座和目标物体在平面上的距离，判断底盘是否需要移动
            "env_time": self.Dcmm.data.time - self.start_time,#计算环境运行了多长时间
            "imgs": {},#用来存取拍摄帧
            "gripper_dist": np.linalg.norm(gripper_center_init - self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),#计算夹爪中心和目标物体的距离，判断是否进入抓取阶段
            "qpos_sum": self.Dcmm.data.joint("gripper2_axis").qpos[0]+self.Dcmm.data.joint("gripper1_axis").qpos[0], # 机械臂关节位置的总和（用于调试）
        }
        self.contacts = {
            # Get contact point from the mujoco model
            "object_contacts": np.array([]),#物体和环境的接触点
            "hand_contacts": np.array([]),
            "left_finger_contacts": [],  # 这里的 list 供 "in" 使用
            "right_finger_contacts": []
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
        # Contact information of the hand
        geom_ids = self.Dcmm.data.contact.geom#把两个物体都收集起来，这个是二维的，会获得所有的碰撞对
        geom1_ids = self.Dcmm.data.contact.geom1#两个发生碰撞的其中一个物体，一维数组
        geom2_ids = self.Dcmm.data.contact.geom2#两个发生碰撞的其中的一个物体
        ## get the contact points of the hand，self.hand_start_id手部第一个零件的id
        #geom1_hand = np.where((geom1_ids < self.object_id) & (geom1_ids >= self.hand_start_id))[0]#第一个条件过滤掉所有比 object ID 大的 geom，第二个条件过滤掉比 hand_start_id 小的 geom
        #这个区间的 ID 全部属于“手部几何体”，然后返回返回满足条件的 index 列表，where返回的不是id，是第几个元素
        #geom2_hand = np.where((geom2_ids < self.object_id) & (geom2_ids >= self.hand_start_id))[0]
        # --- 修改为以下代码 ---
        # 我们直接检查碰撞列表里是否有我们那两个手指的 Geom ID (57 和 59)
        # --- 【核心新增：提取左指 f1 碰到的所有物体 ID】 ---
        # 找到所有 geom1 是 f1 或 geom2 是 f1 的碰撞索引
        f1_g1_idx = np.where(geom1_ids == self.f1_geom_id)[0]
        f1_g2_idx = np.where(geom2_ids == self.f1_geom_id)[0]
        # 提取对方的 ID 并转为 list
        left_finger_contacts = np.concatenate((
            geom_ids[f1_g1_idx][:, 1] if f1_g1_idx.size > 0 else np.array([]),
            geom_ids[f1_g2_idx][:, 0] if f1_g2_idx.size > 0 else np.array([])
        )).tolist()

        # --- 【核心新增：提取右指 f2 碰到的所有物体 ID】 ---
        f2_g1_idx = np.where(geom1_ids == self.f2_geom_id)[0]
        f2_g2_idx = np.where(geom2_ids == self.f2_geom_id)[0]
        # 提取对方的 ID 并转为 list
        right_finger_contacts = np.concatenate((
            geom_ids[f2_g1_idx][:, 1] if f2_g1_idx.size > 0 else np.array([]),
            geom_ids[f2_g2_idx][:, 0] if f2_g2_idx.size > 0 else np.array([])
        )).tolist()
        geom1_hand = np.where((geom1_ids == self.f1_geom_id) | (geom1_ids == self.f2_geom_id))[0]
        geom2_hand = np.where((geom2_ids == self.f1_geom_id) | (geom2_ids == self.f2_geom_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])#碰撞geom1和geom2的摆放顺序是小索引在前，大索引在后
        if geom1_hand.size != 0:
            contacts_geom1 = geom_ids[geom1_hand][:,1]
        if geom2_hand.size != 0:
            contacts_geom2 = geom_ids[geom2_hand][:,0]
        hand_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        ## get the contact points of the object
        geom1_object = np.where((geom1_ids == self.object_id))[0]
        geom2_object = np.where((geom2_ids == self.object_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_object.size != 0:
            contacts_geom1 = geom_ids[geom1_object][:,1]
        if geom2_object.size != 0:
            contacts_geom2 = geom_ids[geom2_object][:,0]
        object_contacts = np.concatenate((contacts_geom1, contacts_geom2))#把“所有物体（object）参与的碰撞事件中，对方的几何体 ID”全部集中到一个数组中。
        #geom1_hand = 哪些碰撞事件里 geom1 是手，geom2_hand = 哪些碰撞事件里 geom2 是手
        #contacts_geom1 = 在这些事件中，手碰到的“对方”对象（geom2），contacts_geom2 = 在 geom2 是手时，手碰到的“对方”对象（geom1）
        #hand_contacts = 手在这一帧碰到的所有对象的 geom id 列表
        ## get the contact points of the base
        geom1_base = np.where((geom1_ids == self.base_id))[0]#geom1_base ＝ 所有碰撞事件中 geom1 等于底座 的事件的索引
        geom2_base = np.where((geom2_ids == self.base_id))[0]#找出所有底座参与的碰撞，geom2_base ＝ 所有碰撞事件中 geom2 等于底座 的事件的索引
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_base.size != 0:
            contacts_geom1 = geom_ids[geom1_base][:,1]
        if geom2_base.size != 0:
            contacts_geom2 = geom_ids[geom2_base][:,0]
        base_contacts = np.concatenate((contacts_geom1, contacts_geom2))
        if self.print_contacts:
            print("object_contacts: ", object_contacts)
            print("hand_contacts: ", hand_contacts)
            print("base_contacts: ", base_contacts)
        return {
            # Get contact point from the mujoco model
            "object_contacts": object_contacts,#物体碰到了什么，判定球是否被抓住 / 是否掉地，两个手都碰到球而且没掉到地上
            "hand_contacts": hand_contacts,#手碰到了什么，判定手是否触碰球 / 安全边界
            "base_contacts": base_contacts,#底座碰到了什么，判定底座是否撞击障碍物
            # --- 【关键：新增以下两个键，完美对应 reward 函数的参数】 ---
            "left_finger_contacts": left_finger_contacts,  # 这里的 list 供 "in" 使用
            "right_finger_contacts": right_finger_contacts
        }

    # === 修改后的代码 ===
    def _get_base_vel(self):
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
        global_base_vel = self.Dcmm.data.qvel[0:2] # 获取全局线速度
        
        # 计算局部线速度 (vx, vy) - 保持不变
        base_vel_x = math.cos(base_yaw) * global_base_vel[0] + math.sin(base_yaw) * global_base_vel[1]
        base_vel_y = -math.sin(base_yaw) * global_base_vel[0] + math.cos(base_yaw) * global_base_vel[1]
        
        # === 【新增】获取底盘自转角速度 (Omega) ===
        # 假设你的底盘在 MuJoCo 中是前三个自由度 (x, y, yaw)，那么 qvel[2] 就是 yaw_rate
        base_omega = self.Dcmm.data.qvel[2] 

        # 返回 3 维数组 [vx, vy, omega]
        return np.array([base_vel_x, base_vel_y, base_omega])
    # def _get_base_vel(self):
    #     base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])
    #     #获取机器人底座的 yaw（朝向）世界坐标系
    #     global_base_vel = self.Dcmm.data.qvel[0:2]#读取底座的线速度（世界坐标系）直接从data中获取的数据都是相对于世界坐标系来说的
    #     base_vel_x = math.cos(base_yaw) * global_base_vel[0] + math.sin(base_yaw) * global_base_vel[1]
    #     base_vel_y = -math.sin(base_yaw) * global_base_vel[0] + math.cos(base_yaw) * global_base_vel[1]
    #     return np.array([base_vel_x, base_vel_y])#机器人前进后退速度，左右速度，这个速度是相对于机器人自身来说的

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
        #
    def _get_relative_ee_quat(self):
        # Caclulate the ee_pos3d w.r.t. the base_link
        quat = relative_quaternion(self.Dcmm.data.body("base_link").xquat, self.Dcmm.data.body("arm_seg6").xquat)
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
    
    def _get_relative_object_pos3d(self):
        # Caclulate the object_pos3d w.r.t. the base_link
        base_yaw = quat2theta(self.Dcmm.data.body("base_link").xquat[0], self.Dcmm.data.body("base_link").xquat[3])#底座在水平平面上的旋转角度。
        x,y = relative_position(self.Dcmm.data.body("arm_base").xpos[0:2], 
                                self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2], 
                                base_yaw)#物体在机械臂基座（arm_base）坐标系下的相对位置，但只针对 平面 XY 方
        return np.array([x, y, 
                         self.Dcmm.data.body(self.Dcmm.object_name).xpos[2]-self.Dcmm.data.body("arm_base").xpos[2]])#物体在 Z 方向（垂直方向）相对于机械臂基座的高度差。
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
    def _get_obs(self):
        ee_pos3d = self._get_relative_ee_pos3d()#机械臂末端基于arm_base的坐标
        obj_pos3d = self._get_relative_object_pos3d()#物体基于arm_base的坐标
        # 初始化记录 qpos
        self.prev_qpos_sum = self.Dcmm.data.joint("gripper1_axis").qpos[0] + \
                     self.Dcmm.data.joint("gripper2_axis").qpos[0]
        if self.init_pos:#初始坐标
            self.prev_ee_pos3d[:] = ee_pos3d[:]
            self.prev_obj_pos3d[:] = obj_pos3d[:]
            self.init_pos = False
        # Add Obs Noise (Additive self.k_obs_base/arm/hand/object)
        self.obj_pos_history.append(obj_pos3d.copy())
        # 在 obs 字典定义的上方获取
        current_hand_qpos = np.array(self.Dcmm.data.qpos[self.hand_qpos_addr : self.hand_qpos_addr + 2])
        obs = {
            "base": {
                "v_lin_3d": self._get_base_vel() + np.random.normal(0, self.k_obs_base, 3),#这个是相对于机器人自身的速度
                #"v_lin_2d": self.Dcmm.data.qvel[0:2],
            },
            "arm": {
                "ee_pos3d": ee_pos3d + np.random.normal(0, self.k_obs_arm, 3),#机械臂末端基于arm_base的坐标
                "ee_quat": self._get_relative_ee_quat() + np.random.normal(0, self.k_obs_arm, 4),#机械臂末端基于底座的相对四元数
                'ee_v_lin_3d': (ee_pos3d - self.prev_ee_pos3d)*self.fps + np.random.normal(0, self.k_obs_arm, 3),#delta机械臂末端相当于arm_base的速度
                "joint_pos": np.array(self.Dcmm.data.qpos[15:21]) + np.random.normal(0, self.k_obs_arm, 6),
            },
            # 修改这里，直接使用上面获取的 current_hand_qpos
            "hand": current_hand_qpos + np.random.normal(0, self.k_obs_hand, 2),
            #"hand": self._get_hand_obs() + np.random.normal(0, self.k_obs_hand, 2),
            "object": {
                "pos3d": obj_pos3d + np.random.normal(0, self.k_obs_object, 3),#物体相对于arm_base的坐标
                "v_lin_3d": self._get_relative_object_v_lin_3d() + np.random.normal(0, self.k_obs_object, 3),
                #"v_lin_3d": (obj_pos3d - self.prev_obj_pos3d)*self.fps + np.random.normal(0, self.k_obs_object, 3),速度计算
                "pos_history": np.array(self.obj_pos_history).flatten() + np.random.normal(0, self.k_obs_object, 3 * self.obj_history_len),
            },
        }
        self.prev_ee_pos3d = ee_pos3d
        self.prev_obj_pos3d = obj_pos3d
        if self.print_obs:
            print("##### print obs: \n", obs)
        # 修改后：
        hand_pos = self.Dcmm.data.qpos[self.hand_qpos_addr : self.hand_qpos_addr + 2]
        # print(f"\n[DEBUG 1 - OBS] Raw Hand Qpos: {hand_pos}")
        # if len(hand_pos) == 0:
        #     print(colored("!!! WARNING: Hand Qpos is EMPTY. Check hand_start_id!", "red"))
        return obs#obs是一个字典
        # return obs_tensor这个函数收集机器人当前状态（位置、速度、姿态、手的状态、物体状态等），并加上随机噪声，返回一个完整的观测字典 obs。

    def _get_hand_obs(self):
        # print("full hand: ", self.Dcmm.data.qpos[21:37])
        hand_obs = np.zeros(2)
        hand_obs[0] = self.Dcmm.data.qpos[21]

        # gripper2 的关节位置（slide)
        hand_obs[1] = self.Dcmm.data.qpos[22]
        '''
        # Thumb
        hand_obs[9] = self.Dcmm.data.qpos[21+13]
        hand_obs[10] = self.Dcmm.data.qpos[21+14]
        hand_obs[11] = self.Dcmm.data.qpos[21+15]
        # Other Three Fingers
        hand_obs[0] = self.Dcmm.data.qpos[21]
        hand_obs[1:3] = self.Dcmm.data.qpos[(21+2):(21+4)]
        hand_obs[3] = self.Dcmm.data.qpos[21+4]
        hand_obs[4:6] = self.Dcmm.data.qpos[(21+6):(21+8)]
        hand_obs[6] = self.Dcmm.data.qpos[21+8]
        hand_obs[7:9] = self.Dcmm.data.qpos[(21+10):(21+12)]
        '''
        return hand_obs
    def _get_gripper_center(self):
        # 获取两个指尖 site 的世界坐标
        p1 = self.Dcmm.data.site("gripper1_tip").xpos
        p2 = self.Dcmm.data.site("gripper2_tip").xpos
        # 计算中点
        return (p1 + p2) / 2.0
    def _get_info(self):
        # Time of the Mujoco environment
        env_time = self.Dcmm.data.time - self.start_time
        ee_distance = np.linalg.norm(self.Dcmm.data.body("arm_seg6").xpos - 
                                    self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        base_distance = np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                        self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2])
        # print("base_distance: ", base_distance)
        gripper_dist = np.linalg.norm(self._get_gripper_center() - self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3])
        qpos1 = self.Dcmm.data.joint("gripper1_axis").qpos[0]
        qpos2 = self.Dcmm.data.joint("gripper2_axis").qpos[0]
        qpos_sum = qpos1 + qpos2
        if self.print_info: 
            print("##### print info")
            print("env_time: ", env_time)
            print("ee_distance: ", ee_distance)
        return {
            # Get contact point from the mujoco model
            "env_time": env_time,
            "ee_distance": ee_distance,
            "base_distance": base_distance,
            "gripper_dist": gripper_dist,
            "qpos_sum": qpos_sum,
        }
    
    def update_target_ctrl(self):#把当前这一时刻的目标控制量（底盘速度、机械臂关节位置、夹爪关节位置）存进一个“动作缓冲区”
        self.action_buffer["base"].append(copy.deepcopy(self.Dcmm.target_base_vel[:]))
        self.action_buffer["arm"].append(copy.deepcopy(self.Dcmm.target_arm_qpos[:]))
        self.action_buffer["hand"].append(copy.deepcopy(self.Dcmm.target_hand_qpos[:]))

    def _get_ctrl(self):#把强化学习输出的 action 信号 → 转换成底盘、机械臂、夹爪的真实控制量（ctrl）
        #作用：把缓冲区里的目标量（速度 / 关节角）转换成 MuJoCo 中 data.ctrl 的 16 维控制向量。
        # Map the action to the control 
        mv_steer, mv_drive = self.Dcmm.move_base_vel(self.action_buffer["base"][0]) # 8 mv_steer = [steer_fl, steer_fr, steer_rl, steer_rr]
        mv_arm = self.Dcmm.arm_pid.update(self.action_buffer["arm"][0], self.Dcmm.data.qpos[15:21], self.Dcmm.data.time) # 6
        #mv_hand = self.Dcmm.hand_pid.update(self.action_buffer["hand"][0], self.Dcmm.data.qpos[21:23], self.Dcmm.data.time) # 16
        mv_hand = self.action_buffer["hand"][0] #
        ctrl = np.concatenate([mv_steer, mv_drive, mv_arm, mv_hand], axis=0)#得到的是控制的力，但是从神经网络里面输出的是位移和角度
        # Add Action Noise (Scale with self.k_act)
        ctrl[:14] *= np.random.normal(1, self.k_act, 14)
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
                    # 1. 安全起见，先移除可能存在的四元数定义，防止加载冲突
                    object_body.attrib.pop("quat", None)
                    object_body.attrib.pop("axisangle", None)

                    # 2. 计算角度：
                    # 如果 Y 轴指向前方，绕 Y 轴旋转 90 度会使 Z 轴向右倒下
                    # Euler 设定通常为 [roll, pitch, yaw] 对应 [x, y, z]
                    roll = 0
                    pitch = 90  # 绕 Y 轴旋转 90 度
                    yaw = 0 # 让躺下的物体在地面上随机指向不同方向
                    
                    euler_str = f"{roll} {pitch} {yaw}"
                    object_body.set("euler", euler_str)
                    # print("### Object Geom Info ###")
                    # for key, value in geom.attrib.items():
                    #     print(f"{key}: {value}")
                else:
                    object_shape = DcmmCfg.object_shape[object_id]
                    geom.set("type", object_shape)  # Replace "box" with the desired type
                    object_size = np.array([np.random.uniform(low=low, high=high) for low, high in DcmmCfg.object_size[object_shape]])
                    geom.set("size", np.array_str(object_size)[1:-1])  # Replace with the desired size
                        # 1. 安全起见，先移除可能存在的四元数定义，防止加载冲突
                    # 替换原来的 geom.set("size", ...)
                    size_str = " ".join([f"{x:.4f}" for x in object_size])
                    geom.set("size", size_str)
                    object_body.attrib.pop("quat", None)
                    object_body.attrib.pop("axisangle", None)

                        # 2. 计算角度：
                        # 如果 Y 轴指向前方，绕 Y 轴旋转 90 度会使 Z 轴向右倒下
                        # Euler 设定通常为 [roll, pitch, yaw] 对应 [x, y, z]
                    roll = 0
                    pitch = 90  # 绕 Y 轴旋转 90 度
                    yaw = 0 # 让躺下的物体在地面上随机指向不同方向
                        
                    euler_str = f"{roll} {pitch} {yaw}"
                    object_body.set("euler", euler_str)
                    # object_mesh = DcmmCfg.object_mesh[object_id]
                    # geom.set("mesh", object_mesh)
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
        x = 0.6*np.random.rand() - 0.3
        y = 1.4 + 0.5 * np.random.rand() # (2.2, 2.5)底座面朝方向的前面
        # Low or High Starting Position
        low_factor = False if np.random.rand() < 0.5 else True
        # low_factor = True
        if low_factor: height = np.random.uniform(0.6,0.8 )#0.7 + 0.3 * np.random.rand()# (0.7, 1.0)low_facor为true时，从低高度里面选高度，各有50%的可能
        else: height = np.random.uniform(0.6, 0.8)#0.8 + 0.4 * np.random.rand() # (1.0, 1.6)
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
        # Clear Buffer
        self.action_buffer["base"].clear()
        self.action_buffer["arm"].clear()
        self.action_buffer["hand"].clear()

    def _reset_simulation(self):#重置并随机化整个仿真
        # Reset the data in Mujoco Simulation
        mujoco.mj_resetData(self.Dcmm.model, self.Dcmm.data)
        mujoco.mj_resetData(self.Dcmm.model_arm, self.Dcmm.data_arm)#把data全部重置为 默认初始状态，上面也是，下面的是单独的机械臂模型
        if self.Dcmm.model.na == 0:
            self.Dcmm.data.act[:] = None
        if self.Dcmm.model_arm.na == 0:
            self.Dcmm.data_arm.act[:] = None
        self.Dcmm.data.ctrl = np.zeros(self.Dcmm.model.nu)
        self.Dcmm.data_arm.ctrl = np.zeros(self.Dcmm.model_arm.nu)#将所有控制都清零
        self.Dcmm.data.qpos[15:21] = DcmmCfg.arm_joints[:]
        self.Dcmm.data.qpos[21:23] = DcmmCfg.hand_joints[:]#手被重置
        self.Dcmm.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]#把关节恢复到默认初始状态
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
        ## Reset the target joint positions of the arm
        self.Dcmm.target_arm_qpos[:] = DcmmCfg.arm_joints[:]
        ## Reset the target joint positions of the hand
        self.Dcmm.target_hand_qpos[:] = DcmmCfg.hand_joints[:]
        ## Reset the reward
        self.stage = "tracking"
        self.terminated = False
        self.reward_touch = 0
        self.reward_stability = 0
        # === 【修改点 1：计算指尖初始距离】 ===
        gripper_center_init = self._get_gripper_center()
        obj_pos_now = self.Dcmm.data.body(self.Dcmm.object_name).xpos
        # 计算指尖到物体的欧式距离
        init_gripper_dist = np.linalg.norm(gripper_center_init - obj_pos_now)
        self.info = {
            "ee_distance": np.linalg.norm(self.Dcmm.data.body("arm_seg6").xpos - 
                                       self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:3]),#ee离arm_base的距离
            "base_distance": np.linalg.norm(self.Dcmm.data.body("arm_base").xpos[0:2] -
                                             self.Dcmm.data.body(self.Dcmm.object_name).xpos[0:2]),#arm_base距离物体的距离
            "evn_time": self.Dcmm.data.time - self.start_time,
            "gripper_dist": init_gripper_dist,#指尖到物体的距离
            "qpos_sum": self.Dcmm.data.joint("gripper2_axis").qpos[0]+self.Dcmm.data.joint("gripper1_axis").qpos[0],
        }
        # Get the observation and info
        
        self.prev_ee_pos3d[:] = self.initial_ee_pos3d[:]#self.initial_ee_pos3d = self._get_relative_ee_pos3d()ee相对于arm_base的相对位置
        self.prev_obj_pos3d = self._get_relative_object_pos3d()#物体基于arm_base的坐标
        self.obj_pos_history.clear()
        for _ in range(self.obj_history_len):
            self.obj_pos_history.append(self.prev_obj_pos3d.copy())
        observation = self._get_obs()
        info = self._get_info()#每次调用前，前面就会用self.info把上一时间步的info存储起来，info就成了当前时间步的info
        # Rendering
        imgs = self.render() if self.render_mode is not None else None
        info['imgs'] = imgs
        #self.init_ee_distance = info["ee_distance"]
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))
        # 在 return observation, info 之前插入
        # print(f"--- RESET CHECK ---")
        # print(f"Gripper1 Qpos: {self.Dcmm.data.joint('gripper1_axis').qpos[0]}")
        # print(f"Gripper2 Qpos: {self.Dcmm.data.joint('gripper2_axis').qpos[0]}")
        # print(f"Hand Target: {self.Dcmm.target_hand_qpos}")
        return observation, info

    def norm_ctrl(self, ctrl, components):
        '''
        Convert the ctrl (dict type) to the numpy array and return its norm value
        Input: ctrl, dict
        Return: norm, float
        '''
        ctrl_array = np.concatenate([ctrl[component]*DcmmCfg.reward_weights['r_ctrl'][component] for component in components])
        return np.linalg.norm(ctrl_array)
    # def compute_arm_alignment_reward(self):
    #     """
    #     优化版：对齐奖励函数
    #     目标30-40cm 处基本对齐，进入 30cm 后奖励平缓，防止猛烈抬头撞击。
    #     """
    #     # 1. 获取物体的世界坐标
    #     obj_pos = self.Dcmm.data.body(self.object_name).xpos

    #     # 2. 获取 arm_seg6 的位置和旋转矩阵
    #     seg6_pos = self.Dcmm.data.body("arm_seg6").xpos
    #     seg6_xmat = self.Dcmm.data.body("arm_seg6").xmat.reshape(3, 3)

    #     # 3. 提取 arm_seg6 的 Z 轴向量（朝前的瞄准方向）
    #     seg6_forward_dir = seg6_xmat[:, 2] 

    #     # 4. 计算物体相对于 arm_seg6 的单位方向向量和距离
    #     relative_pos = obj_pos - seg6_pos
    #     dist = np.linalg.norm(relative_pos)
    #     target_dir = relative_pos / (dist + 1e-6)

    #     # 5. 计算基础对齐得分 (0 到 1 之间)
    #     dot_product = np.dot(seg6_forward_dir, target_dir)
    #     # 使用平方让精度要求更高
    #     alignment_score = np.power(np.maximum(0, dot_product), 2)

    #     # 6. --- 核心改进：饱和门控 (Gate) ---
    #     # 我们希望奖励在 0.5m 处激活，在 0.35m 处就接近饱和（达到 90% 以上）
    #     # 这样机械臂在 30-40cm 处就会因为“利益最大化”而提前对齐
    #     # 且进入 30cm 后，gate 的值几乎不再变化（保持在 0.99 左右），消除“暴增”梯度
    #     gate = 1 / (1 + np.exp(20 * (dist - 0.45))) 

    #     # 7. --- 核心改进：精修微调项 (Precision Bonus) ---
    #     # 为了防止 30cm 以内奖励完全静止导致对不准，添加一个极平缓的线性增量
    #     # 该项在 dist=0.3m 时约 0.35分，在 dist=0m 时为 0.5分，梯度极小，不会引起剧烈动作
    #     precision_bonus = np.clip((1.0 - dist) * 0.5, 0, 0.5)

    #     # 8. 组合最终奖励
    #     # 主奖励权重设为 5.0，精修项作为补充
    #     main_reward = alignment_score * gate * 5.0
    #     fine_tune_reward = alignment_score * precision_bonus 
        
    #     ori_reward = main_reward + fine_tune_reward

    #     # (可选) 调试打印：如果觉得不动或者动太猛，可以取消注释查看数值
    #     # if self.print_reward:
    #     #     print(f"Dist: {dist:.3f}, Alignment: {alignment_score:.3f}, Reward: {ori_reward:.3f}")

    #     return ori_reward


    def compute_arm_alignment_reward(self):
        """
        终极优化版：解决近距离对齐抖动与临门一脚停滞问题
        1. 引入 soft_dist 防止近距离方向跳变
        2. 引入指数级吸引力奖励，解决 10cm 内的动力不足
        3. 强化对齐权重，确保夹爪精准指向物体
        """
        # 1. 获取物体的世界坐标
        obj_pos = self.Dcmm.data.body(self.object_name).xpos

        # 2. 获取 arm_seg6 的位置和旋转矩阵
        seg6_pos = self.Dcmm.data.body("arm_seg6").xpos
        seg6_xmat = self.Dcmm.data.body("arm_seg6").xmat.reshape(3, 3)

        # 3. 提取 arm_seg6 的 Z 轴向量（朝前的瞄准方向）
        seg6_forward_dir = seg6_xmat[:, 2] 

        # 4. 计算相对位置和真实距离
        relative_pos = obj_pos - seg6_pos
        dist = np.linalg.norm(relative_pos)

        # 5. --- 核心改进：平滑方向锁定 (解决数值奇点) ---
        # 引入 0.05m 的软化因子，即使 dist 趋近 0，target_dir 也会保持稳定
        # 这能防止机械臂因为 1mm 的微小晃动导致奖励从 5 跌到 0
        soft_dist = dist + 0.05
        target_dir = relative_pos / soft_dist#归一化向量

        # 6. 计算基础对齐得分
        # 去掉 power(..., 2)，使用线性 dot 增加近距离的“纠偏压力”
        dot_product = np.dot(seg6_forward_dir, target_dir)
        alignment_score = np.maximum(0, dot_product)

        # 7. --- 核心改进：饱和门控 (Gate) ---
        # 保持你原有的逻辑：在 0.35m-0.45m 区域激活主奖励
        gate = 1 / (1 + np.exp(20 * (dist - 0.45))) 

        # 8. --- 核心改进：强力精度吸力 (Exponential Attraction) ---
        # 代替原本平缓的线性 precision_bonus。
        # 当 dist=0.1m 时，约为 0.67；当 dist=0m 时，飙升至 3.0。
        # 这 2.3 分的级差会产生巨大的引力，强迫机械臂走完最后 5cm。
        precision_attraction = 5.0 * np.exp(-15.0 * dist)

        # 9. 组合最终奖励
        # 主奖励：提供远距离导航，权重 5.0
        main_reward = alignment_score * gate * 5.0
        
        # 精修项奖励：只有当对齐程度较好时(>0.7)才激活，防止侧着身子瞎撞
        # 如果你想要求精度更高，可以把 0.7 改成 0.85
        is_aligned = (alignment_score > 0.7)
        fine_tune_reward = is_aligned * precision_attraction 
        
        ori_reward = main_reward + fine_tune_reward

        # 调试打印
        # if self.print_reward:
        #    print(f"Dist: {dist:.3f}, Align: {alignment_score:.2f}, Main: {main_reward:.2f}, Fine: {fine_tune_reward:.2f}")

        return ori_reward

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
# ==================== 【修改标记 1：统一输入格式 & 调试增强版】 ====================
        # 无论 ctrl 是字典还是数组，我们都统一生成两个版本：
        # ctrl_dict: 用于 norm_ctrl 函数访问，ctrl_array: 用于后续的切片操作 [6:8]
        
        ctrl_array = None
        ctrl_dict = None

        if isinstance(ctrl, (dict, OrderedDict)):
            ctrl_dict = ctrl
            try:
                # 拼接顺序：arm, base, hand (根据你 OrderedDict 观察到的顺序)
                ctrl_array = np.concatenate([ctrl['arm'], ctrl['base'], ctrl['hand']])
                
                # 调试打印：如果拼接出的维度不对，立即报告
                if len(ctrl_array) != 9:
                    print(colored(f"[Type Warning] 字典拼接后的维度为 {len(ctrl_array)}，预期为 9！内容: {ctrl.keys()}", "yellow"))
            except KeyError as e:
                print(colored(f"[Type Error] 字典缺少必要的 Key: {e}，请检查 PPO 输出格式", "red"))
                ctrl_array = np.zeros(9) # 保底方案
            except Exception as e:
                print(colored(f"[Type Error] 字典转换异常: {e}", "red"))
                ctrl_array = np.zeros(9)
                
        elif ctrl is None:
            # 这种情况通常发生在环境重置的瞬间
            print(colored("[Type Info] 收到空的 ctrl (None)，已自动初始化为全 0 向量", "cyan"))
            ctrl_dict = {'arm': np.zeros(3), 'base': np.zeros(3), 'hand': np.zeros(3)}
            ctrl_array = np.zeros(9)
            
        elif isinstance(ctrl, np.ndarray):
            ctrl_array = ctrl
            # 调试打印：如果传入的是数组，但长度不是 9
            if len(ctrl_array) != 9:
                print(colored(f"[Type Warning] 收到数组维度为 {len(ctrl_array)}，预期为 9！内容: {ctrl_array}", "yellow"))
            
            # 将数组安全地拆分回字典，供 norm_ctrl 使用
            try:
                ctrl_dict = {
                    'arm': ctrl_array[0:3],
                    'base': ctrl_array[3:6],
                    'hand': ctrl_array[6:9]
                }
            except Exception as e:
                print(colored(f"[Type Error] 数组拆分为字典失败: {e}", "red"))
                ctrl_dict = {'arm': np.zeros(3), 'base': np.zeros(3), 'hand': np.zeros(3)}
        else:
            # 捕获意料之外的数据类型
            print(colored(f"[Type Critical] 收到未知的 ctrl 类型: {type(ctrl)}，请立即检查环境配置！", "red", attrs=['bold']))
            ctrl_array = np.zeros(9)
            ctrl_dict = {'arm': np.zeros(3), 'base': np.zeros(3), 'hand': np.zeros(3)}
        # =====================================
        rewards = 0.0
        #####################################catch新增#############################################
        #世界坐标系（假设小车是朝前）x轴朝右，y轴朝前，z轴朝上
        tip1_pos = self.Dcmm.data.site("gripper1_tip").xpos
        tip2_pos = self.Dcmm.data.site("gripper2_tip").xpos
        current_gripper_width = np.linalg.norm(tip1_pos - tip2_pos)
        prev_gripper_width = getattr(self, 'prev_width', current_gripper_width)
        self.prev_width = current_gripper_width
        gripper_center = (tip1_pos + tip2_pos) / 2.0
        obj_pos_internal = self.Dcmm.data.body(self.object_name).xpos
        #print(obj_pos_internal)
        # 计算物体到指缝中心的精确距离
        dist_to_obj = np.linalg.norm(obj_pos_internal - gripper_center)
        # 【新增】专门计算水平面(XY)的偏移，这决定了物体是否在两指正中间
        dist_xy = np.linalg.norm(obj_pos_internal[:2] - gripper_center[:2])#obj_pos_internal这个就是obj的pos
        error_x = abs(obj_pos_internal[0] - gripper_center[0])
        error_y = abs(obj_pos_internal[1] - gripper_center[1])
        error_z = abs(obj_pos_internal[2] - gripper_center[2])
        # 判定左右指碰撞
        is_touching_l = self.object_id in self.contacts.get("left_finger_contacts", [])
        is_touching_r = self.object_id in self.contacts.get("right_finger_contacts", [])
        ####################################################################################
        reward_distance_away = 0.0
        if info["gripper_dist"] > self.info["gripper_dist"]:
            reward_distance_away = - DcmmCfg.reward_weights["r_away"]#0.2
        ## Object Position Reward (-inf, 0)
        # Compute the closest distance the end-effector comes to the object，self表示上一帧，是在step里面把当前帧的存入，等到下一帧的时候，当前帧就成了上一帧了
        reward_base_pos = (self.info["base_distance"] - info["base_distance"]) * DcmmCfg.reward_weights["r_base_pos"]#奖励权重是0，self代表的是上一步，底座到物体的距离
        reward_ee_pos = (self.info["gripper_dist"] - info["gripper_dist"]) * DcmmCfg.reward_weights["r_ee_pos"]
        reward_ee_precision = math.exp(-50*info["gripper_dist"]**2) * DcmmCfg.reward_weights["r_precision"]
        #reward_far = 0.0
        #if info["ee_distance"] > self.init_ee_distance + DcmmCfg.tracking_far_margin:
            #reward_far = - DcmmCfg.reward_weights["r_far"]
        ## Collision Penalty
        # Compute the Penalty when the arm is collided with the mobile base
        reward_collision = 0
        if self.contacts['base_contacts'].size != 0:
            reward_collision = DcmmCfg.reward_weights["r_collision"]
        tip1_pos = self.Dcmm.data.site("gripper1_tip").xpos
        tip2_pos = self.Dcmm.data.site("gripper2_tip").xpos
        ee_pos = (tip1_pos + tip2_pos) / 2.0
        obj_pos = self.Dcmm.data.body(self.object_name).xpos
        #print(obj_pos)
        dx, dy, dz = ee_pos - obj_pos

        sigma = DcmmCfg.reward_weights.get("axis_sigma", 0.05)#sigma很小就会导致奖励函数很尖锐,必须及其精准对准轴线才能获得奖励,sigma
        #较大就会变的很平缓，稍微偏离轴线也能获得奖励

        r_x = math.exp(- (dx / sigma) ** 2)
        r_y = math.exp(- (dy / sigma) ** 2)
        r_z = math.exp(- (dz / sigma) ** 2)

        w_x = DcmmCfg.reward_weights.get("r_axis_x", 0.3)
        w_y = DcmmCfg.reward_weights.get("r_axis_y", 0.3)
        w_z = DcmmCfg.reward_weights.get("r_axis_z", 0.3)
        #w_min = DcmmCfg.reward_weights.get("r_axis_min", 0.4)

        reward_axis_xyz = (
            w_x * r_x +
            w_y * r_y +
            w_z * r_z 
            
        )
        ## Constraint Penalty
        # Compute the Penalty when the arm joint position is out of the joint limits
        reward_constraint = 0 if self.arm_limit else -1
        reward_constraint *= DcmmCfg.reward_weights["r_constraint"]

        ## Object Touch Success Reward
        # Compute the reward when the object is caught successfully by the hand
        if self.step_touch:#手是否与物体接触
            # print("TRACK SUCCESS!!!!!")
            if not self.reward_touch:
                self.catch_time = self.Dcmm.data.time - self.start_time
            self.reward_touch = DcmmCfg.reward_weights["r_touch"][self.task]
        else:
            self.reward_touch = 0
        # 【关键平移：速度匹配逻辑】
        # 获取末端和物体的线速度
        ee_vel = obs["arm"]["ee_v_lin_3d"]
        obj_vel = obs["object"]["v_lin_3d"]
        
        # 计算余弦相似度：关注方向一致性
        norm_ee = np.linalg.norm(ee_vel) + 1e-6
        norm_obj = np.linalg.norm(obj_vel) + 1e-6
        cos_sim_vel = np.dot(ee_vel, obj_vel) / (norm_ee * norm_obj)
        reward_ee_pos_absolute = -info["gripper_dist"] * 2.0##################有问题###############################
                # 精细化中心对齐奖励保持不变
        reward_precision_base = math.exp(-50 * dist_xy**2) * 5.0 + \
                                          math.exp(-50 * (obj_pos_internal[2] - gripper_center[2])**2) * 2.0
                # 2. 【新增】极高精度线性拉力 (High-Precision Pull)
    # 只有当距离小于 3cm 时触发，强迫误差向 0 逼近
        # ==================== 【基于物理位置 qpos 的闭合奖励】 ====================
        reward_grasp_prime = 0.0
        
        # 获取两个夹爪当前的物理位移（单位：米，0是闭合，0.04是开）
        qpos1 = self.Dcmm.data.joint("gripper1_axis").qpos[0]#slide只有一个自由度，所以数组长度为1，就是在range范围里面取的range="0.005 0.04"
        qpos2 = self.Dcmm.data.joint("gripper2_axis").qpos[0]
        #print(f"Gripper Qpos: {qpos1:.6f}, {qpos2:.6f}")
        # current_qpos_sum = qpos1 + qpos2

        if dist_to_obj < 0.20:  # 当距离小于 4cm 时触发
            # 1. 趋势奖励：只要 qpos 在减小（说明正在闭合），就给奖励
            # prev - current > 0 说明在往 0 走
            # 检查键是否存在
            if "qpos_sum" not in self.info or "qpos_sum" not in info:
                print("错误:info 字典中找不到 qpos_sum 键！")
                print(f"self.info 所有的键: {self.info.keys()}")
            qpos_delta = self.info["qpos_sum"] - info["qpos_sum"]
            #print("qpos_delta:", qpos_delta)
            #print(qpos1, qpos2)
            reward_grasp_prime += np.clip(qpos_delta, 0, None) * 7000.0  # 物理位移很小，系数要大。下限是0，上限是none就是没有上限
            dist_factor = 1.0 / (dist_to_obj + 0.01)
            reward_grasp_prime += np.clip(qpos_delta, 0, None) * 50.0 * dist_factor     
        # 重新定义预张开逻辑：只有远的时候才奖励张开（靠近 0.04）
        reward_pre_open = 0
        # 3. 【新增】正圆心约束 (Centering Constraint)
        # 如果 XY 任何一个轴偏离超过 5mm，就额外扣分，强迫 AI 寻找真正的圆心
        reward_center_constraint = - (error_x + error_y) * 10.0
        reward_center_constraint = 0
        reward_precision_center = reward_precision_base +  reward_center_constraint
        # 在你的 reward 函数或 step 函数中：

# 1. 获取关节在 qpos 中的起始索引
# 这里的 'gripper1_axis' 必须和 XML 中的 joint name 一致
        g1_addr = self.Dcmm.model.jnt_qposadr[mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper1_axis')]
        g2_addr = self.Dcmm.model.jnt_qposadr[mujoco.mj_name2id(self.Dcmm.model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper2_axis')]

        # 2. 读取当前的位置值
        g1_pos = self.Dcmm.data.qpos[g1_addr]
        g2_pos = self.Dcmm.data.qpos[g2_addr]

        # 3. 打印出来
        # print(f"--- Gripper Position Debug ---")
        # print(f"Finger 1 (Red Tip): {g1_pos:.6f} m")
        # print(f"Finger 2 (Green Tip): {g2_pos:.6f} m")
        # if 0.20 < dist_to_obj < 0.50:
        #     reward_pre_open = (qpos1 + qpos2) * 50.0
            #print(qpos1, qpos2)
            #print("0.20 < dist_to_obj < 0.30")
        # 【核心修改：完全同步 Track 任务的精华】
        # 我们把 reward_axis_xyz, reward_vel, reward_distance_away 全部加进来
        # 只有距离近时才开启引导，权重建议 0.5
        #reward_vel = cos_sim_vel * 0.5 if info["gripper_dist"] < 0.3 else 0.0
        r_x_precise = math.exp(-200 * (dx)**2) 
        r_z_precise = math.exp(-200 * (dz)**2)
        # 计算开度分数0.005
        ori_reward = self.compute_arm_alignment_reward()
        open_score = (qpos1 + qpos2) 
        is_properly_aligned = (dist_xy < 0.05) #and (abs(error_z) < 0.06)
        reward_hand = 0


        if self.task == "Catching":
            reward_orient = 0
            if self.stage == "tracking":#跟踪阶段（抓取任务）
                reward_ctrl = - 5*self.norm_ctrl(ctrl_dict, {"hand"})
                # 2. 【核心修改】精准对齐奖 (X 和 Z)
                # 我们希望 X 和 Z 尽早归零。使用强力的高斯奖励
                # 如果 X 偏离 1cm，分数就会大幅下降
                rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
                local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
                hand_z_axis = np.array([0, 0, 1])
                reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
                rewards = (
                    reward_base_pos +        # 基础位置趋势
                    reward_ee_pos +          # 末端位置趋势！！！！！！！！！！！！！！！！！！！！！！！！！，这个有问题
                    reward_ee_precision +    # 全局精度奖！！！！！！！！！！！！！！！！！！！！！！！！这个也是ee_distance，也有问题
                    reward_axis_xyz +        # 【新增】XYZ轴强力磁吸这个也是ee_distance
                    reward_distance_away +   # 【新增】远离即惩罚ee_distance
                    reward_orient +          # 姿态对准
                    #reward_precision_center + # 高精度中心对齐
                    reward_hand +        # 预张开动作
                    reward_collision + 
                    reward_constraint + #只要机械臂超出了预设的物理极限或安全范围56
                    self.reward_touch +
                    reward_ee_pos_absolute #eedistance
                    +ori_reward+reward_ctrl
                    #reward_alignment_xz
                )
                # print("reward_base_pos:", reward_base_pos)
                # print("reward_ee_pos:", reward_ee_pos)
                # print("reward_ee_precision:", reward_ee_precision)
                # print("reward_axis_xyz:", reward_axis_xyz)
                # print("reward_distance_away:", reward_distance_away)
                # print("reward_orient:", reward_orient)
                # print("reward_collision:", reward_collision)
                # print("reward_constraint:", reward_constraint)
                # print("reward_touch:", self.reward_touch)
                # print("reward_ee_pos_absolute:",reward_ee_pos_absolute)
                # print("reward_hand:", reward_hand)
                # print("ori_reward:", ori_reward)
                #print("reward_alignment_xz:",reward_alignment_xz)
                
            #rewards = reward_base_pos + reward_ee_pos + reward_ee_precision + reward_orient + reward_ctrl + reward_collision + reward_constraint + self.reward_touch + reward_distance_away + reward_axis_xyz + reward_vel
            else:#抓取阶段
            # 1. 基础项保留
                reward_ctrl = - 0.5*self.norm_ctrl(ctrl_dict, {"base", "arm"}) - 2*self.norm_ctrl(ctrl_dict, {"hand"})
                reward_orient = 1.0 # 抓取阶段姿态默认给满分
                ##################################################################################################

                
                #######################################################################################################
                # 2. 空间对齐判定 (核心改进点)
                # 使用我们之前讨论的阈值：XY偏差<3cm，高度偏差<5cm
                #is_properly_aligned = (dist_xy < 0.10) and (abs(error_z) < 0.10)
                # 在你的 reward 函数中计算
                rel_vel = self.Dcmm.data.body("arm_seg6").cvel[3:6] - self.object_vel6d[:3] # 相对线速度
                speed = np.linalg.norm(rel_vel)
                allowed_speed = 0.1 + 1.8 * info["gripper_dist"]
                # --- 4. 计算速度惩罚项 ---
                # 只有 speed > allowed_speed 时，speed_error 才会大于 0
                speed_error = np.maximum(0, speed - allowed_speed)
                # reward_velocity_limit = -0.5 * speed_error
                reward_velocity_limit = 0
                # 3. 闭合趋势奖励 (替代原有的 reward_closure)
                reward_closure = 0.0
                qpos_delta = self.info["qpos_sum"] - info["qpos_sum"]#上一帧-当前帧。大于0表示正在闭合
                #print("qpos_delta:", qpos_delta)
                # if is_properly_aligned:
                #     # 只有包络住物体了，才奖励闭合动作
                #     reward_closure = np.clip(qpos_delta, -0.01, 0.01) * 1000.0 
                    #reward_centering = 20.0 
                # else:
                #     # 没对准时闭合要惩罚，并引导对准中心
                #     if qpos_delta > 0: 
                #         reward_closure = -5.0 
                    #reward_centering = math.exp(-100 * dist_xy**2) * 50.0

                # 4. 接触判定 (整合原有的 reward_double_touch)
                reward_contact = 0.0
                if is_touching_l and is_touching_r:
                    if info["qpos_sum"] < 0.015: # 参考 test.xml 最小间距
                        reward_contact = -2.0 # 空抓惩罚
                    else:
                        reward_contact = 100.0 # 真正抓到物体的奖励
                        #reward_closure = 0.0   # 抓住了就不再需要闭合趋势奖
                elif is_touching_l or is_touching_r:
                    reward_contact = 0.0 # 单指触碰奖励，鼓励继续闭合
                r_y_precise = math.exp(-200 * (dy)**2)
                reward_alignment_y = (dy) * 8.0
                # # 5. 提升奖励 (替代并优化原有的 reward_lift)
                # reward_lift = 0.0
                # # 必须双指接触且物体重心显著高于初始平面（比如 > 0.08）
                # if is_touching_l and is_touching_r and obj_pos_internal[2] > 0.08:
                #     # 基础奖金 + 高度线性奖金，鼓励拎得更高
                #     reward_lift = 50.0 + (obj_pos_internal[2] - 0.08) * 500.0
                #     if obj_pos_internal[2] > 0.15:
                #         self.reward_touch += 100.0 # 终极任务完成奖

                # 6. 稳定性奖励 (保留你原来的逻辑)
                if self.reward_touch:
                    self.reward_stability = (info["env_time"] - self.catch_time) * DcmmCfg.reward_weights["r_stability"]
                else:
                    self.reward_stability = 0.0

                seg6_xmat = self.Dcmm.data.body("arm_seg6").xmat.reshape(3, 3)
                seg6_forward_dir = seg6_xmat[:, 2]
                relative_vec = obj_pos - gripper_center
                dist_total = np.linalg.norm(relative_vec)
                axial_dist = np.dot(relative_vec, seg6_forward_dir)#夹爪离物体还有多深
                lateral_dist = np.sqrt(np.maximum(0, dist_total**2 - axial_dist**2))#衡量歪了多少
                reward_center_alignment = 1.5 * math.exp(- (lateral_dist / 0.007) ** 2)
                unit_relative_vec = relative_vec / (dist_total + 1e-6)
                alignment_dot = np.maximum(0, np.dot(seg6_forward_dir, unit_relative_vec))
                reward_perfect_aim = 1.0 * np.power(alignment_dot, 20)
                reward_axial_reach = 1.5 * math.exp(- (dist_total / 0.02) ** 2)
                # --- E. 空间锁定奖 (Volume Bonus) ---
                # 判定：物体是否已经处于两指中间。
                # axial_dist > 0 确保物体在夹爪前方，横向偏移 < 1cm 确保在指缝内。
                # is_in_grasp_volume = (0 < axial_dist < 0.035) and (lateral_dist < 0.01)
                # reward_volume_lock = 20.0 if is_in_grasp_volume else 0.0
                if (0 < axial_dist < 0.035):
                    # print("########################true########################")
                    #if self.info["qpos_sum"] - info["qpos_sum"]>0:#大于0说明正在闭合
                        open_score = (qpos1 + qpos2)
                        #print(open_score)
                        # 情况 A：已经对准了，全力诱导闭合
                        # 此时闭合分数 = 最大开度 - 当前开度
                        reward_hand = (0.08 - open_score) * 150.0 
                else:
                     reward_hand = -(0.08 - open_score) * 100.0 
                # 最终求和：剔除掉那些值为0的旧变量，加入新的有效引导项
                rewards = (reward_base_pos + reward_ee_pos + reward_ee_precision + 
                          reward_orient + reward_ctrl + reward_collision #+ reward_vel 
                          +
                          reward_constraint + self.reward_touch + self.reward_stability + reward_axis_xyz +
                          reward_contact + reward_closure +reward_hand  + ori_reward + reward_velocity_limit+reward_center_alignment
                          +reward_axial_reach+reward_perfect_aim )#reward_alignment_xz  #reward_precision_center
                          #reward_alignment_y +reward_axis_xyz +reward_ee_pos_absolute)
                # print("reward_base_pos:")
                # print(reward_base_pos)
                # print("reward_ee_pos:")
                # print(reward_ee_pos)
                # print("reward_ee_precision:")
                # print(reward_ee_precision)      
                # print("reward_orient:")
                # print(reward_orient)
                # print("reward_ctrl:")
                # print(reward_ctrl)
                # print("reward_collision:")
                # print(reward_collision)
                # print("reward_constraint:")
                # print(reward_constraint)
                # print("reward_touch:")
                # print(self.reward_touch)
                # print("reward_stability:")
                # print(self.reward_stability)
                # print("reward_contact:")
                # print(reward_contact)
                # print("reward_closure:")
                # print(reward_closure)
                # print("reward_hand:")
                # print(reward_hand)
                # print("ori_reward:")
                # print(ori_reward)
                # print("reward_axis_xyz:")
                # print(reward_axis_xyz)
                # print("reward_velocity_limit:")
                # print(reward_velocity_limit)
        elif self.task == 'Tracking':
            ## Ctrl Penalty
            # Compute the norm of base and arm movement through the current actions in the grasping stage
            reward_ctrl = - 0.5*self.norm_ctrl(ctrl, {"base", })- self.norm_ctrl(ctrl, {"hand"})
            ## Object Orientation Reward
            # Compute the dot product of the velocity vector of the object and the z axis of the end_effector
            # C. 【新增】速度匹配奖励 (Velocity Matching Reward)
            # 获取末端和物体的线速度
            ee_vel = obs["arm"]["ee_v_lin_3d"]
            obj_vel = obs["object"]["v_lin_3d"]
            
            # 计算余弦相似度 (Cosine Similarity)：关注方向一致性
            # 即使位置有一点误差，只要速度方向对，就能跟上节奏
            norm_ee = np.linalg.norm(ee_vel) + 1e-6#计算速率，加一个很小的数防止除零错误
            norm_obj = np.linalg.norm(obj_vel) + 1e-6
            cos_sim_vel = np.dot(ee_vel, obj_vel) / (norm_ee * norm_obj)#计算余弦相似度，只关注方向，不关注大小，可以速度更大以便追赶上，不然可能都是相同的速度永远追不上
            
            # 只有当距离比较近 (<0.3m) 时才开始强烈奖励速度匹配，避免远距离时为了凑方向而乱跑
            if info["ee_distance"] < 0.3:
                reward_vel = cos_sim_vel * 0.5  # 权重建议 0.5
            else:
                reward_vel = 0.0
            rotation_matrix = quaternion_to_rotation_matrix(obs["arm"]["ee_quat"])
            local_velocity_vector = np.dot(rotation_matrix.T, obs["object"]["v_lin_3d"])
            hand_z_axis = np.array([0, 0, 1])
            reward_orient = abs(cos_angle_between_vectors(local_velocity_vector, hand_z_axis)) * DcmmCfg.reward_weights["r_orient"]
            ## Add up the rewards
            rewards = (
                    reward_ctrl +
                    reward_base_pos +        # 基础位置趋势
                    reward_ee_pos +          # 末端位置趋势！！！！！！！！！！！！！！！！！！！！！！！！！，这个有问题
                    reward_ee_precision +    # 全局精度奖！！！！！！！！！！！！！！！！！！！！！！！！这个也是ee_distance，也有问题
                    reward_axis_xyz +        # 【新增】XYZ轴强力磁吸这个也是ee_distance
                    reward_distance_away +   # 【新增】远离即惩罚ee_distance
                    reward_orient +          # 姿态对准
                    #reward_precision_center + # 高精度中心对齐
                    reward_hand +        # 预张开动作
                    reward_collision + 
                    reward_constraint + #只要机械臂超出了预设的物理极限或安全范围56
                    self.reward_touch +
                    reward_ee_pos_absolute #eedistance
                    +ori_reward
                    #reward_alignment_xz
                )
            # print("reward_base_pos:", reward_base_pos)
            # print("reward_ee_pos:", reward_ee_pos)
            # print("reward_ee_precision:", reward_ee_precision)
            # print("reward_axis_xyz:", reward_axis_xyz)
            # print("reward_distance_away:", reward_distance_away)
            # print("reward_orient:", reward_orient)
            # print("reward_collision:", reward_collision)
            # print("reward_constraint:", reward_constraint)
            # print("reward_touch:", self.reward_touch)
            # print("reward_ee_pos_absolute:",reward_ee_pos_absolute)
            # print("reward_hand:", reward_hand)
            # print("ori_reward:", ori_reward)
            if self.print_reward:
                if reward_constraint < 0:
                    print("ctrl: ", ctrl_array)
                print("### print reward")
                print("reward_ee_pos: {:.3f}, reward_ee_precision: {:.3f}, reward_orient: {:.3f}, reward_ctrl: {:.3f}, \n".format(
                    reward_ee_pos, reward_ee_precision, reward_orient, reward_ctrl
                ) + "reward_collision: {:.3f}, reward_constraint: {:.3f}, reward_touch: {:.3f}".format(
                    reward_collision, reward_constraint, self.reward_touch
                ))
                print("total reward: {:.3f}\n".format(rewards))
                print("reward_distance_away")
                print(reward_distance_away)
        else:
            raise ValueError("Invalid task: {}".format(self.task))
        
        return rewards

    def _step_mujoco_simulation(self, action_dict):
            ''' 
            actions_dict = {
                    'arm': arm_tensor,
                    'base': base_tensor, (注意：现在这里是 3 维向量 [val1, val2, mode])
                    'hand': hand_tensor,
                } 
            '''
            # ================== 【修改开始：底盘运动模式解析】 ==================
            # 1. 定义物理参数 (建议之后移到 Config 文件中)
            MAX_LIN_VEL = 2.0   # 最大线速度 (m/s)
            MAX_ROT_VEL = 1.5   # 最大自转角速度 (rad/s)
            MAX_STEER = 0.6     # Ackerman 最大转向角 (rad, 约35度)
            WHEELBASE = 0.5     # 轴距 (m)

            # 2. 获取神经网络输出的 3 个原始值
            raw_base = action_dict['base'] 
            val_1, val_2, mode_signal = raw_base[0], raw_base[1], raw_base[2]

            # 3. 初始化目标速度
            cmd_vx, cmd_vy, cmd_omega = 0.0, 0.0, 0.0

            # 4. 模式判定逻辑
            # --- 模式 A: 平移 (Translation) ---
            if mode_signal < -0.3:
                # 逻辑：需要纵向速度(Vy) + 横向速度(Vx)
                cmd_vy = val_1 * MAX_LIN_VEL 
                cmd_vx = val_2 * MAX_LIN_VEL
                cmd_omega = 0.0 # 平移时不自转

            # --- 模式 B: 自转 (Spin) ---
            elif mode_signal > 0.3:
                # 逻辑：原地旋转，只需要角速度(Omega)
                cmd_vx = 0.0
                cmd_vy = 0.0
                cmd_omega = val_1 * MAX_ROT_VEL # val_1 控制旋转快慢
                # val_2 在这里被忽略

            # --- 模式 C: Ackerman (Car-like) ---
            else:
                # 逻辑：像车一样开，需要 纵向速度 + 转向角度
                speed = val_1 * MAX_LIN_VEL
                steer_angle = val_2 * MAX_STEER # val_2 映射为转向角(弧度)

                cmd_vy = speed
                cmd_vx = 0.0 # 车不能横移
                # 利用阿克曼公式计算角速度: omega = v / L * tan(delta)
                cmd_omega = (speed / WHEELBASE) * math.tan(steer_angle)

            # 5. 【关键】将计算出的三个分量全部赋值给底层控制器
            # 原来只赋值 [0:2]，现在赋值 [0], [1], [2]
            self.Dcmm.target_base_vel[0] = cmd_vx *0.4   # 横向速度
            self.Dcmm.target_base_vel[1] = cmd_vy  *0.4 # 纵向速度
            #self.Dcmm.target_base_vel[2] = cmd_omega # 转向角速度 (原来这里是默认的0)
            
            # ================== 【修改结束】 ==================

            # (以下所有代码保持完全不变)
            ## TODO: Low-Pass-Filter the Base Velocity
            # self.Dcmm.target_base_vel[0:2] = action_dict['base'] <-- 这行旧代码被删除了
            
            action_arm = np.concatenate((action_dict["arm"], np.zeros(3)))#ik求解器需要七个数，后面三个元素补0表示不需要改变后面三个元素
            result_QP, _ = self.Dcmm.move_ee_pose(action_arm)#return result_QP:6个关节角度
            if result_QP[1]:#表示 IK 是否求解成功，或者该解是否满足关节限制。，一个布尔标志
                self.arm_limit = True
                self.Dcmm.target_arm_qpos[:] = result_QP[0]
            else:
                #print("IK Failed!!!")
                self.arm_limit = False
            # if self.info["gripper_dist"] < 0.20:#当夹爪距离物体小于20cm时才执行手部动作
            #     # 使用循环直接修改
            #     for key in action_dict["hand"]:
            #         action_dict["hand"][key] = -2
            self.Dcmm.action_hand2qpos(action_dict["hand"])#12维
            # Add Target Action to the Buffer
            self.update_target_ctrl()
            # Reset the Criteria for Successfully Touch
            self.step_touch = False#手是否与物体接触
            for _ in range(self.steps_per_policy):#重复执行循环体 self.steps_per_policy 次。每个动作持续多少仿真步
                # Update the control command according to the latest policy output
                self.Dcmm.data.ctrl[:-1] = self._get_ctrl()
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
                
                # (以下关于碰撞判定、terminated 逻辑的代码完全保持不变)
                if self.contacts['base_contacts'].size != 0:
                    self.terminated = True
                    print(colored("!!! Base Collided !!!", "red")) # 检查是不是底座撞了
                # 1. 判定是否撞到了手臂（非手指部分）
                # 这里的 arm_start_geom_id 是你在 __init__ 中获取的手臂第一个零件的 ID
                mask_coll = self.contacts['object_contacts'] < self.arm_start_geom_id

                # 2. 精确判定是否撞到了两个手指（使用你在 __init__ 中存好的 57 和 59）
                mask_finger1 = (self.contacts['object_contacts'] == self.f1_geom_id)
                mask_finger2 = (self.contacts['object_contacts'] == self.f2_geom_id)
                # mask_coll = self.contacts['object_contacts'] < self.hand_start_id
                # finger1_id = self.hand_start_id + 1
                # finger2_id = self.hand_start_id + 2

                # mask_finger1 = (self.contacts['object_contacts'] == finger1_id)
                # mask_finger2 = (self.contacts['object_contacts'] == finger2_id)

                if self.step_touch == False:
                    if self.task == "Tracking":
                        if np.any(mask_finger1) or np.any(mask_finger2):
                            self.step_touch = True
                    elif self.task == "Catching":
                        if np.any(mask_finger1) and np.any(mask_finger2):
                            self.step_touch = True

                if not self.terminated:
                    if self.task == "Catching":
                        self.terminated = np.any(mask_coll)
                        #print(self.terminated)
                        #print(colored("!!! Stage Error: Object escaped during grasping1 !!!", "yellow"))
                    elif self.task == "Tracking":
                        self.terminated = np.any(mask_coll) or np.any(mask_finger1) or np.any(mask_finger2)
                        #print(colored("!!! Stage Error: Object escaped during grasping2 !!!", "yellow"))
                if self.terminated:
                    break         
            # As of MuJoCo 2.0, force-related quantities like cacc are not computed
            # unless there's a force sensor in the model.
            # See https://github.com/openai/gym/issues/1541
            ####################################################################################################################################修改前
            # if self.Dcmm.data.time - self.start_time < self.object_static_time:#已经过的仿真秒数。是否处于物体静止期，物体静止阶段
            #     self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
            #                                 velocity=np.zeros(6))#物体保持原地不动
            #     self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]#对物体施加一个向上的力，抵消重力，这样物体就不会掉下去
            # elif not self.object_throw:#物体投掷阶段
            #     self.Dcmm.set_throw_pos_vel(pose=np.concatenate((self.object_pos3d[:], self.object_q[:])),
            #                                 velocity=self.object_vel6d[:])
            #     self.Dcmm.data.ctrl[-1] = 0.0
            #     self.Dcmm.data.ctrl[-1] = self.random_mass * -self.Dcmm.model.opt.gravity[2]
            #     self.object_throw = True

            # mujoco.mj_step(self.Dcmm.model, self.Dcmm.data)#推进仿真一步
            # mujoco.mj_rnePostConstraint(self.Dcmm.model, self.Dcmm.data)#计算受约束条件下的逆动力学（比如接触力、约束力），这样才能获得正确的接触信息。

            # # Update the contact information
            # self.contacts = self._get_contacts()#从 MuJoCo 里读取当前所有的碰撞接触情况。返回的是一个字典，是底座手地面等跟物体接触的信息
            # # Whether the base collides
            
            # # Base collision → always terminate
            # if self.contacts['base_contacts'].size != 0:
            #     self.terminated = True

            # # -------------------------------
            # # 1. 基础碰撞判断（地面/环境）
            # # -------------------------------
            # mask_coll = self.contacts['object_contacts'] < self.hand_start_id#物体是不是和夹爪以外的东西接触了

            # # -------------------------------
            # # 2. 取出手指的 geom_id（两根手指）
            # # -------------------------------
            # finger1_id = self.hand_start_id + 1    # gripper1
            # finger2_id = self.hand_start_id + 2    # gripper2

            # # -------------------------------
            # # 3. 判断物体是否碰到各个手指
            # # -------------------------------
            # mask_finger1 = (self.contacts['object_contacts'] == finger1_id)
            # mask_finger2 = (self.contacts['object_contacts'] == finger2_id)

            # # -------------------------------
            # # 4. 根据 task 类型判断接触是否发生成功
            # # -------------------------------
            # if self.step_touch == False:

            #     if self.task == "Tracking":
            #         # Tracking：只要任意一个手指碰到球即可
            #         if np.any(mask_finger1) or np.any(mask_finger2):
            #             self.step_touch = True

            #     elif self.task == "Catching":
            #         # Catching：两个手指都碰到球才算完成
            #         if np.any(mask_finger1) and np.any(mask_finger2):
            #             self.step_touch = True

            #     # ===== 【这里插入】一旦 Tracking 接触，立刻冻结控制 =====
            # # if self.task == "Tracking" and self.step_touch:
            # #     # 冻结底盘（防止继续往前冲）
            # #     self.Dcmm.target_base_vel[:] = 0.0

            # #     # 冻结机械臂（保持当前姿态，防止 PID 拉走）
            # #     self.Dcmm.target_arm_qpos[:] = self.Dcmm.data_arm.qpos[:]
            #     # ===== 【插入结束】 =====
            # # -------------------------------
            # # 5. 判断物体是否“掉落”或失败
            # # -------------------------------
            # if not self.terminated:#初始是false
            #     if self.task == "Catching":
            #         # 掉地 = 失败
            #         self.terminated = np.any(mask_coll)

            #     elif self.task == "Tracking":
            #         # Tracking：掉地 或 碰到任意手指后就算结束（tracking 不需要长时间抓住）
            #         self.terminated = np.any(mask_coll) or np.any(mask_finger1) or np.any(mask_finger2)

            # # -------------------------------
            # # 6. 如果失败则立即结束 episode
            # # -------------------------------
            # if self.terminated:
            #     break

            ###############################################################################################################################修改前
            '''
            if self.contacts['base_contacts'].size != 0:
                self.terminated = True#底座如果碰撞，马上结束
            mask_coll = self.contacts['object_contacts'] < self.hand_start_id#物体和 环境（比如地面）发生接触。
            mask_finger = self.contacts['object_contacts'] > self.hand_start_id#物体和手指碰撞
            mask_hand = self.contacts['object_contacts'] >= self.hand_start_id#物体和整个手碰撞
            mask_palm = self.contacts['object_contacts'] == self.hand_start_id#物体和手掌碰撞
            #这四个全是布尔数组
            # Whether the object is caught
            
            if self.step_touch == False:#是否触碰到物体
                if self.task == "Catching" and np.any(mask_hand):#any的意思是,只要数组里面有一个true,那么就返回true
                    self.step_touch = True
                elif self.task == "Tracking" and np.any(mask_palm):
                    self.step_touch = True
            # Whether the object falls
            if not self.terminated:
                if self.task == "Catching":
                    self.terminated = np.any(mask_coll)
                elif self.task == "Tracking":
                    self.terminated = np.any(mask_coll) or np.any(mask_finger)
            # If the object falls, terminate the episode in advance根据当前任务类型判断“物体是否发生了不可接受的碰撞(掉落或被不允许的部位触碰）”，如果发生则把 terminated 设为 True 并 提前结束当前仿真步循环，从而使整个 episode 提前终止（通常作为失败条件）。
            if self.terminated:
                break
            '''
    def step(self, action):#这里面的action是从PPO传进来的，是[-1,1]*denorm
        ''' 
        actions_dict = {
                'arm': arm_tensor,4维
                'base': base_tensor,2维
                'hand': hand_tensor,12维
            }
            接动作,运行 mujoco,生 obs,生 reward,返回给 PPO
        '''


        self.steps += 1
        # --- 验证输出范围开始 ---
#         if isinstance(action, dict) and 'hand' in action:
#             h_act = action['hand']
#             # 打印当前步 hand 动作的统计信息
#             if self.steps % 10 == 0: # 每10步打印一次，防止刷屏
#                 print(f"DEBUG [Hand Action Raw] -> Min: {np.min(h_act):.4f}, Max: {np.max(h_act):.4f}, Mean: {np.mean(h_act):.4f}")
#     # --- 验证输出范围结束 ---
# # --- 正确的清零方式 ---
        # if isinstance(action, dict):
        #     for key in action.keys():
        #         action[key][:] = 0.0  # 对字典里的每个数组进行清零
        # else:
        #     action[:] = 0.0
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
        gripper_center = self._get_gripper_center()
        # === 【新增保护】 ===
        # 检查观测值里有没有 NaN，如果有，强行重置环境，防止崩到底层
        flat_obs = np.concatenate([v.flatten() for v in obs.values() if isinstance(v, np.ndarray)])
        if np.isnan(flat_obs).any():
            #print(" 检测到 NaN,环境将重置以避免崩溃。")
            self.reset()
            # 返回全 0 的观测，让网络“混”过这一步，或者返回 done=True
            return self._get_obs(), 0, True, False, self._get_info()
        # ===================
        if self.task == 'Catching':
            if info['gripper_dist'] < DcmmCfg.distance_thresh and self.stage == "tracking":
                self.stage = "grasping"
            elif info['gripper_dist'] >= DcmmCfg.distance_thresh * 1.2 and self.stage == "grasping":
                self.terminated = True
                #print(colored("!!! Stage Error: Object escaped during grasping4 !!!", "yellow"))
        # Design the reward function
        reward = self.compute_reward(obs, info, action)
        self.info["base_distance"] = info["base_distance"]
        self.info["ee_distance"] = info["ee_distance"]
        self.info["gripper_dist"] = info["gripper_dist"]
        self.info["qpos_sum"] = info["qpos_sum"]
        # Rendering
        imgs = self.render() if self.render_mode is not None else None
        # Update the imgs
        info['imgs'] = imgs
        ctrl_delay = np.array([len(self.action_buffer['base']),
                               len(self.action_buffer['arm']),
                               len(self.action_buffer['hand'])])
        info['ctrl_params'] = np.concatenate((self.k_arm, self.k_drive, self.k_hand, ctrl_delay))
        # The episode is truncated if the env_time is larger than the predefined time
        # if self.closed:
        #     truncated = True
        #     terminated = True
        #else:
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
        #print("##############################################################################################################3")
        #print("time_ouut")
    #else: truncated = False
        done = terminated or truncated
        if done:
            # TEST ONLY
            # self.reset()
            pass
        #if hasattr(self, 'hand_ctrl_id'):
            #print(f"[DEBUG 3 - CTRL] MuJoCo Ctrl Buffer (Hand): {self.Dcmm.data.ctrl[self.hand_ctrl_id]}")
        #else:
        # 如果你还不确定索引，打印前10个看看
            #print(f"[DEBUG 3 - CTRL] First 10 Ctrl Channels: {self.Dcmm.data.ctrl[:10]}")
        # 在 step 函数的最后，return 之前
        # if self.steps < 3: # 只看每个 episode 的前 3 步
        #     idx0 = self.Dcmm.model.actuator("hand_actuator_0").id
        #     idx1 = self.Dcmm.model.actuator("hand_actuator_1").id
        #     hand_ctrl_idx = [idx0, idx1]
        #     g1 = self.Dcmm.data.joint("gripper1_axis").qpos[0]
        #     g2 = self.Dcmm.data.joint("gripper2_axis").qpos[0]
        #     ctrl = self.Dcmm.data.ctrl[hand_ctrl_idx] # 或者是你控制夹爪的 ctrl 变量
        #     print(f"STEP {self.steps} -> G1:{g1:.4f}, G2:{g2:.4f}, CTRL:{ctrl}")
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