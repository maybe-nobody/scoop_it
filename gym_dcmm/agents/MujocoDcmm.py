"""
Author: Yuanhang Zhang
Version@2024-10-17
All Rights Reserved
ABOUT: this file constains the basic class of the DexCatch with Mobile Manipulation (DCMM) in the MuJoCo simulation environment.
"""
import os, sys
sys.path.append(os.path.abspath('../'))
import copy
import configs.env.DcmmCfg as DcmmCfg
import mujoco
import os
print(">>> os.getcwd() before model init:", os.getcwd())
import time
from utils.ik_pkg.ik_arm_new import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz

from utils.util import calculate_arm_Te
from utils.pid import PID
import numpy as np
from utils.ik_pkg.ik_arm import IKArm
from utils.ik_pkg.ik_base import IKBase
from scipy.spatial.transform import Rotation as R
from collections import deque
import xml.etree.ElementTree as ET
from utils.ik_pkg.ik_arm_new import IKArmQP

# Function to convert XML file to string
def xml_to_string(file_path):#把 XML 文件内容读取并转成字符串
    try:
        # Parse the XML file
        tree = ET.parse(file_path)#读取并拆解 XML 文件，返回一个 ElementTree 对象（树状结构）,ET:解析和操作 XML 文件
        root = tree.getroot()#获取 XML 树的根节点

        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding='unicode')#把一个 XML 元素（Element 对象）转换成字符串,xml_str 就是整棵树的文本表示
        
        return xml_str
    except Exception as e:
        print(f"Error: {e}")
        return None

DEBUG_ARM = False
DEBUG_BASE = False


class MJ_DCMM(object):
    """
    Class of the DexCatch with Mobile Manipulation (DCMM)一个机器人任务环境
    in the MuJoCo simulation environment.

    Args:
    - model: the MuJoCo model of the Dcmm,dcmm的mujoco模型
    - model_arm: the MuJoCo model of the arm,机械臂的mujoco模型
    - viewer: whether to show the viewer of the simulation,是否显示仿真 3D 窗口
    - object_name: the name of the object in the MuJoCo model,MuJoCo 模型中要操作的物体名字
    - timestep: the simulation timestep,仿真步长
    - open_viewer: whether to open the viewer initially,是否在初始化时打开窗口
    self.model # 底座的 MuJoCo 模型对象
    self.model_arm # 机械臂的 MuJoCo 模型对象
    self.data # 仿真状态数据（底座）
    self.data_arm # 仿真状态数据（机械臂）
    self.arm_base_pos # 底座在世界坐标中的位置
    self.current_ee_pos # 机械臂末端执行器的位置
    self.current_ee_quat # 机械臂末端执行器的四元数姿态
    self.drive_pid, self.steer_pid, self.arm_pid, self.hand_pid # PID 控制器
    self.cmd_lin_x, self.cmd_lin_y # 底盘目标速度
    self.steer_ang, self.drive_vel # 底盘控制目标
    self.target_arm_qpos, self.target_hand_qpos # 机械臂和手指目标关节角

    """    
    def __init__(self, 
                 model=None, 
                 model_arm=None, 
                 viewer=True, 
                 object_name='object',
                 object_eval=False, #训练阶段或评估阶段
                 timestep=0.002):
        self.viewer = None
        self.open_viewer = viewer
        # Load the MuJoCo model
        if model is None:#是否提供了 MuJoCo 模型对象,底座
            if not object_eval: model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_OBJECT_PATH)#将两个路径拼接在一起
            else: model_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_DCMM_LEAP_UNSEEN_OBJECT_PATH)
            self.model_xml_string = xml_to_string(model_path)#把xml内容读取并转换为字符串
            #XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"
            #XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/x1_xarm6_leap_right_unseen_object.xml"
            #XML_ARM_PATH = "urdf/xarm6_right.xml"
        else:
            self.model = model
        if model_arm is None:#机械臂
            model_arm_path = os.path.join(DcmmCfg.ASSET_PATH, DcmmCfg.XML_ARM_PATH)
            self.model_arm = mujoco.MjModel.from_xml_path(model_arm_path)#from_xml_path(path) = 从 XML 文件加载一个模型，返回 MjModel 对象,读取arm.xml文件
        #它是 MuJoCo 内部的模型对象，要想得到仿真环境还要把这个对象给渲染器和仿真器
        else:
            self.model_arm = model_arm


        self.model = mujoco.MjModel.from_xml_string(self.model_xml_string)#from_xml_string把 XML 格式的字符串解析成一个 MuJoCo 模型对象,model并不是已经渲染好的环境，而是骨架
        self.model.opt.timestep = timestep#opt是仿真选项
        self.model_arm.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.data_arm = mujoco.MjData(self.model_arm)#记录仿真运行过程中，每一步的所有状态信息
        self.data.qpos[15:21] = DcmmCfg.arm_joints[:]#机械臂的六个弧度，可以解析出.xml文件里面有几个data可以输入
        self.data.qpos[21:23] = DcmmCfg.hand_joints[:]#每个手指有三个自由度，根部有一个自由度，都是0时表示完全未张开
        self.data_arm.qpos[0:6] = DcmmCfg.arm_joints[:]



        '''
        arm_joints = np.array([
        0.0, 0.0, -0.0, 3.07, 2.25, -1.5 
        ])
        hand_joints = np.array([
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        ])

        '''
        #a1arm的qpos是15维，前7个是底座，中间六个是机械臂，最后两个是夹爪
        #link1-6是机械臂
        #arm_seg是机械臂的刚体qp
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model_arm, self.data_arm)#让仿真环境的所有状态，和你在 self.data 里设置的数值保持一致。
        self.arm_base_pos = self.data.body("arm_base").xpos#返回一个xyz坐标，就是底座在虚拟世界中的坐标，这个是根据xml文件和之前的数值算出来的
        self.current_ee_pos = copy.deepcopy(self.data_arm.body("arm_seg6").xpos)#去xml文件里找link6的刚体
        self.current_ee_quat = copy.deepcopy(self.data_arm.body("arm_seg6").xquat)#把这个数组复制一份全新的副本，后续不管 MuJoCo 的数据怎么变，self.current_ee_pos 都保持当时的数值。

        ## Get the joint ID for the body, base, arm, hand and object
        # Note: The joint id of the mm body is 0 by default
        try:
            _ = self.data.body(object_name)#检查模型里是否存在名为 object_name 的刚体，但不实际使用它。
        except:
            print("The object name is not found in the model!\
                  \nPlease check the object name in the .xml file.")
            raise ValueError
        self.object_name = object_name#这个名字是机械臂要抓取的物体
        # Get the geom id of the hand, the floor and the object
        self.hand_start_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'gripper1') - 1#根据名字查找模型中某个对象的 ID
        #把一个叫 mcp_joint 的 geom 的 id 算出来，再减 1，当成‘手指关节在状态数组里的起始下标’。”
        #mcp_joint是手指关节
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')##########未进行修改
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_name)

        # Mobile Base Control
        self.rp_base = np.zeros(3)
        self.rp_ref_base = np.zeros(3)#底盘位置，参考位置的初始化
        self.drive_pid = PID("drive", DcmmCfg.Kp_drive, DcmmCfg.Ki_drive, DcmmCfg.Kd_drive, dim=4, llim=DcmmCfg.llim_drive, ulim=DcmmCfg.ulim_drive, debug=False)
        self.steer_pid = PID("steer", DcmmCfg.Kp_steer, DcmmCfg.Ki_steer, DcmmCfg.Kd_steer, dim=4, llim=DcmmCfg.llim_steer, ulim=DcmmCfg.ulim_steer, debug=False)
        self.arm_pid = PID("arm", DcmmCfg.Kp_arm, DcmmCfg.Ki_arm, DcmmCfg.Kd_arm, dim=6, llim=DcmmCfg.llim_arm, ulim=DcmmCfg.ulim_arm, debug=False)
        self.hand_pid = PID("gripper", DcmmCfg.Kp_hand, DcmmCfg.Ki_hand, DcmmCfg.Kd_hand, dim=2, llim=DcmmCfg.llim_hand, ulim=DcmmCfg.ulim_hand, debug=False)
        self.cmd_lin_y = 0.0
        """
        PID controller class.

        Inputs:
            setpoint: desired value
            Kp: proportional gain
            Ki: integral gain
            Kd: derivative gain
            offset: offset value (default = 0.0)
        """
        '''
        给 4 组电机分别造一个“PID 调节器”对象，以后每帧把“目标值”和“当前反馈”扔进去，
        它就会算出应该发多少力矩/电流，让误差尽快趋近于 0。
        '''
        self.cmd_lin_x = 0.0#底盘x方向速度指令
        self.arm_act = False#机械臂激活开关
        self.steer_ang = np.array([0.0, 0.0, 0.0, 0.0])#四个转向轮的目标转向角5
        self.drive_vel = np.array([0.0, 0.0, 0.0, 0.0])#四个轮毂电机的目标角速度

        ## Define Inverse Kinematics Solver for the Arm初始化ik求解器
        #self.ik_arm = IKArm(solver_type=DcmmCfg.ik_config["solver_type"], ilimit=DcmmCfg.ik_config["ilimit"], 
                            #ps=DcmmCfg.ik_config["ps"], λΣ=DcmmCfg.ik_config["λΣ"], tol=DcmmCfg.ik_config["ee_tol"])
        self.ik_arm = IKArmQP(
        ee_body_name="arm_seg6",
        joint_names=["arm_joint1","arm_joint2","arm_joint3","arm_joint4","arm_joint5","arm_joint6"],  # 按你模型实际名字
        w_pos=1.0,
        w_rot=0.15,
        damping=1e-3,
        tol_pos=1e-1,
        tol_rot=1e-3,
        max_iters=60,
        max_dq=50.55,
        pgd_iters=40,
        line_search=False,
    )

        #solver_type：用哪种算法，ilimit：最大迭代次数，ps：末端位姿误差的尺度权重，末端误差的缩放因子用来把位置误差变成合适的大小，防止太大导致震荡。
        #λΣ越大，越保守，越稳定但是收敛速度慢
        ## Initialize the camera parameters
        self.model.vis.global_.offwidth = DcmmCfg.cam_config["width"]
        self.model.vis.global_.offheight = DcmmCfg.cam_config["height"]
        self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], DcmmCfg.cam_config["name"])

        ## Initialize the target velocity of the mobile base
        self.target_base_vel = np.zeros(3)#底盘速度指令xy和偏航角PID/运动学模块把它转成 drive_vel 与 steer_ang，让底盘实际 qvel[0,1,5] 逼近它。
        self.target_arm_qpos = np.zeros(6)#机械臂的6个目标弧度
        self.target_hand_qpos = np.zeros(2)#夹爪的目标位移
        ## Initialize the target joint positions of the arm
        self.target_arm_qpos[:] = DcmmCfg.arm_joints[:]#机械臂的6个自由度
        ## Initialize the target joint positions of the hand
        self.target_hand_qpos[:] = DcmmCfg.hand_joints[:]
        #	把配置里“手指初始弧度”一次性拷进来，作为夹爪两滑动关节（gripper1/2_axis）的目标位移（0–0.03 m）。后续想开合爪子，只需改这个向量即可。
        self.ik_solution = np.zeros(6)#	预留缓存，存放逆解算出的 6 个臂关节弧度。上层每次调用 IK 后把结果写进来，再赋给 target_arm_qpos，避免频繁创建新数组。

        self.vel_history = deque(maxlen=4)  # store the last 2 velocities
        #底盘速度滤波/微分器。每帧把最新 qvel[0:2]（vx,vy）或 qvel[5]（ωz）append 进去，PID 或观测器用它算加速度或做滑动平均，抑制仿真噪声。
        self.vel_init = False
        #标志位，第一次收到速度信号后置 True，防止用空队列做差分导致 NaN

        self.drive_ctrlrange = self.model.actuator(4).ctrlrange#把第 4 号驱动电机（通常指某个轮毂）的 MuJoCo ctrl 上下限读出来，单位 N⋅m 或 V（看电机增益）。用来饱和/缩放 PID 输出，防止超调
        self.steer_ctrlrange = self.model.actuator(0).ctrlrange#同理，第 0 号转向电机的 ctrl 范围，用于限幅 steer_pid 输出。

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body(i).name))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits: {}, Damping: {}".format(
                    i, self.model.joint(i).name, self.model.jnt_range[i], self.model.dof_damping[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.data.ctrl)))
        for i in range(len(self.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range: {}".format(
                    i,
                    self.model.actuator(i).name,
                    self.model.joint(self.model.actuator(i).trnid[0]).name,
                    self.model.actuator(i).ctrlrange,
                )
            )
        print("\nMobile Base PID Info: \n")
        print(
            "Drive, P: {}, I: {}, D: {}".format(
                self.drive_pid.Kp,
                self.drive_pid.Ki,
                self.drive_pid.Kd,
            )
        )
        print(
            "Steer, P: {}, I: {}, D: {}".format(
                self.steer_pid.Kp,
                self.steer_pid.Ki,
                self.steer_pid.Kd,
            )
        )
        print("\nArm PID Info: \n")
        print(
            "P: {}, I: {}, D: {}".format(
                self.arm_pid.Kp,
                self.arm_pid.Ki,
                self.arm_pid.Kd,
            )
        )
        print("\nHand PID Info: \n")
        print(
            "P: {}, I: {}, D: {}".format(
                self.hand_pid.Kp,
                self.hand_pid.Ki,
                self.hand_pid.Kd,
            )
        )

        print("\nCamera Info: \n")
        for i in range(self.model.ncam):
            print(
                "Camera ID: {}, Camera Name: {}, Camera Mode: {}, Camera FOV (y, degrees): {}, Position: {}, Orientation: {}, \n Intrinsic Matrix: \n{}".format(
                    i,
                    self.model.camera(i).name,
                    self.model.cam_mode[i],
                    self.model.cam_fovy[i],
                    self.model.cam_pos[i],
                    self.model.cam_quat[i],
                    # self.model.cam_pos0[i],
                    # self.model.cam_mat0[i].reshape((3, 3)),
                    self.cam_matrix,
                )
            )
        print("\nSimulation Timestep: ", self.model.opt.timestep)

    # def move_base_vel(self, target_base_vel):
    #     """
    #     target_base_vel: np.array, shape (3,)
    #         [vx, vy, yaw]，其中 yaw 在 RL 模式下通常是 0
    #     """

    #     # === 新增：从 target_base_vel 里取出 vx, vy ===
    #     vx = target_base_vel[0]
    #     vy = target_base_vel[1]

    #     # === 新增：根据 (vx, vy) 自动补 yaw / 切模式 ===
    #     vx_eff, vy_eff, v_yaw = self.base_motion_selector(vx, vy)
    #     try:
    #         self.steer_ang, self.drive_vel = IKBase(vx_eff, vy_eff, v_yaw)
    #     except Exception as e:
    #         print("[IKBase ERROR]")
    #         print("vx_eff, vy_eff, v_yaw =", vx_eff, vy_eff, v_yaw)
    #         raise e
    #     # === 原有 IK，不改 ===
    #     self.steer_ang, self.drive_vel = IKBase(
    #         vx_eff,
    #         vy_eff,
    #         v_yaw
    #     )

    #     # ====== 后面全部保持原样 ======
    #     current_steer_pos = np.array([
    #         self.data.joint("steer_fl").qpos[0],
    #         self.data.joint("steer_fr").qpos[0],
    #         self.data.joint("steer_rl").qpos[0],
    #         self.data.joint("steer_rr").qpos[0]
    #     ])

    #     current_drive_vel = np.array([
    #         self.data.joint("drive_fl").qvel[0],
    #         self.data.joint("drive_fr").qvel[0],
    #         self.data.joint("drive_rl").qvel[0],
    #         self.data.joint("drive_rr").qvel[0]
    #     ])

    #     mv_steer = self.steer_pid.update(
    #         self.steer_ang, current_steer_pos, self.data.time
    #     )
    #     mv_drive = self.drive_pid.update(
    #         self.drive_vel, current_drive_vel, self.data.time
    #     )

    #     mv_steer = np.clip(mv_steer, self.steer_ctrlrange[0], self.steer_ctrlrange[1])
    #     mv_drive = np.clip(mv_drive, self.drive_ctrlrange[0], self.drive_ctrlrange[1])

    #     return mv_steer, mv_drive

    def move_base_vel(self, target_base_vel):#一个长度为 3 的数组，表示目标速度，xyw角速度
        self.steer_ang, self.drive_vel = IKBase(target_base_vel[0], target_base_vel[1], target_base_vel[2])
        #每个转向轮的目标角度，目标速度
        ####################
        ## No bugs so far ##
        ####################
        # Mobile base steering and driving control 
        # TODO: angular velocity is not correct when the robot is self-rotating.
        current_steer_pos = np.array([self.data.joint("steer_fl").qpos[0],#前轮左转向关节的当前角度
                                      self.data.joint("steer_fr").qpos[0], #前轮右
                                      self.data.joint("steer_rl").qpos[0],#后轮左
                                      self.data.joint("steer_rr").qpos[0]])#后轮右
        current_drive_vel = np.array([self.data.joint("drive_fl").qvel[0],#当前转速
                                      self.data.joint("drive_fr").qvel[0], 
                                      self.data.joint("drive_rl").qvel[0],
                                      self.data.joint("drive_rr").qvel[0]])
        mv_steer = self.steer_pid.update(self.steer_ang, current_steer_pos, self.data.time)#PID 控制器根据目标角度和当前角度，输出转向控制信号。
        mv_drive = self.drive_pid.update(self.drive_vel, current_drive_vel, self.data.time)#PID 控制器根据目标速度和当前速度，输出驱动控制信号。
        if np.all(current_drive_vel > 0.0) and np.all(current_drive_vel < self.drive_vel):
            mv_drive = np.clip(mv_drive, 0, self.drive_ctrlrange[1] / 10.0)
        if np.all(current_drive_vel < 0.0) and np.all(current_drive_vel > self.drive_vel):
            mv_drive = np.clip(mv_drive, self.drive_ctrlrange[0] / 10.0, 0)
        #防止轮子从正转突然变反转（或反之）时控制信号过大。
        mv_steer = np.clip(mv_steer, self.steer_ctrlrange[0], self.steer_ctrlrange[1])
        #把转向控制信号限制在允许范围内（防止打角过大）。
        return mv_steer, mv_drive
    ###############################################################################################################################
    # @staticmethod
    # def base_motion_selector(vx, vy):
    #     """
    #     根据 (vx, vy) 自动决定底座运动方式，
    #     并生成 IKBase 需要的 (vx_eff, vy_eff, v_yaw)

    #     输入:
    #         vx, vy : 神经网络输出的二维底座速度

    #     输出:
    #         vx_eff, vy_eff, v_yaw : 送入 IKBase 的三维速度
    #     """
    #     eps = 1e-3
    #     k_parallel = 0.4   # 横移 / 前进 比例阈值
    #     k_yaw = 1.2        # 横移意图 -> yaw 映射系数 

    #     #1. 几乎不动
    #     if abs(vx) < eps and abs(vy) < eps:
    #         return 0.0, 0.0, 0.0
    #     #return vx, vy, 0.0
    #     #2. 没有前进，不可能 Ackerman
    #     if abs(vy) < eps:
    #         return vx, vy, 0.0

    #     #3. 判断横移 / 前进 比例
    #     ratio = abs(vx) / (abs(vy) + eps)#+eps的原因是防止vy接近0时分母为0

    #     #4. 横移主导：Parallel / 蟹行
    #     if ratio > k_parallel:
    #         print(">>> Parallel Mode Activated")
    #         return vx, vy, 0.0

    #     # 5. 前进主导：Ackerman 
    #     else:
    #         v_yaw = k_yaw * vx
    #     print(">>> Ackerman Mode Activated")
    #     return 0.0, vy, v_yaw
    #     #return vx, vy, 0.0
    #     #return 0.0, 0.0, 0.0
    ###############################################################################################################################
    def move_ee_pose(self, delta_pose):
        """
        Move the end-effector to the target pose.
        delta_pose[0:3]: delta x,y,z
        delta_pose[3:6]: delta euler angles roll, pitch, yaw

        Return:
        - The target joint positions of the arm
        """
        self.current_ee_pos[:] = self.data_arm.body("arm_seg6").xpos[:]#机械臂末端相对于机械臂世界坐标系的位置
        self.current_ee_quat[:] = self.data_arm.body("arm_seg6").xquat[:]#机械臂末端相对于机械臂世界坐标系的四元数
        #print("##############################ee_pos######################################3")
        #print(self.current_ee_pos[:])
        target_pos = self.current_ee_pos + delta_pose[0:3]
        #target_pos[2] -= 0.267#########################################################################################减去的东西
        #print("target")
        #print(target_pos)
        #print(time.perf_counter())
        #print("object")
        #print(self.data.body("object").xpos[:])
        self.current_ee_quat = quat_wxyz_to_xyzw(self.current_ee_quat)##########################################后加的
        r_delta = R.from_euler('zxy', delta_pose[3:6])
        r_current = R.from_quat(self.current_ee_quat)
        target_quat = (r_delta * r_current).as_quat()#四元数xyzw
        #print("#############################target#############################")
        #print(target_pos,target_quat
        result_QP = self.ik_arm_solve(target_pos, target_quat)
        #print("################result_qp##############")
        #print(result_QP)
        if DEBUG_ARM: print("result_QP: ", result_QP)
        # Update the qpos of the arm with the IK solution
        self.data_arm.qpos[0:6] = result_QP[0]
        mujoco.mj_fwdPosition(self.model_arm, self.data_arm)
        
        # Compute the ee_length
        relative_ee_pos = target_pos - self.data_arm.body("arm_base").xpos
        ee_length = np.linalg.norm(relative_ee_pos)


        return result_QP, ee_length
    
    def ik_arm_solve(self, target_pose, target_quate):
        """
        Solve the IK problem for the arm.
        """
        # Update the arm joint position to the previous one
        target_quate = quat_xyzw_to_wxyz(target_quate)######################################################添加的
        Tep = calculate_arm_Te(target_pose, target_quate)#输入四元数的顺序要求是wxyz
        if DEBUG_ARM: print("Tep: ", Tep)
        result_QP = self.ik_arm.solve(self.model_arm, self.data_arm, Tep, self.data_arm.qpos[0:6])
        #print(result_QP)
        return result_QP

    def set_throw_pos_vel(self, 
                          pose = np.array([0, 0, 0, 1, 0, 0, 0]), #位置+四元数
                          velocity = np.array([0, 0, 0, 0, 0, 0])):#线速度+角速度
        self.data.qpos[23:30] = pose
        self.data.qvel[22:28] = velocity
    def action_hand2qpos(self, action_hand):
        """
                Convert the action of the 2-DoF gripper to the joint target positions.

                参数
                ----
                action_hand : array-like, shape (2,)
                    action_hand[0] : gripper1_axis 关节的位移增量（米）
                    action_hand[1] : gripper2_axis 关节的位移增量（米）

                作用
                ----
                - 把控制策略输出的 2 维夹爪动作，累加到 self.target_hand_qpos 里；
                - self.target_hand_qpos 之后会作为 PID 的期望位置，驱动两个滑动关节；
                - 为了防止目标超出关节物理范围，这里会把目标夹爪开合量限制在 [0, 0.03] 米。
        """

        # 更新两个夹爪关节的目标位置（增量控制）action_hand范围是[-0.15,0.15]，因为神经网络最后一层输出的是[-1,1],乘了一个denorm(0.15)
        self.target_hand_qpos[0] += action_hand[0] * 0.2  # gripper1_axis
        self.target_hand_qpos[1] += action_hand[1] * 0.2 # gripper2_axis

        # 安全：把目标位置限制在 MJCF 里给的关节范围 [0, 0.03] 米
        self.target_hand_qpos[:] = np.clip(self.target_hand_qpos, 0.00, 0.04)

    '''
    def action_hand2qpos(self, action_hand):
        """
        Convert the action of the hand to the joint positions.
        """
        # Thumb
        self.target_hand_qpos[13] += action_hand[9]
        self.target_hand_qpos[14] += action_hand[10]
        self.target_hand_qpos[15] += action_hand[11]
        # Other Three Fingers
        self.target_hand_qpos[0] += action_hand[0]
        self.target_hand_qpos[2] += action_hand[1]
        self.target_hand_qpos[3] += action_hand[2]
        self.target_hand_qpos[4] += action_hand[3]
        self.target_hand_qpos[6] += action_hand[4]
        self.target_hand_qpos[7] += action_hand[5]
        self.target_hand_qpos[8] += action_hand[6]
        self.target_hand_qpos[10] += action_hand[7]
        self.target_hand_qpos[11] += action_hand[8]
    '''
    def pixel_2_world(self, pixel_x, pixel_y, depth, camera="top"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(DcmmCfg.cam_config["width"], DcmmCfg.cam_config["height"], camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x, 
                                pixel_y, 
                                1]) * (depth)
        
        # Get position relative to camera
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        # Transform to the global frame axis
        pos_c[1] *= -1
        pos_c[1], pos_c[2] = pos_c[2], pos_c[1]
        # Get world position
        pos_w = self.cam_rot_mat @ (pos_c) + self.cam_pos

        return pos_c, pos_w

    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """

        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend

        return near / (1 - depth * (1 - near / far))
        #“把 MuJoCo 给你的 0–1 深度图翻回成以米为单位的实际深度图，后续想生成点云、做避障、算距离都能直接用。”
    def create_camera_data(self, width, height, camera):
        """
        Initializes all camera parameters that only need to be calculated once.
        """

        cam_id = self.model.camera(camera).id#camera是字符串，id是数字串
        # Get field of view
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3)) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id] + self.data.body("base_link").xpos - self.data.body("arm_base").xpos
        self.cam_init = True
        #这段 create_camera_data 就是给前面那个 pixel_2_world 做“准备工作”的——把相
        #机的内参矩阵（K）和外参（R、t）算出来并存起来，只算一次，以后直接用。