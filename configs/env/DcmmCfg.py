import os
import numpy as np
from pathlib import Path

## Define the model path
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# Use Leap Hand
XML_DCMM_LEAP_OBJECT_PATH = "urdf/test.xml"#全部（带手）训练集
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/test.xml"#评估集
XML_ARM_PATH = "urdf/a1arm.xml"#机械臂（不带手）
## Weight Saved Path
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## The distance threshold to change the stage from 'tracking' to 'grasping'
distance_thresh = 0.20#某关键点离物体剩0.25m后将跟踪任务切换为抓取任务

## Define the initial joint positions of the arm and the hand
arm_joints = np.array([
   0.0, 0.3, -0.0, 0, 0, 0 
])

hand_joints = np.array([
    0.037,0.037
])

## Define the reward weights
reward_weights = {
    "r_base_pos": 3,
    "r_ee_pos": 20.0,
    "r_precision": 20.0,
    "r_orient": 0.0,
    "r_touch": {
        'Tracking': 15.0,
        'Catching': 15.0
    },
    "r_constraint": 2.0,
    "r_stability": 20.0,
    "r_ctrl": {
        'base': 0.5,
        'arm': 1.0,
        'hand': 0.2,
    },
    "r_collision": -10.0,
    "r_away":0.2,
    "r_far":7.0,
    "r_axis_x": 0.3,
    "r_axis_y": 0.3,
    "r_axis_z": 0.2,
    "r_axis_min": 0.4,
    "axis_sigma": 0.06,
}

## Define the camera params for the MujocoRenderer.
cam_config = {
    "name": "top",
    "width": 640,
    "height": 480,
}

## Define the params of the Double Ackerman model.
RangerMiniV2Params = { 
  'wheel_radius': 0.1,                  # in meter //ranger-mini 0.1
  'steer_track': 0.364,                 # in meter (left & right wheel distance) //ranger-mini 0.364
  'wheel_base': 0.494,                   # in meter (front & rear wheel distance) //ranger-mini 0.494
  'max_linear_speed': 1,              # in m/s
  'max_angular_speed': 3.5,             # in rad/s
  'max_speed_cmd': 7.0,                # in rad/s
  'max_steer_angle_ackermann': 0.6981,  # 40 degree
  'max_steer_angle_parallel': 1.570,    # 180 degree
  'max_round_angle': 0.935671,
  'min_turn_radius': 0.47644,
}


## Define IK
ik_config = {
    "solver_type": "QP", 
    "ps": 0.001, 
    "λΣ": 12.5, 
    "ilimit": 100, 
    "ee_tol": 1e-4
}

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
#object_shape = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
object_shape = ["cylinder", "cylinder", "cylinder", "cylinder", "cylinder"]
object_mesh = ["bottle_mesh", "bottle_mesh", "bottle_mesh", "bottle_mesh", "bottle_mesh"]
#object_mesh = ["bottle_mesh", "bread_mesh", "bowl_mesh", "cup_mesh", "winnercup_mesh"]
#object_mesh = ["bottle_mesh"]
# object_size = {
#     "sphere": np.array([[0.015, 0.020]]),#一个数组表示的是某个参数的上下限，球
#     "capsule": np.array([[0.01, 0.015], [0.025, 0.04]]),#半径，半长度
#     "cylinder": np.array([[0.01, 0.015], [0.02, 0.025]]),#圆柱，半径半长度
#     "box": np.array([[0.01, 0.015], [0.01, 0.015], [0.01, 0.015]]),#半长度
#     "ellipsoid": np.array([[0.01, 0.015], [0.01, 0.015], [0.01, 0.015]]),
# }
object_size = {
    "cylinder": np.array([[0.015, 0.016], [0.055, 0.06]]),#一个数组表示的是某个参数的上下限，球
    "cylinder": np.array([[0.015, 0.016], [0.055, 0.06]]),#半径，半长度
    "cylinder": np.array([[0.015, 0.016], [0.055, 0.06]]),#圆柱，半径半长度
    "cylinder": np.array([[0.015, 0.016], [0.055, 0.06]]),#半长度
    "cylinder": np.array([[0.015, 0.016], [0.055, 0.06]]),
}
object_mass = np.array([0.035, 0.075])
object_damping = np.array([5e-3, 2e-2])
object_static = np.array([0.25, 0.75])
## Observation Noise
k_obs_base = 0.01
k_obs_arm = 0.001
k_obs_object = 0.01
k_obs_hand = 0.01
## Actions Noise
k_act = 0.025
## Action Delay
act_delay = {
    'base': [1,],
    'arm': [1,],
    'hand': [1,],
}

## Define PID params for wheel drive and steering. 
# driving
Kp_drive = 5
Ki_drive = 1e-3
Kd_drive = 1e-1
llim_drive = -200
ulim_drive = 200
# steering
Kp_steer = 50.0
Ki_steer = 2.5
Kd_steer = 7.5
llim_steer = -50
ulim_steer = 50

## Define PID params for the arm and hand. 
Kp_arm = np.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([40.0, 40.0, 40.0, 5.0, 10.0, 1])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])

Kp_hand = np.array([4e-1, 1e-2,])
Ki_hand = 1e-2
Kd_hand = np.array([3e-2, 1e-3,])
llim_hand = -5.0
ulim_hand = 5.0
hand_mask = np.array([1, 1])#有的是1，有的是0，这是为了固定一些关节