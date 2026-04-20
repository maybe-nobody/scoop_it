import os, sys
sys.path.append(os.path.abspath('../../'))
import configs.env.DcmmCfg as DcmmCfg
import math
import numpy as np

def Damper(value, min, max):#小于最小值返回最小值，大于最大值返回最大值
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return value

def IKBase(v_lin_x, v_lin_y, v_yaw = 0.0):#这几个都是基于底盘坐标系的
    """
    Calculate the inverse kinematics for a 4 wheel drive mobile base.

    Inputs:
        v_lin_x: linear x velocity of the mobile base (base_link frame)
        v_lin_y: linear y velocity of the mobile base (base_link frame)
        v_yaw: angular velocity of the mobile base (base_link frame)
        data: mujoco data object
    Outputs:
        data: updated mujoco data object
    """
    if math.fabs(v_lin_y) < 0.01: v_lin_y = 0.0#fabs是绝对值的意思，如果绝对值小于0.01，那么直接取成0
    if math.fabs(v_lin_x) < 0.01: v_lin_x = 0.0
    if math.fabs(v_yaw) < 0.01: v_yaw = 0.0
    if math.fabs(v_lin_y) < 0.01 and math.fabs(v_lin_x) < 0.01 and math.fabs(v_yaw) < 0.01:
        return np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])#return到调用IKbase的地方
    sign = math.copysign(1.0, v_lin_y)#返回的是一个大小等于第一个数，符号和第二个数相同的数
    if math.fabs(v_lin_x) > 0.01:
        ## PARALLEL MOTION MODE
        steer_cmd = -math.atan(v_lin_x / (v_lin_y+1e-5))#计算轮子应该转到的方向角，后面要加的这个东西是防止vy等于0，不能除以0
        steer_cmd = Damper(steer_cmd, -DcmmCfg.RangerMiniV2Params['max_steer_angle_parallel'], 
                           DcmmCfg.RangerMiniV2Params['max_steer_angle_parallel'])#最大最小的限制
        vel_cmd = sign*math.hypot(v_lin_y, v_lin_x)/DcmmCfg.RangerMiniV2Params['wheel_radius']#math.hypot(a, b) = sqrt(a*a + b*b)算出来的是底盘的实际速度大小，只看大小不看方向，现在是线速度，除以半径得到角速度
        return np.array([steer_cmd, steer_cmd, steer_cmd, steer_cmd]), np.array([vel_cmd, vel_cmd, vel_cmd, vel_cmd])
    else:
        if v_yaw == 0:#不转弯
            radius = np.inf#转弯半径就是无穷大(直线行驶)
        else:
            radius = abs(v_lin_y / v_yaw)#v_yaw是角速度，转弯半径。
        vel_fl = sign * math.hypot(v_lin_y - v_yaw*DcmmCfg.RangerMiniV2Params['steer_track']/2, #steer_track左右轮之间的距离，除以2就是中心到左右轮之间的距离，用这个距离乘以角速度就是轮子由于转弯比中心的速度多(少)了多少
                            v_yaw*DcmmCfg.RangerMiniV2Params['wheel_base']/2)/DcmmCfg.RangerMiniV2Params['wheel_radius']
        vel_fr = sign * math.hypot(v_lin_y + v_yaw*DcmmCfg.RangerMiniV2Params['steer_track']/2,
                                v_yaw*DcmmCfg.RangerMiniV2Params['wheel_base']/2)/DcmmCfg.RangerMiniV2Params['wheel_radius']
        vel_rl = sign * math.hypot(v_lin_y - v_yaw*DcmmCfg.RangerMiniV2Params['steer_track']/2,
                                v_yaw*DcmmCfg.RangerMiniV2Params['wheel_base']/2)/DcmmCfg.RangerMiniV2Params['wheel_radius']
        vel_rr = sign * math.hypot(v_lin_y + v_yaw*DcmmCfg.RangerMiniV2Params['steer_track']/2,
                                v_yaw*DcmmCfg.RangerMiniV2Params['wheel_base']/2)/DcmmCfg.RangerMiniV2Params['wheel_radius']
        
        if math.fabs(radius) < DcmmCfg.RangerMiniV2Params['min_turn_radius']:
            ## SPIN MOTION MODE
            fl_steering = math.copysign(math.pi/2, v_yaw)
            fr_steering = math.copysign(math.pi/2, v_yaw)
            rl_steering = -fl_steering
            rr_steering = -fr_steering
        else:
            ## ACKERMAN MOTION MODE
            fl_steering = math.atan(v_yaw*DcmmCfg.RangerMiniV2Params['wheel_base'] / 
                                (2.0*v_lin_y - v_yaw*DcmmCfg.RangerMiniV2Params['steer_track']))
            fr_steering = math.atan(v_yaw*DcmmCfg.RangerMiniV2Params['wheel_base'] / 
                                    (2.0*v_lin_y + v_yaw*DcmmCfg.RangerMiniV2Params['steer_track']))
            rl_steering = -fl_steering
            rr_steering = -fr_steering

        ## Update the joint velocity and position.
        steer_ang = np.array([fl_steering, fr_steering, rl_steering, rr_steering])
        drive_vel = np.array([vel_fl, vel_fr, vel_rl, vel_rr])

        return steer_ang, drive_vel
