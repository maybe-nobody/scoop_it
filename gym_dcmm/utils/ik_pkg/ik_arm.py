"""
Implement the Inverse Kinematics (IK) for the XArm6.

Reference Link: https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
                https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
                https://github.com/petercorke/robotics-toolbox-python
"""
import os, sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('./gym_dcmm/'))
import mujoco
import numpy as np
from abc import ABC, abstractmethod
import time
import qpsolvers as qp
from functools import wraps
from utils.util import calculate_arm_Te, angle_axis_python

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

DEBUG_IK = True

class IK(ABC):
    """
    An abstract super class which provides basic functionality to perform numerical inverse
    kinematics (IK). Superclasses can inherit this class and implement the solve method.
    """

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 100,#单次搜索的最大迭代次数在放弃之前，最多尝试多少步
        # slimit: int = 100,
        tol: float = 2e-3,# 误差阈值 E < tol 视为收敛
        we: np.ndarray = np.ones(6),# 6×6 对角矩阵，对末端 6 自由度误差加权
        # problems: int = 1000,
        reject_jl: bool = True,# 是否拒绝超关节范围的解
        ps: float=0.1,# 距离关节限位多近开始惩罚
        λΣ: float=0.0,#让关节尽量远离限位（越大越“怕”撞限位）
        λm: float=0.0, #让机械臂保持“灵活”姿态（越大越追求“能伸能转”）
        copy: bool = False,
        debug: bool = False
    ):
        """
        name: The name of the IK algorithm
        ilimit: How many iterations are allowed within a search before a new search is started
        # slimit: How many searches are allowed before being deemed unsuccessful
        tol: Maximum allowed residual error E
        we: A 6 vector which assigns weights to Cartesian degrees-of-freedom
        # problems: Total number of IK problems within the experiment
        reject_jl: Reject solutions with joint limit violations
        ps: The minimum angle/distance (in radians or metres) in which the joint is allowed to approach to its limit
        λΣ: The gain for joint limit avoidance. Setting to 0.0 will remove this completely from the solution
        λm: The gain for maximisation. Setting to 0.0 will remove this completely from the solution (always 0.0 for now)
        """

        # Solver parameters
        self.name = name
        # self.slimit = slimit
        self.ilimit = ilimit
        self.tol = tol
        self.We = np.diag(we)
        self.reject_jl = reject_jl
        self.λΣ = λΣ
        self.λm = λm
        self.ps = ps
        self.copy = copy
        self.debug = debug

    def solve(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q0: np.ndarray):
        """
        This method will attempt to solve the IK problem and obtain joint coordinates
        which result the the end-effector pose Tep.

        The method returns a tuple:
        q: The joint coordinates of the solution (ndarray). Note that these will not
            be valid if failed to find a solution
        success: True if a solution was found (boolean)
        iterations: The number of iterations it took to find the solution (int)
        # searches: The number of searches it took to find the solution (int)
        residual: The residual error of the solution (float)
        jl_valid: True if joint coordinates q are within the joint limits
        total_t: The total time spent within the step method
        """

        # Iteration count
        if DEBUG_IK or self.debug:#两个开关，有一个是true的时候就打印
            print("=== IK SOLVE START ===")
            print("Initial q0:", q0)
            print("Target Pose Tep:\n", Tep)

        i = 0
        total_i = 0#总迭代数
        total_t = 0.0#累计在 step() 中花费的总时间
        q = np.zeros(model.nv)#nv和nu都是一个数字q：当前迭代使用的关节角，初始化为 q0，后面每次 step() 会更新 q。
        q[:] = q0[:]
        error = -1#当前误差 E（初始化为 -1 表示还没算）
        q_solved = np.zeros(model.nv)#记录“目前为止最好的 q”（即使最后失败，也会保留最后一次迭代的 q）
        # print("initial q: ", q)
        # print("initial q0: ", q0)
        while i <= self.ilimit:#在放弃之前，最多尝试多少步
            i += 1

            # Attempt a step
            try:
                t, E, q = self.step(model, data, Tep, q, i)#Tep: 4×4 目标末端位姿（，这里会用IK_QP类中的step函数进行计算
                if DEBUG_IK or self.debug:
                    print(f"[IK Step {i}] Error E = {E}")
                    print("Current q:", q)

                error = E
                q_solved[:] = q[:]

                # Acclumulate total time
                total_t += t
            except np.linalg.LinAlgError:#如果执行过程中抛出了 np.linalg.LinAlgError 这个异常
                # Abandon search and try again
                print("break LinAlgError")
                break

            # Check if we have arrived
            if E < self.tol:
                # Wrap q to be within +- 180 deg
                # If your robot has larger than 180 deg range on a joint
                # this line should be modified in incorporate the extra range
                # q = (q + np.pi) % (2 * np.pi) - np.pi

                # Check if we have violated joint limits
                jl_valid = self.check_jl(model, q)#检查关节是否违反关节上下限

                if not jl_valid and self.reject_jl:
                    # Abandon search and try again
                    # print("break limits!!!!!!!!!!!!!!!!!!!!!")
                    if DEBUG_IK or self.debug: print("break limits!!!!!!!!!!!!!!!!!!!!!")
                    continue
                else:
                    if DEBUG_IK or self.debug: 
                        print("q_solved: {}, error: {}".format(q_solved, error))
                        print("iteration: {}, total_t: {}".format(i, total_t))
                        print("solved ik!! \n")
                    if DEBUG_IK or self.debug:
                        print("=== IK SOLVED SUCCESS ===")
                        print("Final q:", q)
                        print("Final error:", E)
                        print("Iterations:", i)

                    return q, True, total_i + i, E, jl_valid, total_t

        # Note: If we make it here, then we have failed because of the iteration limit or the joint limits
        # print("q_solved: {}, error: {}".format(q_solved, error))
        # print("iteration: {}, total_t: {}".format(i, total_t))
        # print("failed ik!! \n")
        if DEBUG_IK or self.debug: 
            print("q_solved: {}, error: {}".format(q_solved, error))
            print("iteration: {}, total_t: {}".format(i, total_t))
            print("failed ik!! \n")
        # Return the initial joint position (not the last solution we get)
        # data.qpos[:] = q0[:]
        if DEBUG_IK or self.debug:
            print("=== IK FAILED ===")
            print("Initial q0:", q0)
            print("Last q:", q_solved)
            print("Final error E:", error)
            print("Iteration:", i)
            print("Target Pose Tep:\n", Tep)
            print("===================")

        return q, False, np.nan, E, np.nan, np.nan

    def error(self, Te: np.ndarray, Tep: np.ndarray):
        """
        Calculates the engle axis error between current end-effector pose Te and
        the desired end-effector pose Tep. Also calulates the quadratic error E
        which is weighted by the diagonal matrix We.

        Returns a tuple:
        e: angle-axis error (ndarray in R^6)
        E: The quadratic error weighted by We
        """
        # e = rtb.angle_axis(Te, Tep)
        e = angle_axis_python(Te, Tep)
        E = 0.5 * e @ self.We @ e

        return e, E

    def check_jl(self, model: mujoco.MjModel, q: np.ndarray):
        """
        Checks if the joints are within their respective limits

        Returns a True if joints within feasible limits otherwise False
        """

        # Loop through the joints in the ETS
        for i in range(model.nv):

            # Get the corresponding joint limits
            ql0 = model.joint(i).range[0]
            ql1 = model.joint(i).range[1]
            # print("ql0: ", ql0)
            # print("ql1: ", ql1)

            # Check if q exceeds the limits
            if q[i] < ql0 or q[i] > ql1:
                # print("i: ", i)
                # print("q[i]: ", q[i])
                # print("ql0: ", ql0)
                # print("ql1: ", ql1)
                return False

        # If we make it here, all the joints are fine
        return True

    @abstractmethod
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray):
        """
        Superclasses will implement this method to perform a step of the implemented
        IK algorithm
        """
        pass

def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        t_start = time.time()
        E, q = func(*args, **kw)
        t_end = time.time()
        t = t_end - t_start
        return t, E, q
    return wrap

class QP(IK):
    def __init__(self, name="QP", λj=1.0, λs=1.0, **kwargs):
        super().__init__(name, **kwargs)

        self.name = f"QP (λj={λj}, λs={λs})"
        self.λj = λj
        self.λs = λs

        if self.λΣ > 0.0:
            self.name += ' Σ'

        if self.λm > 0.0:
            self.name += ' Jm'
        # print("self.ilimit: ", self.ilimit)

    @timing
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray, i: int):
        # Calculate forward kinematics (Te)
        # mujoco.mj_resetData(model, data)
        data.qpos[:] = q[:]#把当前这一步的关节角 q 写进 MuJoCo 的状态里
        # Do not use mj_kinematics, it does more than foward the position kinematics!
        # mujoco.mj_kinematics(model, data)
        mujoco.mj_fwdPosition(model, data)#只做位置的前向计算
        Te = calculate_arm_Te(data.body("arm_seg6").xpos, data.body("arm_seg6").xquat)#Te = 当前末端实际位姿，把 (xpos, xquat) 拼成一个 4×4 的齐次变换矩阵 Te：
        # print("Tep: ", Tep)
        # print("Te: ", Te)
        # exit(1)
        # Calculate the error
        e, E = self.error(Te, Tep)#计算误差 e 和误差标量 E
        if E < self.tol and i <= 1:#说明误差很小了
            # print("NO NEED to calculate IK!!!!!!!!!!!!!")
            # data.qpos[:] = q[:]
            return E, q
        
        # Calculate the Jacobian
        jacp = np.zeros((3, model.nv))#3*nv的矩阵，线速度雅可比 jacp（位置）
        jacr = np.zeros((3, model.nv))#角速度雅可比 jacr（姿态）
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, model.body("arm_seg6").id)#MuJoCo 内置函数，计算指定刚体（这里是 arm_seg6）的雅可比矩阵：有两个输出，就分别对应上面两个
        J = np.concatenate((jacp, jacr), axis=0)#之后 QP 就是基于这个 J 来算 ∆q 的。
        # print("J: \n", J)
        # Quadratic component of objective function
        Q = np.eye(model.nv + 6)#eye是单位矩阵

        # Joint velocity component of Q
        Q[: model.nv, : model.nv] *= self.λj#λj 越大，优化更倾向于动得少。

        # Slack component of Q
        Q[model.nv :, model.nv :] = self.λs * (1 / np.sum(np.abs(e))) * np.eye(6)

        # The equality contraints
        Aeq = np.concatenate((J, np.eye(6)), axis=1)
        beq = 2*e.reshape((6,))

        # The inequality constraints for joint limit avoidance
        if self.λΣ > 0.0:#限制 Δq 的方向和大小，让机器人远离关节上/下限，避免打到关节边界
            Ain = np.zeros((model.nv + 6, model.nv + 6))
            bin = np.zeros(model.nv + 6)

            # Form the joint limit velocity damper
            Ain_l = np.zeros((model.nv, model.nv))
            Bin_l = np.zeros(model.nv)

            for i in range(model.nv):
                ql0 = model.joint(i).range[0]
                ql1 = model.joint(i).range[1]
                # Calculate the influence angle/distance (in radians or metres) in null space motion becomes active
                pi = (model.joint(i).range[1] - model.joint(i).range[0])/2

                if ql1 - q[i] <= pi:
                    Bin_l[i] = ((ql1 - q[i]) - self.ps) / (pi - self.ps)
                    Ain_l[i, i] = 1

                if q[i] - ql0 <= pi:
                    Bin_l[i] = -(((ql0 - q[i]) + self.ps) / (pi - self.ps))
                    Ain_l[i, i] = -1

            Ain[: model.nv, : model.nv] = Ain_l
            bin[: model.nv] =  (1.0 / self.λΣ) * Bin_l
        else:
            Ain = None
            bin = None
        
        # TODO: Manipulability maximisation
        # if self.λm > 0.0:
        #     Jm = ets.jacobm(q).reshape((model.nv,))
        #     c = np.concatenate(((1.0 / self.λm) * -Jm, np.zeros(6)))
        # else:
        #     c = np.zeros(model.nv + 6)
        c = np.zeros(model.nv + 6)#线性项 c（目标函数中的“线性部分”）
            
        # print("Q: ", Q)
        # print("c: ", c)
        # print("Ain: ", Ain)
        # print("bin: ", bin)
        # print("Aeq: ", Aeq)
        # print("beq: ", beq)
        xd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=None, ub=None, solver='quadprog')#求出来的优化变量向量
        # print("xd: ", xd)
        # print("xd: ", xd[: 6])
        q += xd[: model.nv]
        # print("q: ", q)
        # data.qpos[:] = q[:]
        return E, q#这就是 QP 逆运动学的一次迭代，由于函数前面有@time，所以其实是三个输出，还会输出一个t时间

def null_Σ(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, ps: float):
    """
    Formulates a relationship between joint limits and the joint velocity.
    When this is projected into the null-space of the differential kinematics
    to attempt to avoid exceeding joint limits

    q: The joint coordinates of the robot
    ps: The minimum angle/distance (in radians or metres) in which the joint is
        allowed to approach to its limit
    pi: The influence angle/distance (in radians or metres) in which the velocity
        damper becomes active

    returns: Σ 
    """

    # Add cost to going in the direction of joint limits, if they are within
    # the influence distance
    Σ = np.zeros((model.nv, 1))

    for i in range(model.nv):
        qi = q[i]
        ql0 = model.joint(i).range[0]
        ql1 = model.joint(i).range[1]
        pi = (model.joint(i).range[1] - model.joint(i).range[0])/2

        if qi - ql0 <= pi:
            Σ[i, 0] = (
                -np.power(((qi - ql0) - pi), 2) / np.power((ps - pi), 2)
            )
        if ql1 - qi <= pi:
            Σ[i, 0] = (
                np.power(((ql1 - qi) - pi), 2) / np.power((ps - pi), 2)
            )

    return -Σ

def calc_qnull(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q: np.ndarray,
        J: np.ndarray,
        λΣ: float,
        # λm: float,
        ps: float,
    ):
    """
    Calculates the desired null-space motion according to the gains λΣ and λm.
    This is a helper method that is used within the `step` method of an IK solver

    Returns qnull: the desired null-space motion
    """

    qnull_grad = np.zeros(model.nv)
    qnull = np.zeros(model.nv)

    # Add the joint limit avoidance if the gain is above 0
    if λΣ > 0:
        Σ = null_Σ(model, data, q, ps)
        qnull_grad += (1.0 / λΣ * Σ).flatten()

    # TODO: Add the manipulability maximisation if the gain is above 0
    # if λm > 0:
    #     Jm = ets.jacobm(q)
    #     qnull_grad += (1.0 / λm * Jm).flatten()

    # Calculate the null-space motion
    if λΣ > 0.0:
        null_space = (np.eye(model.nv) - np.linalg.pinv(J) @ J)
        qnull = null_space @ qnull_grad

    return qnull.flatten()

class LM_Chan(IK):
    def __init__(self, λ=1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.name = f"LM (Chan λ={λ})"
        self.λ = λ

        if self.λΣ > 0.0:
            self.name += ' Σ'

        if self.λm > 0.0:
            self.name += ' Jm'

    @timing
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray):
        # Calculate forward kinematics (Te)
        mujoco.mj_resetData(model, data)
        data.qpos = q
        # Do not use mj_kinematics, it does more than foward the position kinematics!
        # mujoco.mj_kinematics(model, data)
        mujoco.mj_fwdPosition(model, data)
        Te = np.eye(4)
        Te[:3,3] = data.body("arm_seg6").xpos
        res = np.zeros(9)
        mujoco.mju_quat2Mat(res, data.body("arm_seg6").xquat)
        Te[:3,:3] = res.reshape((3,3))
        # print(Te)

        # Calculate the error
        e, E = self.error(Te, Tep)
        # Calculate the Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, model.body("arm_seg6").id)
        J = np.concatenate((jacp, jacr), axis=0)
        g = J.T @ self.We @ e

        Wn = self.λ * E * np.eye(model.nv)

        # Null-space motion
        qnull = calc_qnull(model, data, q, J, self.λΣ, self.ps)
        print("qnull: ", qnull)
        q += np.linalg.inv(J.T @ self.We @ J + Wn) @ g + qnull

        return E, q

class IKArm:
    def __init__(self, solver_type='QP', ps=0.001, λΣ=10, λj=0.1, λs=1.0, λ=0.1, tol=2e-3, ilimit=100):
        if solver_type=='QP':
            self.solver = QP(λj=λj, λs=λs, ps=ps, λΣ=λΣ, tol=tol, ilimit=ilimit)
        elif solver_type=='LM_Chan':
            self.solver = LM_Chan(λ=λ, ps=ps, λΣ=λΣ, tol=tol, ilimit=ilimit)
        else:
            raise ValueError("Invalid solver type")
    
    def solve(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q0: np.ndarray):#Tep: 4×4 目标末端位姿（位置 + 姿态）
        #q0: 起始关节角（当前机械臂状态）
        self.q0 = np.zeros(model.nv)
        self.q0[:] = q0[:]
        # print("before self.q0: ", self.q0)
        result_IK = self.solver.solve(model, data, Tep, q0)
        # print("after self.q0: ", self.q0)
        if not result_IK[1]:
            print("Failed result_IK: ", result_IK)
            return self.q0, result_IK[1], result_IK[2], result_IK[3], result_IK[4], result_IK[5]
        return result_IK#(q, success, iterations迭代次数, E最终误差, jl_valid是否在关节范围内, total_t求解时间)

