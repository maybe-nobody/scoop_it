# ik_arm_new.py
# QP-based IK for a 6-DoF arm in MuJoCo, with a "tracking" mode that is stable when the target changes every control tick.
#
# Public API (compatible with your current call site):
#   IKArmQP.solve(model, data, Tep, q_init) -> (q_sol, success, iters, cost, reached, solve_time)
#
# Key change vs your previous version:
#   - mode="track" (default): does ONE damped resolved-rate QP step per call (good for moving targets).
#   - mode="reach": runs multiple iterations to converge to a fixed target (classic IK).
#
# Notes:
#   - This solver assumes Tep is in the SAME world frame as data.body(ee_body_name).xpos/xquat.
#   - MuJoCo uses wxyz for xquat; SciPy uses xyzw. Keep that consistent when building Tep.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import mujoco
def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)

def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

def _clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _rot_log(R: np.ndarray) -> np.ndarray:

    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        w = np.array([R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0],
                      R[1, 0] - R[0, 1]], dtype=float) * 0.5
        return w

    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=float) * theta


def solve_box_qp_pgd(H: np.ndarray,
                     g: np.ndarray,
                     lb: np.ndarray,
                     ub: np.ndarray,
                     x0: Optional[np.ndarray] = None,
                     iters: int = 50,
                     tol: float = 1e-7) -> np.ndarray:

    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float).reshape(-1)
    n = g.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).reshape(n).copy()

    try:
        L = float(np.linalg.norm(H, 2))
    except Exception:
        L = float(np.linalg.norm(H))
    if not np.isfinite(L) or L <= 1e-12:
        L = 1.0
    step = 1.0 / L

    x = _clamp(x, lb, ub)
    for _ in range(int(iters)):
        grad = H @ x + g
        x_new = _clamp(x - step * grad, lb, ub)
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


@dataclass
class IKDebug:
    pos_err: np.ndarray
    rot_err: np.ndarray
    J: np.ndarray
    dq: np.ndarray
    cost: float


class IKArmQP:
    def __init__(
        self,
        ee_body_name: str = "arm_seg6",
        joint_names=None,  # list of 6 joint names in order (recommended)
        w_pos: float = 1.0,
        w_rot: float = 0.35,
        damping: float = 1e-3,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        max_iters: int = 60,
        max_dq: float = 0.25,
        pgd_iters: int = 40,
        pgd_tol: float = 1e-6,
        line_search: bool = True,
        mode: str = "track",          # "track" or "reach"
        kp_pos: float = 0.6,
        kp_rot: float = 0.4,
        target_filter_alpha: float = 1.0,
        debug: bool = False,
    ):
        self.ee_body_name = ee_body_name
        self.joint_names = joint_names

        self.w_pos = float(w_pos)
        self.w_rot = float(w_rot)
        self.damping = float(damping)

        self.tol_pos = float(tol_pos)
        self.tol_rot = float(tol_rot)
        self.max_iters = int(max_iters)
        self.max_dq = float(max_dq)

        self.pgd_iters = int(pgd_iters)
        self.pgd_tol = float(pgd_tol)

        self.line_search = bool(line_search)

        self.mode = str(mode).lower()
        if self.mode not in ("track", "reach"):
            raise ValueError("IKArmQP.mode must be 'track' or 'reach'.")

        self.kp_pos = float(kp_pos)
        self.kp_rot = float(kp_rot)

        self.target_filter_alpha = float(np.clip(float(target_filter_alpha), 0.0, 1.0))

        self.debug = bool(debug)
        self.last_debug: Optional[IKDebug] = None

        # lazy init cache
        self._cached_model_id = None
        self._ee_bid = None
        self._jnt_ids = None
        self._qpos_adrs = None
        self._dof_adrs = None
        self._jlows = None
        self._jhighs = None

        self._Tep_filt: Optional[np.ndarray] = None

    def _lazy_init(self, model: mujoco.MjModel):
        mid = id(model)
        if self._cached_model_id == mid:
            return

        self._ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)# EE body 在 model.body[] 数组里的索引
        if self._ee_bid < 0:
            raise ValueError(f"[IKArmQP] body not found: {self.ee_body_name}")

        if self.joint_names is None:
            guess = [f"arm_joint{i}" for i in range(1, 7)]
            jids = []
            ok = True
            for n in guess:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if jid < 0:
                    ok = False
                    break
                jids.append(jid)
            if ok:
                jnt_ids = jids
            else:
                jnt_ids = list(range(min(6, model.njnt)))#把给的关节名字转换成id
                if len(jnt_ids) != 6:
                    raise ValueError("[IKArmQP] joint_names is None and model has <6 joints; please pass joint_names.")
        else:
            if len(self.joint_names) != 6:
                raise ValueError("[IKArmQP] joint_names must be a list of 6 joint names.")
            jnt_ids = []
            for name in self.joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"[IKArmQP] joint not found: {name}")
                jnt_ids.append(jid)

        self._jnt_ids = np.array(jnt_ids, dtype=int)

        qpos_adrs = []
        dof_adrs = []
        jlows = []
        jhighs = []
        for jid in self._jnt_ids:
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            qpos_adrs.append(qadr)
            dof_adrs.append(dadr)

            if int(model.jnt_limited[jid]) == 1:
                rng = model.jnt_range[jid]
                jlows.append(float(rng[0]))
                jhighs.append(float(rng[1]))
            else:
                jlows.append(-np.inf)
                jhighs.append(np.inf)

        self._qpos_adrs = np.array(qpos_adrs, dtype=int)
        self._dof_adrs = np.array(dof_adrs, dtype=int)
        self._jlows = np.array(jlows, dtype=float)
        self._jhighs = np.array(jhighs, dtype=float)

        self._cached_model_id = mid
        self._Tep_filt = None

    def get_q(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        self._lazy_init(model)
        return np.array(data.qpos[self._qpos_adrs], dtype=float)

    def set_q(self, model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray):
        self._lazy_init(model)
        data.qpos[self._qpos_adrs] = np.asarray(q, dtype=float).reshape(6)

    def _fk_Te(self, data: mujoco.MjData) -> np.ndarray:
        p = data.body(self._ee_bid).xpos.copy()
        q = data.body(self._ee_bid).xquat.copy()  # wxyz
        Rm = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(Rm, q)
        Te = np.eye(4, dtype=float)
        Te[:3, :3] = Rm.reshape(3, 3)
        Te[:3, 3] = p
        return Te

    def _jacobian_6x6(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        nv = model.nv
        jacp = np.zeros((3, nv), dtype=float)
        jacr = np.zeros((3, nv), dtype=float)
        mujoco.mj_jacBody(model, data, jacp, jacr, self._ee_bid)
        cols = self._dof_adrs
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def _pose_error(self, Te: np.ndarray, Tep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        e_pos = Tep[:3, 3] - Te[:3, 3]
        R_err = Tep[:3, :3] @ Te[:3, :3].T
        e_rot = _rot_log(R_err)
        return e_pos, e_rot

    def solve(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q_init: Optional[np.ndarray] = None):
        t0 = time.perf_counter()
        self._lazy_init(model)

        Tep = np.asarray(Tep, dtype=float).reshape(4, 4)

        # optional target filter
        if self._Tep_filt is None or self.target_filter_alpha >= 0.999:
            Tep_f = Tep
            if self._Tep_filt is None:
                self._Tep_filt = Tep.copy()
        else:
            a = self.target_filter_alpha
            self._Tep_filt[:3, 3] = (1 - a) * self._Tep_filt[:3, 3] + a * Tep[:3, 3]
            self._Tep_filt[:3, :3] = (1 - a) * self._Tep_filt[:3, :3] + a * Tep[:3, :3]
            U, _, Vt = np.linalg.svd(self._Tep_filt[:3, :3])
            self._Tep_filt[:3, :3] = U @ Vt
            Tep_f = self._Tep_filt.copy()

        if q_init is None:
            q = self.get_q(model, data)
        else:
            q = np.asarray(q_init, dtype=float).reshape(6).copy()
        q = _clamp(q, self._jlows, self._jhighs)

        if self.mode == "track":
            q_sol, success, cost = self._solve_track_step(model, data, Tep_f, q)
            it_used = 1
            reached = success
        else:
            q_sol, success, it_used, cost, reached = self._solve_reach(model, data, Tep_f, q)

        solve_time = time.perf_counter() - t0
        return q_sol, bool(success), int(it_used), float(cost), bool(reached), float(solve_time)

    def _solve_track_step(self, model, data, Tep, q):
        self.set_q(model, data, q)
        mujoco.mj_fwdPosition(model, data)

        Te = self._fk_Te(data)
        e_pos, e_rot = self._pose_error(Te, Tep)

        pos_norm = float(np.linalg.norm(e_pos))
        rot_norm = float(np.linalg.norm(e_rot))
        cost = pos_norm + rot_norm
        success = (pos_norm <= self.tol_pos) and (rot_norm <= self.tol_rot)

        J = self._jacobian_6x6(model, data)
        if np.linalg.norm(J) < 1e-12 or not np.isfinite(J).all():
            if self.debug:
                self.last_debug = IKDebug(e_pos, e_rot, J, np.zeros(6), cost)
            return q, success, cost

        e_task = np.concatenate([self.kp_pos * e_pos, self.kp_rot * e_rot])
        W = np.diag([self.w_pos, self.w_pos, self.w_pos, self.w_rot, self.w_rot, self.w_rot])
        WJ = W @ J
        We = W @ e_task

        H = WJ.T @ WJ + self.damping * np.eye(6)
        g = -(WJ.T @ We)

        lb = np.maximum(-self.max_dq * np.ones(6), self._jlows - q)
        ub = np.minimum(+self.max_dq * np.ones(6), self._jhighs - q)

        dq = solve_box_qp_pgd(H, g, lb, ub, iters=self.pgd_iters, tol=self.pgd_tol)
        q_next = _clamp(q + dq, self._jlows, self._jhighs)

        if self.debug:
            self.last_debug = IKDebug(e_pos, e_rot, J, dq, cost)

        return q_next, success, cost

    def _solve_reach(self, model, data, Tep, q):
        reached = False
        final_cost = np.inf
        it_used = 0
        success = False

        for it in range(self.max_iters):
            it_used = it + 1
            self.set_q(model, data, q)
            mujoco.mj_fwdPosition(model, data)

            Te = self._fk_Te(data)
            e_pos, e_rot = self._pose_error(Te, Tep)

            pos_norm = float(np.linalg.norm(e_pos))
            rot_norm = float(np.linalg.norm(e_rot))
            final_cost = pos_norm + rot_norm

            if (pos_norm <= self.tol_pos) and (rot_norm <= self.tol_rot):
                reached = True
                success = True
                break

            J = self._jacobian_6x6(model, data)
            if np.linalg.norm(J) < 1e-12 or not np.isfinite(J).all():
                success = False
                break

            e_task = np.concatenate([self.kp_pos * e_pos, self.kp_rot * e_rot])
            W = np.diag([self.w_pos, self.w_pos, self.w_pos, self.w_rot, self.w_rot, self.w_rot])
            WJ = W @ J
            We = W @ e_task

            H = WJ.T @ WJ + self.damping * np.eye(6)
            g = -(WJ.T @ We)

            lb = np.maximum(-self.max_dq * np.ones(6), self._jlows - q)
            ub = np.minimum(+self.max_dq * np.ones(6), self._jhighs - q)

            dq = solve_box_qp_pgd(H, g, lb, ub, iters=self.pgd_iters, tol=self.pgd_tol)

            if self.line_search:
                best_q = q.copy()
                best_cost = final_cost
                step = 1.0
                for _ in range(6):
                    q_try = _clamp(q + step * dq, self._jlows, self._jhighs)
                    self.set_q(model, data, q_try)
                    mujoco.mj_fwdPosition(model, data)
                    Te2 = self._fk_Te(data)
                    epos2, erot2 = self._pose_error(Te2, Tep)
                    c2 = float(np.linalg.norm(epos2)) + float(np.linalg.norm(erot2))
                    if c2 < best_cost:
                        best_cost = c2
                        best_q = q_try
                    step *= 0.5
                q = best_q
            else:
                q = _clamp(q + dq, self._jlows, self._jhighs)

            if self.debug:
                self.last_debug = IKDebug(e_pos, e_rot, J, dq, final_cost)

        return vfq, success, it_used, final_cost, reached