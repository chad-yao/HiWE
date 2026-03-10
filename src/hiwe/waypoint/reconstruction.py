"""Trajectory reconstruction and error metrics for waypoint extraction."""

import math
import numpy as np
from scipy.spatial.transform import Rotation

PI = np.pi
EPS = np.finfo(float).eps * 4.0

_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def linear_interpolation(p1, p2, t):
    """Compute the linear interpolation between two 3D points."""
    return p1 + t * (p2 - p1)


def unit_vector(data, axis=None, out=None):
    """Returns ndarray normalized by length, i.e. euclidean norm, along axis."""
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def gripper_distance(point, line_start, line_end):
    dis = []
    is_list = isinstance(point, list)
    if is_list:
        for i in range(len(point)):
            d1 = abs(point[i] - line_start)
            d2 = abs(point[i] - line_end)
            dis.append(min(d1, d2))
    else:
        dis.append(min(abs(point - line_start), abs(point - line_end)))
    return np.mean(dis)


def point_line_distance(point, line_start, line_end):
    """Compute the shortest distance between a 3D point and a line segment."""
    line_vector = line_end - line_start
    point_vector = point - line_start
    t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    t = np.clip(t, 0, 1)
    projection = linear_interpolation(line_start, line_end, t)
    return np.linalg.norm(point - projection)


def point_quat_distance(point, quat_start, quat_end, t, total):
    pred_point = quat_slerp(quat_start, quat_end, fraction=t / total)
    err_quat = (
        Rotation.from_quat(pred_point) * Rotation.from_quat(point).inv()
    ).magnitude()
    return err_quat


def dp_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """Compute the geometric trajectory error from the waypoints (DP format)."""
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    gt_pos = [p["robot0_eef_pos"] for p in gt_states]
    gt_quat = [p["robot0_eef_quat"] for p in gt_states]

    keypoints_pos = [gt_pos[k] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]

    state_err = []
    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]
        start_keypoint_quat = keypoints_quat[i]
        end_keypoint_quat = keypoints_quat[i + 1]

        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_pos[start_idx:end_idx]
        segment_points_quat = gt_quat[start_idx:end_idx]

        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            rot_err = point_quat_distance(
                segment_points_quat[i],
                start_keypoint_quat,
                end_keypoint_quat,
                i,
                len(segment_points_quat),
            )
            state_err.append(pos_err + rot_err)

    return np.max(state_err), np.mean(state_err)


def act_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """Compute the geometric trajectory error from the waypoints (ACT format)."""
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    gt_pos = [p[:7] for p in gt_states]
    gt_quat = []
    gt_grip = []

    keypoints_pos = [actions[k, :7] for k in waypoints]
    keypoints_quat = []
    keypoints_grip = []
    state_err = []

    n_segments = len(waypoints) - 1
    if_pos = True if len(gt_pos) > 0 else False
    if_quat = True if len(gt_quat) > 0 else False
    if_grip = True if len(gt_grip) > 0 else False
    for i in range(n_segments):
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        if if_pos:
            start_keypoint_pos = keypoints_pos[i]
            end_keypoint_pos = keypoints_pos[i + 1]
            segment_points_pos = gt_pos[start_idx:end_idx]
        if if_quat:
            start_keypoint_quat = keypoints_quat[i]
            end_keypoint_quat = keypoints_quat[i + 1]
            segment_points_quat = gt_quat[start_idx:end_idx]
        if if_grip:
            start_keypoint_grip = keypoints_grip[i]
            end_keypoint_grip = keypoints_grip[i + 1]
            segment_points_grip = gt_grip[start_idx:end_idx]

        for i in range(len(segment_points_pos)):
            pos_err = (
                point_line_distance(
                    segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
                )
                if if_pos
                else 0
            )
            grip_err = (
                gripper_distance(
                    segment_points_grip[i], start_keypoint_grip, end_keypoint_grip
                )
                if if_grip
                else 0
            )
            rot_err = (
                point_quat_distance(
                    segment_points_quat[i],
                    start_keypoint_quat,
                    end_keypoint_quat,
                    i,
                    len(segment_points_quat),
                )
                if if_quat
                else 0
            )
            state_err.append(pos_err + rot_err + grip_err)

    return np.max(state_err), np.mean(state_err)


def total_state_err(err_dict):
    return err_dict["err_pos"] + err_dict["err_quat"]


def total_traj_err(err_list):
    return np.max(err_list)


def compute_state_error(gt_state, pred_state):
    """Compute the state error between the ground truth and predicted states."""
    err_pos = np.linalg.norm(
        gt_state["robot0_eef_pos"] - pred_state["robot0_eef_pos"]
    )
    err_quat = (
        Rotation.from_quat(gt_state["robot0_eef_quat"])
        * Rotation.from_quat(pred_state["robot0_eef_quat"]).inv()
    ).magnitude()
    err_joint_pos = np.linalg.norm(
        gt_state["robot0_joint_pos"] - pred_state["robot0_joint_pos"]
    )
    state_err = dict(
        err_pos=err_pos, err_quat=err_quat, err_joint_pos=err_joint_pos
    )
    return state_err


def dynamic_time_warping(seq1, seq2, idx1=0, idx2=0, memo=None):
    if memo is None:
        memo = {}

    if idx1 == len(seq1):
        return 0, []

    if idx2 == len(seq2):
        return float("inf"), []

    if (idx1, idx2) in memo:
        return memo[(idx1, idx2)]

    distance_with_current = total_state_err(
        compute_state_error(seq1[idx1], seq2[idx2])
    )
    error_with_current, subseq_with_current = dynamic_time_warping(
        seq1, seq2, idx1 + 1, idx2 + 1, memo
    )
    error_with_current += distance_with_current

    error_without_current, subseq_without_current = dynamic_time_warping(
        seq1, seq2, idx1, idx2 + 1, memo
    )

    if error_with_current < error_without_current:
        memo[(idx1, idx2)] = error_with_current, [idx2] + subseq_with_current
    else:
        memo[(idx1, idx2)] = error_without_current, subseq_without_current

    return memo[(idx1, idx2)]
