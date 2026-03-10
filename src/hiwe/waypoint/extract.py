"""Automatic waypoint selection using dynamic programming."""

import copy
import numpy as np

from .reconstruction import act_waypoint_trajectory, dp_waypoint_trajectory


def dp_waypoint_selection(
    env=None,
    actions=None,
    gt_states=None,
    err_threshold=None,
    initial_states=None,
    remove_obj=None,
    pos_only=False,
):
    """Select waypoints to reconstruct trajectory within error threshold."""
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
    num_frames = len(actions)
    initial_waypoints = [num_frames - 1]
    memo = {}

    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = act_waypoint_trajectory if pos_only else dp_waypoint_trajectory
    min_error, mins = func(
        actions, gt_states, list(range(1, num_frames))
    )
    if err_threshold < min_error and mins < min_error / 2:
        return list(range(1, num_frames))
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            waypoints = [
                j - k for j in initial_waypoints if j >= k and j < i
            ] + [i - k]

            total_traj_err, similarity = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
            )

            if total_traj_err < err_threshold and similarity < err_threshold / 2:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    waypoints = list(set(waypoints))
    waypoints.sort()

    return waypoints
