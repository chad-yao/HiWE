"""Extract waypoints from DP dataset using hierarchical key intervals."""

import argparse
import h5py
import numpy as np
from tqdm import tqdm

from hiwe.waypoint import dp_waypoint_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to DP HDF5 dataset",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
    )
    parser.add_argument("--end_idx", type=int, default=199)
    args = parser.parse_args()

    num_waypoints = []
    num_frames = []

    with h5py.File(args.dataset, "r+") as f:
        demos = list(f["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]

        for idx in tqdm(range(args.start_idx, args.end_idx + 1)):
            ep = demos[idx]
            states = f[f"data/{ep}/states"][()]
            initial_states = [
                dict(states=states[i]) for i in range(len(states))
            ]
            for i in range(len(initial_states)):
                initial_states[i]["model"] = f[f"data/{ep}"].attrs[
                    "model_file"
                ]
            traj_len = states.shape[0]

            eef_pos = f[f"data/{ep}/obs/robot0_eef_pos"][()]
            eef_quat = f[f"data/{ep}/obs/robot0_eef_quat"][()]
            joint_pos = f[f"data/{ep}/obs/robot0_joint_pos"][()]
            get_label_data = f[f"data/{ep}/label-test"][()]

            labels = []
            if get_label_data.size > 0:
                current_label = get_label_data[0]
                start_index = 0
                for i in range(1, len(get_label_data)):
                    if get_label_data[i] != current_label:
                        labels.append(
                            [current_label, [start_index, i - 1]]
                        )
                        current_label = get_label_data[i]
                        start_index = i
                labels.append(
                    [
                        current_label,
                        [start_index, len(get_label_data) - 1],
                    ]
                )

            gt_states = [
                dict(
                    robot0_eef_pos=eef_pos[i],
                    robot0_eef_quat=eef_quat[i],
                    robot0_joint_pos=joint_pos[i],
                )
                for i in range(traj_len)
            ]

            try:
                actions = f[f"data/{ep}/actions"][()]
            except KeyError:
                raise NotImplementedError(
                    "No absolute actions found, need to convert first."
                )

            threshold = [0.005, 0.005]
            waypoints = []
            for label, indice in labels:
                adjusted_start, adjusted_end = indice[0], indice[1]
                if adjusted_end <= adjusted_start + 1:
                    continue
                sub_waypoints = dp_waypoint_selection(
                    actions=actions[adjusted_start:adjusted_end],
                    gt_states=gt_states[adjusted_start:adjusted_end],
                    err_threshold=threshold[label],
                    initial_states=initial_states,
                    remove_obj=True,
                )
                waypoints += [adjusted_start + wp for wp in sub_waypoints]
                waypoints += [adjusted_start, adjusted_end]

            waypoints = sorted(set(p for p in waypoints if p != 0))
            num_waypoints.append(len(waypoints))
            num_frames.append(traj_len)

            waypoint_path = f"data/{ep}/waypoints_dp"
            if waypoint_path in f:
                del f[waypoint_path]
            f.create_dataset(waypoint_path, data=waypoints)

    print(
        f"Average waypoints: {np.mean(num_waypoints):.1f}, "
        f"frames: {np.mean(num_frames):.1f}, "
        f"ratio: {np.mean(num_frames) / np.mean(num_waypoints):.2f}"
    )


if __name__ == "__main__":
    main()
